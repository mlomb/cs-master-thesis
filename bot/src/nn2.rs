use std::alloc::{alloc, dealloc, Layout};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// BLAS
use cblas_sys;
use cblas_sys::{CblasRowMajor, CblasTrans};

/// Memory is aligned to 32 bits
pub struct Tensor<T> {
    layout: Layout,
    data: *mut T,
}

impl<T> Tensor<T> {
    fn from_slice(from: &[T]) -> Self {
        let layout = Layout::from_size_align(from.len() * std::mem::size_of::<T>(), 32).unwrap();
        let data = unsafe { alloc(layout) } as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(from.as_ptr(), data, from.len());
        }
        Self { layout, data }
    }

    fn zeros(size: usize) -> Self {
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), 32).unwrap();
        let data = unsafe { alloc(layout) } as *mut T;
        unsafe {
            std::ptr::write_bytes(data, 0, size);
        }
        Self { layout, data }
    }

    fn as_ptr(&self) -> *const T {
        self.data as *const T
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.data
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len()) }
    }

    fn as_mut_slice(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len()) }
    }

    fn len(&self) -> usize {
        self.layout.size() / std::mem::size_of::<T>()
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    ReLU,
}

pub struct DenseLayer<const R: usize, const C: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
    activation: Activation,
}

impl<const R: usize, const C: usize> DenseLayer<R, C> {
    pub fn new(weights: Vec<f32>, bias: Vec<f32>, activation: Activation) -> Self {
        assert!(weights.len() == R * C);
        assert!(bias.len() == C);
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert!(input.len() == R);
        debug_assert!(output.len() == C);

        unsafe {
            cblas_sys::cblas_sgemv(
                CblasRowMajor,
                CblasTrans,
                R as i32,
                C as i32,
                1.0,
                self.weights.as_ptr(),
                C as i32,
                input.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }

        // use cols because weights are transposed

        match self.activation {
            Activation::None => {
                for i in 0..C {
                    let o = unsafe { output.get_unchecked_mut(i) };
                    let b = unsafe { self.bias.get_unchecked(i) };
                    *o += *b;
                }
            }
            Activation::ReLU => {
                for i in 0..C {
                    let o = unsafe { output.get_unchecked_mut(i) };
                    let b = unsafe { self.bias.get_unchecked(i) };
                    *o = (*o + *b).max(0.0);
                }
            }
        }
    }
}

pub struct DenseLayerQuantized<const I: usize, const O: usize> {
    weights: Tensor<i8>,
    bias: Tensor<i32>,
    intermediate: Tensor<i32>,
    activation: Activation,
}

impl<const I: usize, const O: usize> DenseLayerQuantized<I, O> {
    fn new(weights: Tensor<i8>, bias: Tensor<i32>, activation: Activation) -> Self {
        assert!(weights.len() == I * O);
        assert!(bias.len() == O);

        Self {
            weights,
            bias,
            intermediate: Tensor::zeros(O),
            activation,
        }
    }

    // https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#linear-layer-4
    fn forward(&self, input: &[i8], output: &mut [i8]) {
        debug_assert!(input.len() == I);
        debug_assert!(output.len() == O);

        // special case when O == 1
        if output.len() == 1 {
            let mut accum: i32 = self.bias.as_slice()[0];
            let weights = self.weights.as_slice();

            // do product
            for i in 0..I {
                accum += i32::from(weights[i]) * i32::from(input[i]);
            }

            println!("accum = {}", accum);
            println!("value = {}", (accum + 51) as f64 * 0.0320897251367569);

            assert!(self.activation == Activation::None);
            output[0] = accum.try_into().unwrap();
        } else {
            const REGISTER_WIDTH: usize = 256 / 8;

            debug_assert!(I % REGISTER_WIDTH == 0); // processing 32 elements at a time
            debug_assert!(O % 4 == 0); // processing 4 elements at a time

            let num_in_chunks: usize = I / REGISTER_WIDTH;
            let num_out_chunks: usize = O / 4;

            let input_ptr = input.as_ptr();
            let intermediate_ptr = self.intermediate.as_mut_ptr();
            let weights_ptr = self.weights.as_ptr();
            let bias_ptr = self.bias.as_ptr();

            unsafe {
                for i in 0..num_out_chunks {
                    let offset0 = (i * 4 + 0) * I;
                    let offset1 = (i * 4 + 1) * I;
                    let offset2 = (i * 4 + 2) * I;
                    let offset3 = (i * 4 + 3) * I;

                    let mut sum0 = _mm256_setzero_si256();
                    let mut sum1 = _mm256_setzero_si256();
                    let mut sum2 = _mm256_setzero_si256();
                    let mut sum3 = _mm256_setzero_si256();

                    for j in 0..num_in_chunks {
                        let inp =
                            _mm256_load_si256(input_ptr.add(j * REGISTER_WIDTH) as *const __m256i);

                        let w0 = _mm256_load_si256(
                            weights_ptr.add(offset0 + j * REGISTER_WIDTH) as *const __m256i
                        );
                        let w1 = _mm256_load_si256(
                            weights_ptr.add(offset1 + j * REGISTER_WIDTH) as *const __m256i
                        );
                        let w2 = _mm256_load_si256(
                            weights_ptr.add(offset2 + j * REGISTER_WIDTH) as *const __m256i
                        );
                        let w3 = _mm256_load_si256(
                            weights_ptr.add(offset3 + j * REGISTER_WIDTH) as *const __m256i
                        );

                        Self::m256_add_dpbusd_epi32(&mut sum0, inp, w0);
                        Self::m256_add_dpbusd_epi32(&mut sum1, inp, w1);
                        Self::m256_add_dpbusd_epi32(&mut sum2, inp, w2);
                        Self::m256_add_dpbusd_epi32(&mut sum3, inp, w3);
                    }

                    let bias = _mm_load_si128(bias_ptr.add(i * 4) as *const __m128i);

                    // -------------------------------------
                    // m256_haddx4
                    // https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#m256_haddx4
                    sum0 = _mm256_hadd_epi32(sum0, sum1);
                    sum2 = _mm256_hadd_epi32(sum2, sum3);

                    sum0 = _mm256_hadd_epi32(sum0, sum2);

                    let sum128lo = _mm256_castsi256_si128(sum0);
                    let sum128hi = _mm256_extracti128_si256(sum0, 1);

                    let mut outval = _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
                    outval = _mm_srai_epi32(outval, 6); // divide by 64
                                                        // -------------------------------------

                    _mm_store_si128(intermediate_ptr.add(i * 4) as *mut __m128i, outval);
                }
            }

            assert!(self.activation == Activation::ReLU);
            for i in 0..O {
                let int = unsafe { self.intermediate.as_slice().get_unchecked(i) };
                let out = unsafe { output.get_unchecked_mut(i) };
                *out = (*int).max(0) as i8;
            }
        }
    }

    /// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#m256_add_dpbusd_epi32
    #[inline]
    unsafe fn m256_add_dpbusd_epi32(acc: &mut __m256i, a: __m256i, b: __m256i) {
        #[cfg(target_feature = "avx512vnni,avx512vl")]
        {
            acc = _mm256_dpbusd_epi32(acc, a, b);
        }

        #[cfg(not(target_feature = "avx512vnni,avx512vl"))]
        {
            let mut product0 = _mm256_maddubs_epi16(a, b);
            let one = _mm256_set1_epi16(1);
            product0 = _mm256_madd_epi16(product0, one);
            *acc = _mm256_add_epi32(*acc, product0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_dense_layer() {
        unreachable!();
        /*
        let v: Value = serde_json::from_str(include_str!(
            //"/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/notebooks/tflite_model.json"
            "/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/notebooks/rq-mse-256-clipped-0.496.json"
        ))
        .unwrap();

        fn parse_weights(v: &Value) -> Tensor<i8> {
            Tensor::from_slice(
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i8)
                    .collect::<Vec<i8>>()
                    .as_slice(),
            )
        }
        fn parse_bias(v: &Value) -> Tensor<i32> {
            Tensor::from_slice(
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u8)
                    .collect::<Vec<u8>>()
                    .chunks(4)
                    .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                    .collect::<Vec<i32>>()
                    .as_slice(),
            )
        }

        let dense1 = DenseLayerQuantized::<768, 256>::new(
            parse_weights(&v["buffers"][9]["data"]),
            parse_bias(&v["buffers"][8]["data"]),
            Activation::ReLU,
        );
        let dense2 = DenseLayerQuantized::<256, 64>::new(
            parse_weights(&v["buffers"][7]["data"]),
            parse_bias(&v["buffers"][6]["data"]),
            Activation::ReLU,
        );
        let dense3 = DenseLayerQuantized::<64, 64>::new(
            parse_weights(&v["buffers"][5]["data"]),
            parse_bias(&v["buffers"][4]["data"]),
            Activation::ReLU,
        );
        let dense4 = DenseLayerQuantized::<64, 1>::new(
            parse_weights(&v["buffers"][3]["data"]),
            parse_bias(&v["buffers"][2]["data"]),
            Activation::None,
        );

        // print first 10
        {
            println!(
                "dense1.weights = {:?}",
                dense1
                    .weights
                    .as_slice()
                    .iter()
                    .take(10)
                    .collect::<Vec<_>>()
            );
            println!(
                "dense1.bias = {:?}",
                dense1.bias.as_slice().iter().take(10).collect::<Vec<_>>()
            );
            println!(
                "dense2.weights = {:?}",
                dense2
                    .weights
                    .as_slice()
                    .iter()
                    .take(10)
                    .collect::<Vec<_>>()
            );
            println!(
                "dense2.bias = {:?}",
                dense2.bias.as_slice().iter().take(10).collect::<Vec<_>>()
            );
            println!(
                "dense3.weights = {:?}",
                dense3
                    .weights
                    .as_slice()
                    .iter()
                    .take(10)
                    .collect::<Vec<_>>()
            );
            println!(
                "dense3.bias = {:?}",
                dense3.bias.as_slice().iter().take(10).collect::<Vec<_>>()
            );
            println!(
                "dense4.weights = {:?}",
                dense4
                    .weights
                    .as_slice()
                    .iter()
                    .take(10)
                    .collect::<Vec<_>>()
            );
            println!(
                "dense4.bias = {:?}",
                dense4.bias.as_slice().iter().take(10).collect::<Vec<_>>()
            );
        }

        let input = Tensor::from_slice(&[
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, 127, 127, 127, -128, -128, 127, 127, 127, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, 127, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, 127, -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128, 127, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, 127, 127, -128, 127,
            127, 127, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, 127, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128,
            -128, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            -128, -128, -128, -128,
        ]); // expect output=4.7
        let h1 = Tensor::zeros(256);
        let h2 = Tensor::zeros(64);
        let h3 = Tensor::zeros(64);
        let h4 = Tensor::zeros(1);

        let mut ms = 0;

        for _ in 0..1000 {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                dense1.forward(&input.as_slice(), &mut h1.as_mut_slice());
                println!("h1 = {:?}", h1.as_slice());
                dense2.forward(&h1.as_slice(), &mut h2.as_mut_slice());
                println!("h2 = {:?}", h2.as_slice());
                dense3.forward(&h2.as_slice(), &mut h3.as_mut_slice());
                println!("h3 = {:?}", h3.as_slice());
                dense4.forward(&h3.as_slice(), &mut h4.as_mut_slice());
                println!("h4 = {:?}", h4.as_slice());
                break;
            }
            ms += start.elapsed().as_millis();
            break;
        }

        println!("ms = {}", ms as f32 / 1000.0);
        println!("output = {:?}", h4.as_slice());
         */
    }
}
