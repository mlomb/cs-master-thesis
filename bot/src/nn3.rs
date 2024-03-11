use std::fmt::Debug;
use std::{
    alloc::{alloc, dealloc, Layout},
    fmt::Formatter,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use serde_json::Value;

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

impl<T: Debug> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?})", self.as_slice())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    // ReLU,
    ClippedReLU,
}

pub struct FullyConnectedQuantized {
    num_inputs: usize,
    num_outputs: usize,
    weights: Tensor<i8>,
    bias: Tensor<i32>,
    intermediate: Tensor<i32>,
    activation: Activation,
}

impl FullyConnectedQuantized {
    pub fn new(weights: Tensor<i8>, bias: Tensor<i32>, activation: Activation) -> Self {
        let num_inputs = weights.len() / bias.len();
        let num_outputs = bias.len();

        Self {
            num_inputs,
            num_outputs,
            weights,
            bias,
            intermediate: Tensor::zeros(num_outputs),
            activation,
        }
    }

    // https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#linear-layer-4
    fn forward(&self, input: &[i8], output: &mut [i8]) {
        debug_assert!(input.len() == self.num_inputs);
        debug_assert!(output.len() == self.num_outputs);

        // special case when O == 1
        if output.len() == 1 {
        } else {
            const REGISTER_WIDTH: usize = 256 / 8;

            debug_assert!(self.num_inputs % REGISTER_WIDTH == 0); // processing 32 elements at a time
            debug_assert!(self.num_outputs % 4 == 0); // processing 4 elements at a time

            let num_in_chunks: usize = self.num_inputs / REGISTER_WIDTH;
            let num_out_chunks: usize = self.num_outputs / 4;

            let input_ptr = input.as_ptr();
            let intermediate_ptr = self.intermediate.as_mut_ptr();
            let weights_ptr = self.weights.as_ptr();
            let bias_ptr = self.bias.as_ptr();

            unsafe {
                for i in 0..num_out_chunks {
                    let offset0 = (i * 4 + 0) * self.num_inputs;
                    let offset1 = (i * 4 + 1) * self.num_inputs;
                    let offset2 = (i * 4 + 2) * self.num_inputs;
                    let offset3 = (i * 4 + 3) * self.num_inputs;

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

            println!("intermediate = {:?}", self.intermediate.as_slice());

            assert!(self.activation == Activation::ClippedReLU);
            /*
            for i in 0..self.num_outputs {
                let int = unsafe { self.intermediate.as_slice().get_unchecked(i) };
                let out = unsafe { output.get_unchecked_mut(i) };
                *out = (*int).max(0) as i8;
            }
            */
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

struct SequentialModel {
    layers: Vec<FullyConnectedQuantized>,
    buffers: Vec<Tensor<i8>>,
}

impl SequentialModel {
    pub fn load(model_path: &str) -> Self {
        let content = std::fs::read_to_string(model_path).expect("file should be read");
        let json: serde_json::Value =
            serde_json::from_str(&content.as_str()).expect("file should be proper JSON");

        let layers_data = json
            .get("layers")
            .expect("layers missing from json")
            .as_array()
            .expect("layers must be an array");

        fn parse_tensor<T: TryFrom<i64>>(v: &Value) -> Tensor<T>
        where
            T::Error: Debug,
        {
            let values = v
                .as_array()
                .expect("values must be an array")
                .iter()
                .map(|x| x.as_i64().expect("a number"))
                .collect::<Vec<i64>>();

            Tensor::from_slice(
                &values
                    .iter()
                    .map(|x| T::try_from(*x).expect("outside bounds"))
                    .collect::<Vec<T>>(),
            )
        }

        let mut layers = vec![];
        let mut buffers = vec![];

        for layer_data in layers_data {
            let num_inputs = layer_data
                .get("num_inputs")
                .expect("num_inputs missing")
                .as_i64()
                .unwrap();
            let num_outputs = layer_data
                .get("num_outputs")
                .expect("num_outputs missing")
                .as_i64()
                .unwrap();
            let weights = parse_tensor(layer_data.get("weights").expect("weights missing"));
            let bias = parse_tensor(layer_data.get("bias").expect("bias missing"));
            let activation = match layer_data.get("activation") {
                Some(Value::String(s)) => match s.as_str() {
                    "clip" => Activation::ClippedReLU,
                    _ => panic!("unknown activation"),
                },
                _ => Activation::None,
            };

            let layer = FullyConnectedQuantized::new(weights, bias, activation);
            assert_eq!(layer.num_inputs, num_inputs as usize);
            assert_eq!(layer.num_outputs, num_outputs as usize);
            layers.push(layer);

            buffers.push(Tensor::zeros(num_outputs as usize));
        }

        Self { layers, buffers }
    }

    pub fn forward(&self, input: &Tensor<i8>) -> &Tensor<i8> {
        let mut input = input.as_slice();
        for i in 0..self.layers.len() {
            self.layers[i].forward(input, self.buffers[i].as_mut_slice());
            input = self.buffers[i].as_slice();
        }
        self.buffers.last().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs::File;

    #[test]
    fn test_nn3() {
        let model = SequentialModel::load("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/notebooks/runs/20240310_220627_eval_basic_4096/models/0.json");

        let mut ms = 0;

        for _ in 0..1000 {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                model.forward(&Tensor::from_slice(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 127,
                    127, 0, 0, 127, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0,
                    127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0,
                    127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 127, 127, 127, 127, 0, 127, 127, 127, 0, 0, 0, 0, 127, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 127, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 127, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]));

                break;
            }
            ms += start.elapsed().as_millis();
            break;
        }

        println!("ms = {}", ms as f32 / 1000.0);
    }
}
