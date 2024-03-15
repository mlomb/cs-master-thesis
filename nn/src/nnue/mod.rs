pub mod accumulator;

use std::fmt::Debug;
use std::fs::File;
use std::io::{Cursor, Read};
use std::{
    alloc::{alloc, dealloc, Layout},
    fmt::Formatter,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use shakmaty::Color;

use self::accumulator::NnueAccumulator;

/// Memory is aligned to 32 bits
pub struct Tensor<T> {
    layout: Layout,
    data: *mut T,
}

impl<T> Tensor<T> {
    fn from_cursor(cursor: &mut Cursor<Vec<u8>>, len: usize) -> std::io::Result<Self> {
        let mut data = vec![0u8; len * std::mem::size_of::<T>()];
        cursor.read_exact(data.as_mut_slice())?;

        Ok(Self::from_slice(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const T, len)
        }))
    }

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

struct FeatureTransformer {
    weight: Tensor<i16>,
    bias: Tensor<i16>,
}

struct SequentialModel {
    layers: Vec<FullyConnectedQuantized>,
    buffers: Vec<Tensor<i8>>,
}

impl SequentialModel {
    pub fn load(model_path: &str) -> Self {
        /*
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
        */

        let mut layers = vec![];
        let mut buffers = vec![];

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

struct NnueModel {
    accum: NnueAccumulator,
}

impl NnueModel {
    pub fn load(num_features: usize, model_path: &str) -> std::io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(buffer);

        const FT: usize = 256;
        const L1: usize = 512;
        const L2: usize = 32;
        const L3: usize = 1;

        let w: Tensor<i16> = Tensor::from_cursor(&mut cursor, num_features * FT)?;
        let b: Tensor<i16> = Tensor::from_cursor(&mut cursor, FT)?;
        let mut accum = NnueAccumulator::new(w, b);

        let active_features = vec![
            668, 324, 624, 690, 473, 204, 97, 336, 568, 148, 667, 212, 199, 265, 760, 356, 501,
            457, 604, 213, 636, 544, 86, 208, 281, 209, 581, 639, 328, 431, 120, 363, 425, 300, 67,
            338, 579, 66, 582, 78, 482, 456, 30, 635, 33, 31, 39, 77, 299, 487, 629, 516, 375, 451,
            511, 234, 361, 494, 692, 404, 754, 764, 519, 254, 483, 211, 210, 84, 239, 409, 54, 720,
            512, 109, 587, 362, 734, 396, 528, 10, 192, 448, 174, 428, 181, 748, 155, 309, 65, 331,
            137, 350, 81, 468, 405, 470, 250, 490, 220, 76, 548, 290, 72, 244, 394, 620, 63, 716,
            659, 314, 118, 728, 49, 662, 411, 605, 227, 168, 513, 7, 196, 275, 23,
        ];
        accum.refresh(&active_features, Color::White);
        println!("accum = {:?}", accum.perspective(Color::White).as_slice());

        Ok(Self { accum })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn3() {
        let mut nnue_model = NnueModel::load(
            768,
            "/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn",
        );

        unreachable!();

        let model = SequentialModel::load(
            "/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn",
        );

        /*
        [20, 3, 4, -16, 13, -41, -33, -32, -26, -21, -27, -19, 0, -40, -62, 0, 18, 20, -1, -28, -7, 24, 25, 19, 26, -1, -9, -3, -12, -19, -9, -45, -37, 13, -24, 22, -27, 22, -32, -14, -8, 11, 43, -28, -19, -14, 4, -2, -36, -5, 12, -6, -49, 40, 6, -6, 46, -14, -69, 27, 11, 4, -13, -2, -7, -24, 11, -14, 7, 9, -82, 22, 11, -26, 71, 3, -60, -19, 17, 5, 10, -23, -6, -37, 79, -71, 37, -3, -33, -14, -21, -1, 28, -7, -48, 45, -27, -32, -11, 2, -11, 25, -42, 26, -1, -24, 17, 19, 35, 22, 11, 42, 4, -15, -29, -56, -1, -21, -36, -72, 10, -27, 5, 31, 0, -9, -28, -21, -17, 12, -10, -65, -9, -13, -28, 7, 55, 2, -9, 26, 20, -9, -73, 1, -30, 4, -22, -22, 11, -17, -3, 5, 1, -47, -5, -1, -26, 23, -47, -16, -4, -30, -14, 36, -50, 34, 16, -10, -32, -1, 13, -13, -13, 38, 22, 12, 50, 37, -9, 41, 1, -52, -20, -1, -22, 21, 25, 39, 73, 13, 1, 9, -23, 32, 4, 44, 13, 17, 19, 12, 35, 4, 32, -14, 32, 46, 7, -4, -15, 2, -34, 24, 20, 7, -17, -13, -47, -1, -5, 38, -28, -9, -16, 25, -1, 48, -5, 27, -39, 16, 27, -43, 0, -8, 0, -72, -3, 2, -47, 24, 8, -37, 8, 4, -21, -2, -42, -26, -15, -22, 46, 24, 13, -18, -24, -28]
        [4, -31, 15, 7, -25, -19, 40, 26, 49, 3, 25, 23, -23, -13, 45, 20, 27, -57, 11, -18, 52, -56, 18, -27, 0, -43, -22, -16, -6, 23, -3, 18, 44, 33, 6, -9, -20, 38, -15, -7, -50, -35, 7, 44, -5, -2, -46, -29, -29, -43, 11, -25, -11, -65, 34, 1, 8, -53, 7, -22, -4, 5, -47, -2, -3, 5, 24, -24, 73, -51, -12, 18, -7, 16, -8, 16, -40, 21, -17, -18, 15, 32, -43, -1, 47, 52, -22, 4, -14, -25, 31, 51, 5, 8, -1, -13, 37, 2, -13, -13, 15, 9, -16, 7, -53, 13, 8, -32, -17, 23, -9, -27, 2, 41, -15, -1, 34, -9, 5, -10, 14, 23, 16, 9, -28, -29, 41, 15, 3, 1, 41, 4, -26, 18, 32, 30, -17, -17, 45, 41, -20, -21, 23, 16, 24, -24, 11, 32, -14, 3, 13, -53, 7, -31, -18, 48, 15, -19, -1, -16, -20, 51, 5, 30, 11, 15, 8, 3, 53, 0, 13, -79, -85, 29, -15, -21, -8, 20, -4, -9, -8, -15, -6, 1, -8, 40, 12, -17, -27, 8, -41, -17, -27, 46, -17, 4, -40, -19, -8, -2, -21, -6, 10, -30, 0, -15, -34, 25, -15, 7, -11, -18, 16, -9, 61, -3, -4, 4, -44, 7, 23, 53, 16, -14, 41, -20, 38, 40, 61, 20, 22, -30, -17, 37, -4, -47, -22, -21, -11, -52, 10, -22, -29, 9, -39, -30, 23, 30, -20, -51, -15, -33, -12, -31, -15, 64]
         */

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
