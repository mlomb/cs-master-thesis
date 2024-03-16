mod accumulator;
mod crelu;
mod linear;
mod tensor;

use self::accumulator::NnueAccumulator;
use self::crelu::{crelu_16, crelu_32};
use self::linear::linear;
use self::tensor::Tensor;
use shakmaty::Color;
use std::fs::File;
use std::io::{Cursor, Read};

const FT: usize = 256;
const L1: usize = 32;
const L2: usize = 32;
const LO: usize = 1;

struct LinearLayer {
    num_inputs: usize,
    num_outputs: usize,

    // This buffer is used by the previous layer to prepare the data for this layer
    input_buffer: Tensor<i8>,
    // This buffer is used just after computing the linear layer, before applying the activation
    intermediate_buffer: Tensor<i32>,

    weight: Tensor<i8>,
    bias: Tensor<i32>,
}

impl LinearLayer {
    fn new(cursor: &mut Cursor<Vec<u8>>, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,

            input_buffer: Tensor::zeros(num_inputs),
            intermediate_buffer: Tensor::zeros(num_outputs),

            weight: Tensor::from_cursor(cursor, num_inputs * num_outputs).unwrap(),
            bias: Tensor::from_cursor(cursor, num_outputs).unwrap(),
        }
    }

    fn forward_linear(&self) {
        unsafe {
            linear(
                self.num_inputs,
                self.num_outputs,
                self.input_buffer.as_ptr(),
                self.weight.as_ptr(),
                self.bias.as_ptr(),
                self.intermediate_buffer.as_mut_ptr(),
            );
        }
    }

    fn forward_relu(&self, output: &Tensor<i8>) {
        unsafe {
            crelu_32(
                self.num_outputs,
                self.intermediate_buffer.as_ptr(),
                output.as_mut_ptr(),
            );
        }
    }
}

pub struct NnueModel {
    accum: NnueAccumulator,

    linear1: LinearLayer,
    linear2: LinearLayer,
    linear_out: LinearLayer,
    output: Tensor<i32>,
}

impl NnueModel {
    pub fn load(num_features: usize, model_path: &str) -> std::io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(buffer);

        let ft_w: Tensor<i16> = Tensor::from_cursor(&mut cursor, num_features * FT)?; // column-major
        let ft_b: Tensor<i16> = Tensor::from_cursor(&mut cursor, FT)?;
        let accum = NnueAccumulator::new(ft_w, ft_b);

        Ok(Self {
            accum,
            linear1: LinearLayer::new(&mut cursor, 2 * FT, L1),
            linear2: LinearLayer::new(&mut cursor, L1, L2),
            linear_out: LinearLayer::new(&mut cursor, L2, LO),
            output: Tensor::zeros(LO),
        })
    }

    pub fn forward(&mut self, perspective: Color) -> i32 {
        unsafe {
            let to_move_ft = self.accum.perspective(perspective).as_ptr();
            let not_to_move_ft = self.accum.perspective(perspective.other()).as_ptr();

            let (to_move, not_to_move) = self.linear1.input_buffer.as_mut_slice().split_at_mut(FT);
            crelu_16(FT, to_move_ft, to_move.as_mut_ptr());
            crelu_16(FT, not_to_move_ft, not_to_move.as_mut_ptr());

            self.linear1.forward_linear();
            self.linear1.forward_relu(&self.linear2.input_buffer);

            self.linear2.forward_linear();
            self.linear2.forward_relu(&self.linear_out.input_buffer);

            self.linear_out.forward_linear();

            return self.linear_out.intermediate_buffer.as_slice()[0];
        }
    }

    pub fn get_accumulator(&mut self) -> &mut NnueAccumulator {
        &mut self.accum
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
        )
        .unwrap();
        let active_features = vec![
            668, 324, 624, 690, 473, 204, 97, 336, 568, 148, 667, 212, 199, 265, 760, 356, 501,
            457, 604, 213, 636, 544, 86, 208, 281, 209, 581, 639, 328, 431, 120, 363, 425, 300, 67,
            338, 579, 66, 582, 78, 482, 456, 30, 635, 33, 31, 39, 77, 299, 487, 629, 516, 375, 451,
            511, 234, 361, 494, 692, 404, 754, 764, 519, 254, 483, 211, 210, 84, 239, 409, 54, 720,
            512, 109, 587, 362, 734, 396, 528, 10, 192, 448, 174, 428, 181, 748, 155, 309, 65, 331,
            137, 350, 81, 468, 405, 470, 250, 490, 220, 76, 548, 290, 72, 244, 394, 620, 63, 716,
            659, 314, 118, 728, 49, 662, 411, 605, 227, 168, 513, 7, 196, 275, 23,
        ];

        let mut ms = 0;

        for _ in 0..1000 {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                nnue_model
                    .get_accumulator()
                    .refresh(&active_features, Color::White);
                nnue_model
                    .get_accumulator()
                    .refresh(&active_features, Color::Black);
                nnue_model.forward(Color::White);
            }
            ms += start.elapsed().as_millis();
        }

        println!("model output = {}", nnue_model.forward(Color::White));
        println!("ms = {}", ms as f32 / 1000.0);
    }
}
