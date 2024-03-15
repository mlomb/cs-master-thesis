mod accumulator;
mod crelu;
mod linear;
mod tensor;

use self::accumulator::NnueAccumulator;
use self::crelu::crelu_16;
use self::linear::linear;
use self::tensor::Tensor;
use shakmaty::Color;
use std::fs::File;
use std::io::{Cursor, Read};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const FT: usize = 256;
const L1: usize = 32;
const L2: usize = 32;
const LO: usize = 1;

struct NnueModel {
    accum: NnueAccumulator,
    buffer1: Tensor<i8>,

    linear1_weight: Tensor<i8>,
    linear1_bias: Tensor<i32>,
    linear1_buffer: Tensor<i32>,
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

        let linear1_weight: Tensor<i8> = Tensor::from_cursor(&mut cursor, 2 * FT * L1)?; // row-major
        let linear1_bias: Tensor<i32> = Tensor::from_cursor(&mut cursor, L1)?;

        Ok(Self {
            accum,

            buffer1: Tensor::zeros(2 * FT),
            linear1_weight,
            linear1_bias,
            linear1_buffer: Tensor::zeros(L1),
        })
    }

    pub fn forward(&mut self, perspective: Color) {
        unsafe {
            let to_move_ft = self.accum.perspective(perspective).as_ptr();
            let not_to_move_ft = self.accum.perspective(perspective.other()).as_ptr();

            let (to_move, not_to_move) = self.buffer1.as_mut_slice().split_at_mut(FT);
            crelu_16(FT, to_move_ft, to_move.as_mut_ptr());
            crelu_16(FT, not_to_move_ft, not_to_move.as_mut_ptr());

            println!("to_move = {:?}", to_move);
            println!("not_to_move = {:?}", not_to_move);

            println!("self.linear1_weight = {:?}", self.linear1_weight);
            println!("self.linear1_bias = {:?}", self.linear1_bias);

            linear(
                2 * FT,
                L1,
                self.buffer1.as_ptr(),
                self.linear1_weight.as_ptr(),
                self.linear1_bias.as_ptr(),
                self.linear1_buffer.as_mut_ptr(),
            );

            println!("linear1 = {:?}", self.linear1_buffer);
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
                break;
            }
            ms += start.elapsed().as_millis();
            break;
        }

        println!("ms = {}", ms as f32 / 1000.0);
    }
}
