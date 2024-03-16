mod crelu;
mod linear;
mod tensor;

use self::crelu::{crelu_16, crelu_32};
use self::linear::{linear, linear_partial_refresh, linear_partial_update};
use self::tensor::Tensor;
use shakmaty::Color;
use std::fs::File;
use std::io::{Cursor, Read};

const FT: usize = 256;
const L1: usize = 32;
const L2: usize = 32;

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
    // Feature transformer accumulator
    accumulator: [Tensor<i16>; 2], // indexed by color
    ft_weight: Tensor<i16>,
    ft_bias: Tensor<i16>,

    // Linear layers
    linear1: LinearLayer,
    linear2: LinearLayer,
    linear_out: LinearLayer,
}

impl NnueModel {
    pub fn load(num_features: usize, model_path: &str) -> std::io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(buffer);

        Ok(Self {
            accumulator: [Tensor::zeros(FT), Tensor::zeros(FT)],
            ft_weight: Tensor::from_cursor(&mut cursor, num_features * FT)?,
            ft_bias: Tensor::from_cursor(&mut cursor, FT)?,

            linear1: LinearLayer::new(&mut cursor, 2 * FT, L1),
            linear2: LinearLayer::new(&mut cursor, L1, L2),
            linear_out: LinearLayer::new(&mut cursor, L2, 1),
        })
    }

    pub fn refresh(&mut self, active_features: &[u16], perspective: Color) {
        unsafe {
            linear_partial_refresh(
                self.ft_weight.len() / FT,
                FT,
                active_features,
                self.ft_weight.as_ptr(),
                self.ft_bias.as_ptr(),
                self.accumulator[perspective as usize].as_mut_ptr(),
            );
        }
    }

    pub fn update(&mut self, added_features: &[u16], removed_features: &[u16], perspective: Color) {
        unsafe {
            linear_partial_update(
                self.ft_weight.len() / FT,
                FT,
                added_features,
                removed_features,
                self.ft_weight.as_ptr(),
                self.accumulator[perspective as usize].as_mut_ptr(),
            );
        }
    }

    pub fn forward(&mut self, perspective: Color) -> i32 {
        unsafe {
            let to_move_ft = self.accumulator[perspective as usize].as_ptr();
            let not_to_move_ft = self.accumulator[perspective.other() as usize].as_ptr();

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
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    /// Make sure that refreshing and updating (adding/removing features) gives the same output
    #[test]
    fn test_update() {
        let mut nnue_model = NnueModel::load(
            768,
            "/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn",
        )
        .unwrap();

        let all_features = vec![
            668, 324, 624, 690, 473, 204, 97, 336, 568, 148, 667, 212, 199, 265, 760, 356, 501,
            457, 604, 213, 636, 544, 86, 208, 281, 209, 581, 639, 328, 431, 120, 363, 425, 300, 67,
            338, 579, 66, 582, 78, 482, 456, 30, 635, 33, 31, 39, 77, 299, 487, 629, 516, 375, 451,
            511, 234, 361, 494, 692, 404, 754, 764, 519, 254, 483, 211, 210, 84, 239, 409, 54, 720,
            512, 109, 587, 362, 734, 396, 528, 10, 192, 448, 174, 428, 181, 748, 155, 309, 65, 331,
            137, 350, 81, 468, 405, 470, 250, 490, 220, 76, 548, 290, 72, 244, 394, 620, 63, 716,
            659, 314, 118, 728, 49, 662, 411, 605, 227, 168, 513, 7, 196, 275, 23,
        ];
        let initial_features = vec![
            490, 254, 362, 225, 3, 279, 516, 482, 667, 309, 468, 748, 331, 652, 336, 425, 726, 133,
            49, 720, 577, 568, 208, 629, 581, 537, 210, 209, 409, 492, 636, 635, 457, 760, 491, 6,
            196, 220, 63, 523, 76, 66, 483, 234, 0, 118, 199, 754, 33, 411, 604, 227, 299, 109,
            683, 333, 404, 155, 375, 448, 456, 212, 587, 511, 73, 239, 507, 690, 484, 639, 394,
            668, 701, 47, 77, 755, 728, 513, 137, 519, 547, 579, 7, 405, 692, 660, 451, 723, 204,
            605, 27, 30, 31, 659, 716, 300, 65, 528, 149, 501, 662, 226, 260, 192, 651, 356, 624,
            548, 266, 67, 290, 78, 72, 23, 79, 338, 81, 86, 328, 631, 702, 419, 616,
        ];
        nnue_model.refresh(initial_features.as_slice(), Color::White);
        nnue_model.update(&[213, 512, 97, 120], &[631, 702, 419, 616], Color::White);
        nnue_model.update(&[275, 428, 265, 466], &[6, 728, 683, 723], Color::White);
        nnue_model.update(&[363, 640, 431, 350], &[0, 577, 491, 660], Color::White);
        nnue_model.update(&[494, 734, 553, 544], &[547, 226, 79, 651], Color::White);
        nnue_model.update(&[54, 569, 582, 281], &[73, 701, 466, 260], Color::White);
        nnue_model.update(&[764, 148, 174, 84], &[279, 225, 569, 149], Color::White);
        nnue_model.update(&[396, 473, 314, 250], &[133, 507, 492, 266], Color::White);
        nnue_model.update(&[244, 211, 620, 39], &[484, 523, 640, 27], Color::White);
        nnue_model.update(&[181, 487, 168, 470], &[553, 755, 652, 537], Color::White);
        nnue_model.update(&[361, 324, 728, 10], &[47, 726, 333, 3], Color::White);
        // copy accumulator from white to black
        nnue_model.accumulator[Color::Black as usize]
            .as_mut_slice()
            .copy_from_slice(nnue_model.accumulator[Color::White as usize].as_slice());
        let output_with_updates = nnue_model.forward(Color::White);

        nnue_model.refresh(all_features.as_slice(), Color::White);
        nnue_model.refresh(all_features.as_slice(), Color::Black);
        let output_with_refresh = nnue_model.forward(Color::White);

        assert_eq!(output_with_updates, output_with_refresh);
    }

    #[test]
    fn test_nn3() {
        let mut nnue_model = NnueModel::load(
            768,
            "/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn",
        )
        .unwrap();

        let all_features = vec![
            668, 324, 624, 690, 473, 204, 97, 336, 568, 148, 667, 212, 199, 265, 760, 356, 501,
            457, 604, 213, 636, 544, 86, 208, 281, 209, 581, 639, 328, 431, 120, 363, 425, 300, 67,
            338, 579, 66, 582, 78, 482, 456, 30, 635, 33, 31, 39, 77, 299, 487, 629, 516, 375, 451,
            511, 234, 361, 494, 692, 404, 754, 764, 519, 254, 483, 211, 210, 84, 239, 409, 54, 720,
            512, 109, 587, 362, 734, 396, 528, 10, 192, 448, 174, 428, 181, 748, 155, 309, 65, 331,
            137, 350, 81, 468, 405, 470, 250, 490, 220, 76, 548, 290, 72, 244, 394, 620, 63, 716,
            659, 314, 118, 728, 49, 662, 411, 605, 227, 168, 513, 7, 196, 275, 23,
        ];
        nnue_model.refresh(all_features.as_slice(), Color::White);
        nnue_model.refresh(all_features.as_slice(), Color::Black);

        let mut ms = 0;

        for _ in 0..1000 {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                nnue_model.forward(Color::White);
                break;
            }
            ms += start.elapsed().as_millis();
        }

        println!("model output = {}", nnue_model.forward(Color::White));
        println!("ms = {}", ms as f32 / 1000.0);
    }
}
