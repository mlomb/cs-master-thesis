use crate::feature_set::FeatureSet;

use super::crelu::{crelu_16, crelu_32};
use super::linear::{linear, linear_partial_refresh, linear_partial_update};
use super::tensor::Tensor;
use shakmaty::Color;
use std::fs::File;
use std::io::{BufRead, Cursor, Read};

const FT: usize = 256;
const L1: usize = 32;
const L2: usize = 32;

struct LinearLayer<W, B> {
    num_inputs: usize,
    num_outputs: usize,

    weight: Tensor<W>,
    bias: Tensor<B>,

    // this buffer is used by the previous layer to prepare the data for this layer
    // note: this is unused in the feature transform layer :(
    input_buffer: Tensor<W>,
    // this buffer is used just after computing the linear layer, before applying the activation
    intermediate_buffer: Tensor<B>,
}

impl<W, B> LinearLayer<W, B> {
    fn new(cursor: &mut Cursor<Vec<u8>>, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,

            weight: Tensor::from_cursor(cursor, num_inputs * num_outputs).unwrap(),
            bias: Tensor::from_cursor(cursor, num_outputs).unwrap(),

            input_buffer: Tensor::zeros(num_inputs),
            intermediate_buffer: Tensor::zeros(num_outputs),
        }
    }
}

/// Neural Network Update Efficent (NNUE)
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md
pub struct NnueModel {
    feature_set: Box<dyn FeatureSet>,

    accumulator: [Tensor<i16>; 2], // indexed by perspective (color)

    feature_transform: LinearLayer<i16, i16>,
    linear1: LinearLayer<i8, i32>,
    linear2: LinearLayer<i8, i32>,
    linear_out: LinearLayer<i8, i32>,
}

impl NnueModel {
    pub fn load(model_path: &str) -> std::io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(buffer);

        let mut str_buffer = Vec::new();
        cursor.read_until(0, &mut str_buffer).unwrap();
        str_buffer.pop(); // remove null byte
        let feature_set_str = std::str::from_utf8(&str_buffer).unwrap();

        let feature_set: Box<dyn FeatureSet> = match feature_set_str {
            "basic" => Box::new(crate::feature_set::basic::Basic::new()),
            _ => panic!("Unknown NNUE model feature set: {}", feature_set_str),
        };
        let num_features = feature_set.num_features();

        println!(
            "info string Loading NNUE model with feature set: {}",
            feature_set_str
        );

        Ok(Self {
            feature_set,
            accumulator: [Tensor::zeros(FT), Tensor::zeros(FT)],
            feature_transform: LinearLayer::new(&mut cursor, num_features, FT),
            linear1: LinearLayer::new(&mut cursor, 2 * FT, L1),
            linear2: LinearLayer::new(&mut cursor, L1, L2),
            linear_out: LinearLayer::new(&mut cursor, L2, 1),
        })
    }

    pub fn refresh_accumulator(&mut self, active_features: &[u16], perspective: Color) {
        unsafe {
            linear_partial_refresh(
                self.feature_transform.num_inputs,
                self.feature_transform.num_outputs,
                active_features,
                self.feature_transform.weight.as_ptr(),
                self.feature_transform.bias.as_ptr(),
                self.accumulator[perspective as usize].as_mut_ptr(),
            );
        }
    }

    pub fn update_accumulator(
        &mut self,
        added_features: &[u16],
        removed_features: &[u16],
        perspective: Color,
    ) {
        unsafe {
            linear_partial_update(
                self.feature_transform.num_inputs,
                self.feature_transform.num_outputs,
                added_features,
                removed_features,
                self.feature_transform.weight.as_ptr(),
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

            Self::forward_hidden(&self.linear1);
            crelu_32(
                L1,
                self.linear1.intermediate_buffer.as_ptr(),
                self.linear2.input_buffer.as_mut_ptr(),
            );

            Self::forward_hidden(&self.linear2);
            crelu_32(
                L2,
                self.linear2.intermediate_buffer.as_ptr(),
                self.linear_out.input_buffer.as_mut_ptr(),
            );

            Self::forward_hidden(&self.linear_out);

            self.linear_out.intermediate_buffer.as_slice()[0]
        }
    }

    unsafe fn forward_hidden(layer: &LinearLayer<i8, i32>) {
        linear(
            layer.num_inputs,
            layer.num_outputs,
            layer.input_buffer.as_ptr(),
            layer.weight.as_ptr(),
            layer.bias.as_ptr(),
            layer.intermediate_buffer.as_mut_ptr(),
        );
    }

    pub fn get_feature_set(&self) -> &dyn FeatureSet {
        self.feature_set.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Make sure that refreshing and updating (adding/removing features) gives the same output
    #[test]
    fn test_update() {
        let mut nnue_model =
            NnueModel::load("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn")
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
        nnue_model.refresh_accumulator(initial_features.as_slice(), Color::White);
        nnue_model.update_accumulator(&[213, 512, 97, 120], &[631, 702, 419, 616], Color::White);
        nnue_model.update_accumulator(&[275, 428, 265, 466], &[6, 728, 683, 723], Color::White);
        nnue_model.update_accumulator(&[363, 640, 431, 350], &[0, 577, 491, 660], Color::White);
        nnue_model.update_accumulator(&[494, 734, 553, 544], &[547, 226, 79, 651], Color::White);
        nnue_model.update_accumulator(&[54, 569, 582, 281], &[73, 701, 466, 260], Color::White);
        nnue_model.update_accumulator(&[764, 148, 174, 84], &[279, 225, 569, 149], Color::White);
        nnue_model.update_accumulator(&[396, 473, 314, 250], &[133, 507, 492, 266], Color::White);
        nnue_model.update_accumulator(&[244, 211, 620, 39], &[484, 523, 640, 27], Color::White);
        nnue_model.update_accumulator(&[181, 487, 168, 470], &[553, 755, 652, 537], Color::White);
        nnue_model.update_accumulator(&[361, 324, 728, 10], &[47, 726, 333, 3], Color::White);
        // copy accumulator from white to black
        nnue_model.accumulator[Color::Black as usize]
            .as_mut_slice()
            .copy_from_slice(nnue_model.accumulator[Color::White as usize].as_slice());
        let output_with_updates = nnue_model.forward(Color::White);

        nnue_model.refresh_accumulator(all_features.as_slice(), Color::White);
        nnue_model.refresh_accumulator(all_features.as_slice(), Color::Black);
        let output_with_refresh = nnue_model.forward(Color::White);

        assert_eq!(output_with_updates, output_with_refresh);
    }

    #[test]
    fn test_nn3() {
        let mut nnue_model =
            NnueModel::load("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn")
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
        nnue_model.refresh_accumulator(all_features.as_slice(), Color::White);
        nnue_model.refresh_accumulator(all_features.as_slice(), Color::Black);

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
        println!("time for 1000 = {}", ms as f32 / 1000.0);
    }
}
