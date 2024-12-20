use super::crelu::{crelu_16, crelu_32};
use super::linear::{linear, linear_partial_refresh, linear_partial_update};
use super::tensor::Tensor;
use crate::feature_set::build::build_feature_set;
use crate::feature_set::FeatureSet;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufRead, Cursor, Read};

/// A linear layer in the network
struct LinearLayer<W, B> {
    num_inputs: usize,
    num_outputs: usize,

    weight: Tensor<W>, // num_inputs * num_outputs
    bias: Tensor<B>,   // num_outputs

    // this buffer is used by the previous layer to prepare the data for this layer (to avoid allocations)
    // note: this is unused in the feature transform layer :(
    input_buffer: Tensor<W>,
    // this buffer is used just after computing the linear layer, before applying the activation
    intermediate_buffer: Tensor<B>,
}

impl<W, B> LinearLayer<W, B> {
    fn new(cursor: &mut Cursor<&[u8]>, num_inputs: usize, num_outputs: usize) -> Self {
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

impl LinearLayer<i8, i32> {
    /// Forward pass of a hidden layer, using the input buffer and storing the result in the intermediate buffer.
    unsafe fn forward_hidden(&self) {
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

/// Neural Network Update Efficient (NNUE)
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md
pub struct NnueModel {
    feature_set: FeatureSet,

    pub arch: String,
    pub params: usize,

    linear1: LinearLayer<i16, i16>,
    linear2: LinearLayer<i8, i32>,
    linear_out: LinearLayer<i8, i32>,
}

impl NnueModel {
    /// Loads a model from a .nn file
    pub fn load(model_path: &str) -> io::Result<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        Self::from_memory(&buffer)
    }

    /// Loads a model from a .nn file in memory.
    /// Format description can be found in `scripts/lib/serialize.py`
    pub fn from_memory(buffer: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor::new(buffer);

        // read feature set name
        let mut str_buffer = Vec::new();
        cursor.read_until(0, &mut str_buffer).unwrap();
        str_buffer.pop(); // remove null byte
        let feature_set_str = std::str::from_utf8(&str_buffer).unwrap();

        // read network sizes
        let num_features = cursor.read_u32::<LittleEndian>().unwrap() as usize;
        let num_l1 = cursor.read_u32::<LittleEndian>().unwrap() as usize;
        let num_l2 = cursor.read_u32::<LittleEndian>().unwrap() as usize;
        let num_out = 1;

        let feature_set = build_feature_set(feature_set_str);
        assert_eq!(num_features, feature_set.num_features() as usize);

        Ok(Self {
            feature_set,
            linear1: LinearLayer::new(&mut cursor, num_features, num_l1),
            linear2: LinearLayer::new(&mut cursor, 2 * num_l1, num_l2),
            linear_out: LinearLayer::new(&mut cursor, num_l2, num_out),

            arch: format!("({}[{}]→{})x2→{}→1", feature_set_str, num_features, num_l1, num_l2),
            params: 
                // l1
                num_features * num_l1 + num_l1 +
                // l2
                (2 * num_l1) * num_l2 + num_l2 +
                // out
                num_out * num_l2 + num_l2 + num_out,
        })
    }

    /// Refreshes the accumulator with the given features (slow)
    pub fn refresh_accumulator(&self, accumulator: &Tensor<i16>, active_features: &[u16]) {
        unsafe {
            linear_partial_refresh(
                self.linear1.num_inputs,
                self.linear1.num_outputs,
                active_features,
                self.linear1.weight.as_ptr(),
                self.linear1.bias.as_ptr(),
                accumulator.as_mut_ptr(),
            );
        }
    }

    /// Updates the accumulator with the given added and removed features (rows).
    /// It does not know if the features were already active. It is the caller's responsibility to
    /// ensure that rows are not added/removed twice.
    pub fn update_accumulator(
        &self,
        accumulator: &Tensor<i16>,
        added_features: &[u16],
        removed_features: &[u16],
    ) {
        unsafe {
            linear_partial_update(
                self.linear1.num_inputs,
                self.linear1.num_outputs,
                added_features,
                removed_features,
                self.linear1.weight.as_ptr(),
                accumulator.as_mut_ptr(),
            );
        }
    }

    /// Forward pass of the network, skipping the first layer and instead taking the accumulated values for each side
    pub fn forward(&self, to_move_accum: &Tensor<i16>, not_to_move_accum: &Tensor<i16>) -> i32 {
        unsafe {
            // layer 1 already computed in accumulator
            let to_move_accum = to_move_accum.as_ptr();
            let not_to_move_accum = not_to_move_accum.as_ptr();
            let l1_out = self.linear1.num_outputs; // size of each accumulator

            // split the input buffer of the layer 2 into two parts
            let (to_move, not_to_move) = self
                .linear2
                .input_buffer
                .as_mut_slice()
                .split_at_mut(l1_out);

            // fill the input to the layer 2 doing the crelu of the two accumulators (output of the first layer)
            crelu_16(l1_out, to_move_accum, to_move.as_mut_ptr());
            crelu_16(l1_out, not_to_move_accum, not_to_move.as_mut_ptr());

            // forward layer 2
            self.linear2.forward_hidden();
            crelu_32(
                self.linear2.num_outputs,
                self.linear2.intermediate_buffer.as_ptr(),
                self.linear_out.input_buffer.as_mut_ptr(),
            );

            // forward output layer
            self.linear_out.forward_hidden();

            self.linear_out.intermediate_buffer.as_slice()[0]
        }
    }

    pub fn get_feature_set(&self) -> &FeatureSet {
        &self.feature_set
    }

    pub fn get_num_features(&self) -> usize {
        self.linear1.num_outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Make sure that refreshing and updating (adding/removing features) gives the same output
    #[test]
    fn test_update() {
        let nnue_model = NnueModel::from_memory(include_bytes!("../../../models/best.nn")).unwrap();

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

        let accum_updates = Tensor::zeros(nnue_model.get_num_features());
        nnue_model.refresh_accumulator(&accum_updates, initial_features.as_slice());
        nnue_model.update_accumulator(&accum_updates, &[213, 512, 97, 120], &[631, 702, 419, 616]);
        nnue_model.update_accumulator(&accum_updates, &[275, 428, 265, 466], &[6, 728, 683, 723]);
        nnue_model.update_accumulator(&accum_updates, &[363, 640, 431, 350], &[0, 577, 491, 660]);
        nnue_model.update_accumulator(&accum_updates, &[494, 734, 553, 544], &[547, 226, 79, 651]);
        nnue_model.update_accumulator(&accum_updates, &[54, 569, 582, 281], &[73, 701, 466, 260]);
        nnue_model.update_accumulator(&accum_updates, &[764, 148, 174, 84], &[279, 225, 569, 149]);
        nnue_model.update_accumulator(&accum_updates, &[396, 473, 314, 250], &[133, 507, 492, 266]);
        nnue_model.update_accumulator(&accum_updates, &[244, 211, 620, 39], &[484, 523, 640, 27]);
        nnue_model.update_accumulator(&accum_updates, &[181, 487, 168, 470], &[553, 755, 652, 537]);
        nnue_model.update_accumulator(&accum_updates, &[361, 324, 728, 10], &[47, 726, 333, 3]);

        let accum_refresh = Tensor::zeros(nnue_model.get_num_features());
        nnue_model.refresh_accumulator(&accum_refresh, &all_features.as_slice());

        assert_eq!(accum_updates.as_slice(), accum_refresh.as_slice()); // thus forward gives the same output
    }
}
