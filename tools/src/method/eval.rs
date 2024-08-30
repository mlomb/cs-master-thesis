use super::{Sample, SampleEncoder};
use crate::pos_encoding::{encode_position, encoded_size};
use nn::feature_set::FeatureSet;
use std::io::Write;

pub struct EvalEncoding;

impl SampleEncoder for EvalEncoding {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize {
        1 * encoded_size(feature_set)
    }

    fn y_size(&self) -> usize {
        4
    }

    fn write_sample(
        &self,
        sample: &Sample,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    ) {
        encode_position(&sample.position, feature_set, write_x);

        // side to move score
        write_y
            .write_all(&f32::to_le_bytes(sample.score as f32))
            .unwrap();
    }
}
