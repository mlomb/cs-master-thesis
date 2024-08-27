pub mod eval;
pub mod pqr;

use nn::feature_set::FeatureSet;
use std::io::BufRead;
use std::io::Write;

pub trait ReadSample {
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;
    fn y_size(&self) -> usize;

    fn read_sample(
        &mut self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    );
}
