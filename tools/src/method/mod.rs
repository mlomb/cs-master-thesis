pub mod eval;
pub mod pqr;

use nn::feature_set::FeatureSet;
use std::io::BufRead;
use std::io::Write;

pub trait Sample {
    /// Size of the input tensor
    fn x_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;

    /// Size of the target tensor
    fn y_size(&self) -> usize;

    /// Read a single sample from a read buffer, and write the encoding into the write buffers.
    /// It may not read a sample, not write anything (e.g. skipping capture positions).
    fn read_sample(
        &self,
        read: &mut dyn BufRead,
        write_x: &mut dyn Write,
        write_y: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    );
}
