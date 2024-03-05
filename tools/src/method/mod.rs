pub mod eval;
pub mod pqr;
pub mod stats_topk;

use nn::feature_set::FeatureSet;
use shakmaty::Chess;
use std::io;
use std::io::BufRead;
use std::io::Write;

pub trait WriteSample {
    fn write_sample(&mut self, write: &mut dyn Write, positions: &Vec<Chess>) -> io::Result<()>;
}

pub trait ReadSample {
    fn sample_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;

    fn read_sample(
        &mut self,
        read: &mut dyn BufRead,
        write: &mut dyn Write,
        feature_set: &Box<dyn FeatureSet>,
    );
}
