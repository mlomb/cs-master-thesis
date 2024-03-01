pub mod eval;
pub mod pqr;
pub mod stats_topk;

use nn::feature_set::FeatureSet;
use shakmaty::Chess;
use std::io;
use std::io::BufReader;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;

pub trait WriteSample {
    fn write_sample(&mut self, write: &mut dyn Write, positions: &Vec<Chess>) -> io::Result<()>;
}

pub trait ReadSample {
    fn sample_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;

    fn read_sample(
        &mut self,
        read: &mut BufReader<Box<dyn Read>>,
        write: &mut Cursor<&mut [u8]>,
        feature_set: &Box<dyn FeatureSet>,
    );
}
