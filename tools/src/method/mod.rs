pub mod eval;
pub mod pqr;
pub mod stats_topk;

use nn::feature_set::FeatureSet;
use shakmaty::Chess;
use std::io::BufReader;
use std::io::Read;
use std::{fs::File, io};

pub trait WriteSample {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()>;
}

pub trait ReadSample {
    fn read_sample(
        &mut self,
        file: &mut BufReader<File>,
        buffer: &mut [u64],
        feature_set: &Box<dyn FeatureSet>,
    ) -> usize;
}
