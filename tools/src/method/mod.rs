pub mod eval;
pub mod pqr;
pub mod stats_topk;

use nn::feature_set::FeatureSet;
use shakmaty::Chess;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Cursor;
use std::{fs::File, io};

pub trait WriteSample {
    fn write_sample(
        &mut self,
        write: &mut BufWriter<File>,
        positions: &Vec<Chess>,
    ) -> io::Result<()>;
}

pub trait ReadSample {
    fn sample_size(&self, feature_set: &Box<dyn FeatureSet>) -> usize;

    fn read_sample(
        &mut self,
        read: &mut BufReader<File>,
        write: &mut Cursor<&mut [u8]>,
        feature_set: &Box<dyn FeatureSet>,
    );
}
