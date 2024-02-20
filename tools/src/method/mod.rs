pub mod eval;
pub mod pqr;
pub mod stats_topk;

use shakmaty::Chess;
use std::{fs::File, io};

pub trait Method {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()>;
    //fn read_sample()
}
