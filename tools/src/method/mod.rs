pub mod eval;
pub mod pqr;

use shakmaty::Chess;
use std::{fs::File, io};

pub trait TrainingMethod {
    fn write_sample(&mut self, file: &mut File, positions: &Vec<Chess>) -> io::Result<()>;
    //fn read_sample()
}
