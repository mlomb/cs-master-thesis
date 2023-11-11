use crate::games::connect4::*;
use ndarray::{ArrayD, IxDyn};

pub trait TensorEncodeable {
    fn encode(&self) -> ArrayD<f32>;
}

// TODO: move this to another file
impl TensorEncodeable for Connect4 {
    fn encode(&self) -> ArrayD<f32> {
        let mut tensor = ArrayD::zeros(IxDyn(&[7, 6, 2]));
        let who_plays = self.0.who_plays();

        for row in 0..ROWS {
            for col in 0..COLS {
                if let Some(at) = self.0.get_at(row, col) {
                    tensor[[col, row, if at == who_plays { 0 } else { 1 }]] = 1.0;
                }
            }
        }

        tensor
    }
}
