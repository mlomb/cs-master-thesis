use crate::games::connect4::*;
use ndarray::{ArrayD, Axis, IxDyn};

pub trait TensorEncodeable {
    fn encode(&self) -> ArrayD<f32>;

    fn concat(left: &ArrayD<f32>, right: &ArrayD<f32>) -> ArrayD<f32>;
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

    fn concat(left: &ArrayD<f32>, right: &ArrayD<f32>) -> ArrayD<f32> {
        assert_eq!(left.shape(), &[7, 6, 2]);
        assert_eq!(right.shape(), &[7, 6, 2]);

        let mut tensor = left.clone();
        tensor
            .append(Axis(2), right.view().into_shape(right.shape()).unwrap())
            .unwrap();

        assert_eq!(tensor.shape(), &[7, 6, 4]);

        tensor
    }
}
