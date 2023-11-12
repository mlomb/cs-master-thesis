use std::cmp::Ordering;

use crate::games::connect4::*;
use ndarray::{s, ArrayViewD, ArrayViewMut3, ArrayViewMutD, Dimension, IxDyn};

pub trait TensorEncodeable {
    /// Returns the shape of the input tensor.
    fn input_shape() -> IxDyn;

    /// Returns the size of the input tensor (product of all dimensions)
    fn input_shape_size() -> usize {
        Self::input_shape().as_array_view().iter().product()
    }

    /// Encodes the input tensor for the given positions.
    fn encode_input(left: &Self, right: &Self, tensor: &mut ArrayViewMutD<f32>);

    /// Decodes the output tensor into the ordering
    fn decode_output(tensor: &ArrayViewD<f32>) -> Ordering;
}

// TODO: move this to another file
fn encode_connect4(board: &Connect4, tensor: &mut ArrayViewMut3<f32>) {
    assert_eq!(tensor.shape(), &[7, 6, 2]);
    let who_plays = board.0.who_plays();

    // clear tensor
    // TODO: clear in another place
    tensor.fill(0.0);

    for row in 0..ROWS {
        for col in 0..COLS {
            if let Some(at) = board.0.get_at(row, col) {
                tensor[[col, row, if at == who_plays { 0 } else { 1 }]] = 1.0;
            }
        }
    }
}

impl TensorEncodeable for Connect4 {
    fn input_shape() -> IxDyn {
        IxDyn(&[7, 6, 4])
    }

    fn encode_input(left: &Self, right: &Self, tensor: &mut ArrayViewMutD<f32>) {
        assert_eq!(tensor.shape(), &[7, 6, 4]);

        encode_connect4(left, &mut tensor.slice_mut(s![.., .., ..2]));
        encode_connect4(right, &mut tensor.slice_mut(s![.., .., 2..]));
    }

    fn decode_output(tensor: &ArrayViewD<f32>) -> Ordering {
        if tensor[[0]] > tensor[[1]] {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}
