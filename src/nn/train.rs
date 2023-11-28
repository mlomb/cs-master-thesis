use super::shmem::open_shmem;
use ndarray::{ArrayViewMut, ArrayViewMutD, Dimension, IxDyn};
use shared_memory::Shmem;

/// Interaction with the train.py script
pub struct Train<'a> {
    input_tensor: ArrayViewMutD<'a, f32>,
    output_tensor: ArrayViewMutD<'a, f32>,

    // shared memory with Python
    // we need to keep these alive, since we
    // use them in the tensors above
    #[allow(dead_code)]
    status_shmem: Shmem,
    #[allow(dead_code)]
    inputs_shmem: Shmem,
    #[allow(dead_code)]
    outputs_shmem: Shmem,
}

impl Train<'_> {
    pub fn new(
        batch_size: usize,
        mut input_shape_vec: Vec<usize>,
        mut output_shape_vec: Vec<usize>,
    ) -> Self {
        // add train batch axis
        input_shape_vec.insert(0, batch_size);
        output_shape_vec.insert(0, batch_size);

        // build ndarray dyn shapes
        let inputs_shape = IxDyn(&input_shape_vec);
        let outputs_shape = IxDyn(&output_shape_vec);

        let inputs_size = inputs_shape.as_array_view().iter().product::<usize>();
        let outputs_size = outputs_shape.as_array_view().iter().product::<usize>();

        // create shared memory buffers for training
        let status_shmem = open_shmem("train-status", 2 * 4).unwrap();
        let inputs_shmem = open_shmem("train-inputs", inputs_size * 4).unwrap();
        let outputs_shmem = open_shmem("train-outputs", outputs_size * 4).unwrap();

        Train {
            input_tensor: unsafe {
                ArrayViewMut::from_shape_ptr(inputs_shape, inputs_shmem.as_ptr() as *mut f32)
            },
            output_tensor: unsafe {
                ArrayViewMut::from_shape_ptr(outputs_shape, outputs_shmem.as_ptr() as *mut f32)
            },

            status_shmem,
            inputs_shmem,
            outputs_shmem,
        }
    }

    pub fn fit(&mut self, model: &str) {}
}
