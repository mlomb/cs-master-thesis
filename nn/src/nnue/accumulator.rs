use shakmaty::Color;

use super::{linear::linear_partial_init, Tensor};

pub struct NnueAccumulator {
    accum: [Tensor<i16>; 2],

    weight: Tensor<i16>,
    bias: Tensor<i16>,
}

impl NnueAccumulator {
    pub fn new(weight: Tensor<i16>, bias: Tensor<i16>) -> NnueAccumulator {
        NnueAccumulator {
            accum: [Tensor::zeros(bias.len()), Tensor::zeros(bias.len())],
            weight,
            bias,
        }
    }

    pub fn perspective(&mut self, color: Color) -> &mut Tensor<i16> {
        &mut self.accum[color as usize]
    }

    pub fn refresh(&mut self, active_features: &Vec<u16>, perspective: Color) {
        let num_outputs = self.bias.len();
        let num_inputs = self.weight.len() / num_outputs;

        unsafe {
            linear_partial_init(
                num_inputs,
                num_outputs,
                active_features.as_slice(),
                self.weight.as_ptr(),
                self.bias.as_ptr(),
                self.accum[perspective as usize].as_mut_ptr(),
            )
        }

        /*
        let weight_slice = self.weight.as_slice();
        let bias_slice = self.bias.as_slice();
        let accum_slice = self.accum[perspective as usize].as_mut_slice();

        // Load bias into accum
        accum_slice.copy_from_slice(bias_slice);

        // Add enabled features
        for &a in active_features {
            for i in 0..num_outs {
                accum_slice[i] += weight_slice[a as usize * num_outs + i];
            }
        }
        */
    }
}
