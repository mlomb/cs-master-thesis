use ndarray::ArrayD;

pub trait TensorEncodeable {
    fn encode(&self) -> ArrayD<f32>;
}
