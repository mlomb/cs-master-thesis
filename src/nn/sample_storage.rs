pub struct SampleStorage<T>
where
    T: Sized,
{
    samples: Vec<T>,
    capacity: usize,
}
