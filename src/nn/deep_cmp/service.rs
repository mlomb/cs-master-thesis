use crate::core::position::Position;
use ndarray::{ArrayD, Axis};
use ort::Session;
use std::hash::Hash;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::{cmp::Ordering, collections::HashMap};

use super::encoding::TensorEncodeable;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pair<P>(P, P);

pub struct DeepCmpService<Position> {
    session: Session,
    inferences: AtomicUsize,
    hs: Arc<Mutex<HashMap<Pair<Position>, Ordering>>>,

    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl<P> DeepCmpService<P>
where
    P: Position + TensorEncodeable + Eq + Hash + Send + Sync,
{
    pub fn new(session: Session) -> Self {
        Self {
            session: session,
            inferences: AtomicUsize::new(0),
            hs: Arc::new(Mutex::new(HashMap::new())),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    pub fn compare(&self, left: &P, right: &P) -> Ordering {
        let pair = Pair(left.clone(), right.clone());

        if let Some(ordering) = self.hs.lock().unwrap().get(&pair) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return *ordering;
        }
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inferences
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let mut input_tensor = ArrayD::zeros(P::input_shape());
        P::encode_input(left, right, &mut input_tensor.view_mut());

        // add batch size axis
        input_tensor = input_tensor.insert_axis(Axis(0));

        let outputs = self
            .session
            .run(ort::inputs![input_tensor].unwrap())
            .unwrap();

        let output_tensor = outputs[0].extract_tensor::<f32>().unwrap();
        let res = P::decode_output(&output_tensor.view().index_axis(Axis(0), 0));

        self.hs.lock().unwrap().insert(pair, res);
        res
    }
}
