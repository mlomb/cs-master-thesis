use super::evaluator::DeepCmpEvaluator;
use crate::core::value::Value;
use std::cmp::Ordering;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pair<P>(P, P);

#[derive(Clone)]
pub struct DeepCmpValue<P> {
    pub evaluator: *const DeepCmpEvaluator<P>,
    pub position: P,
    pub point_of_view: bool,
}

impl<P> Value for DeepCmpValue<P>
where
    P: Clone,
{
    fn compare(&self, other: &DeepCmpValue<P>) -> Ordering {
        Ordering::Equal
        /*
        let pair = Pair(self, other.clone());

        if let Some(ordering) = self.hs.get(&pair) {
            self.hits += 1;
            return *ordering;
        }

        self.misses += 1;

        // insert one axis: batch size
        let b1 = left.pos.encode().insert_axis(Axis(0));
        let b2 = right.pos.encode().insert_axis(Axis(0));

        let mut input = b1;
        input
            .append(Axis(3), b2.view().into_shape(b2.shape()).unwrap())
            .unwrap();

        let outputs = self.session.run(ort::inputs![input].unwrap()).unwrap();
        let data = outputs[0]
            .extract_tensor::<f32>()
            .unwrap()
            .view()
            .t()
            .into_owned();

        println!("{}", data);

        self.inferences += 1;

        let res = if data[[0, 0]] < data[[1, 0]] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        };

        self.hs.insert(pair, res);
        res
        */
    }

    fn opposite(&self) -> DeepCmpValue<P> {
        DeepCmpValue {
            evaluator: self.evaluator,
            position: self.position.clone(),
            point_of_view: !self.point_of_view,
        }
    }
}
