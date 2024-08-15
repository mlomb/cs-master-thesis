use super::FeatureSet;

pub fn build_feature_set(name: &str) -> Box<dyn FeatureSet> {
    use crate::feature_set::fs_axes::Axes::*;
    use crate::feature_set::fs_axes::*;

    match name {
        "half-piece" => Box::new(AxesFeatureSet {
            blocks: vec![AxesBlock {
                first_axis: Vertical,
                second_axis: Some(Horizontal),
            }],
        }),
        "fs-hv" => Box::new(AxesFeatureSet {
            blocks: vec![AxesBlock {
                first_axis: Horizontal,
                second_axis: Some(Vertical),
            }],
        }),
        "fs-d1d2" => Box::new(AxesFeatureSet {
            blocks: vec![AxesBlock {
                first_axis: Diagonal1,
                second_axis: Some(Diagonal2),
            }],
        }),
        _ => panic!("Unknown NNUE model feature set: {}", name),
    }
}
