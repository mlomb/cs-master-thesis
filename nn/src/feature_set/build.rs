use super::FeatureSet;

pub fn build_feature_set(name: &str) -> Box<dyn FeatureSet> {
    use crate::feature_set::fs_axes::Axes::*;
    use crate::feature_set::fs_axes::*;

    match name {
        "half-piece" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { first_axis: Vertical, second_axis: Some(Horizontal) }],
        }),
        "hv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { first_axis: Horizontal, second_axis: Some(Vertical) }],
        }),
        "h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: None },
                AxesBlock { first_axis: Vertical, second_axis: None, },
            ],
        }),
        "hv+h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: Some(Vertical) },
                AxesBlock { first_axis: Horizontal, second_axis: None },
                AxesBlock { first_axis: Vertical, second_axis: None, },
            ],
        }),
        "d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { first_axis: Diagonal1, second_axis: Some(Diagonal2) }],
        }),
        "d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Diagonal1, second_axis: None },
                AxesBlock { first_axis: Diagonal2, second_axis: None }
            ],
        }),
        "hv+d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: Some(Vertical) },
                AxesBlock { first_axis: Diagonal1, second_axis: Some(Diagonal2) },
            ],
        }),
        "h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: None },
                AxesBlock { first_axis: Vertical, second_axis: None, },
                AxesBlock { first_axis: Diagonal1, second_axis: None },
                AxesBlock { first_axis: Diagonal2, second_axis: None }
            ],
        }),
        "hv+h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: Some(Vertical) },
                AxesBlock { first_axis: Horizontal, second_axis: None },
                AxesBlock { first_axis: Vertical, second_axis: None, },
                AxesBlock { first_axis: Diagonal1, second_axis: None },
                AxesBlock { first_axis: Diagonal2, second_axis: None }
            ],
        }),
        "hd1+vd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Horizontal, second_axis: Some(Diagonal1) },
                AxesBlock { first_axis: Vertical, second_axis: Some(Diagonal2) },
            ],
        }),
        "vd1+hd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { first_axis: Vertical, second_axis: Some(Diagonal1) },
                AxesBlock { first_axis: Horizontal, second_axis: Some(Diagonal2) },
            ],
        }),
        _ => panic!("Unknown NNUE model feature set: {}", name),
    }
}
