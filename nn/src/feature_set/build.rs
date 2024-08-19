use super::FeatureSet;

pub fn build_feature_set(name: &str) -> Box<dyn FeatureSet> {
    use crate::feature_set::fs_axes::Axes::*;
    use crate::feature_set::fs_axes::*;

    match name {
        "half-piece" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Vertical, Horizontal] }],
        }),
        "hv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Horizontal, Vertical] }],
        }),
        "h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal] },
                AxesBlock { axes: vec![Vertical] },
            ],
        }),
        "hv+h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical] },
                AxesBlock { axes: vec![Horizontal] },
                AxesBlock { axes: vec![Vertical] },
            ],
        }),
        "d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Diagonal1, Diagonal2] }],
        }),
        "d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Diagonal1] },
                AxesBlock { axes: vec![Diagonal2] }
            ],
        }),
        "hv+d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical] },
                AxesBlock { axes: vec![Diagonal1, Diagonal2] },
            ],
        }),
        "h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal] },
                AxesBlock { axes: vec![Vertical] },
                AxesBlock { axes: vec![Diagonal1] },
                AxesBlock { axes: vec![Diagonal2] }
            ],
        }),
        "hv+h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical] },
                AxesBlock { axes: vec![Horizontal] },
                AxesBlock { axes: vec![Vertical] },
                AxesBlock { axes: vec![Diagonal1] },
                AxesBlock { axes: vec![Diagonal2] }
            ],
        }),
        "hd1+vd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Diagonal1] },
                AxesBlock { axes: vec![Vertical, Diagonal2] },
            ],
        }),
        "vd1+hd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Vertical, Diagonal1] },
                AxesBlock { axes: vec![Horizontal, Diagonal2] },
            ],
        }),
        "king-hv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![King, Horizontal, Vertical] },
            ],
        }),
        _ => panic!("Unknown NNUE model feature set: {}", name),
    }
}
