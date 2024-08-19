use super::FeatureSet;

pub fn build_feature_set(name: &str) -> Box<dyn FeatureSet> {
    use crate::feature_set::fs_axes::Axes::*;
    use crate::feature_set::fs_axes::*;

    match name {
        "half-piece" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Vertical, Horizontal], incl_king: true }],
        }),
        "hv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true }],
        }),
        "h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
            ],
        }),
        "hv+h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true },
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
            ],
        }),
        "d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Diagonal1, Diagonal2], incl_king: true }],
        }),
        "d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Diagonal2], incl_king: true }
            ],
        }),
        "hv+d1d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true },
                AxesBlock { axes: vec![Diagonal1, Diagonal2], incl_king: true },
            ],
        }),
        "h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
                AxesBlock { axes: vec![Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Diagonal2], incl_king: true }
            ],
        }),
        "hv+h+v+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true },
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
                AxesBlock { axes: vec![Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Diagonal2], incl_king: true }
            ],
        }),
        "hd1+vd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Vertical, Diagonal2], incl_king: true },
            ],
        }),
        "vd1+hd2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Vertical, Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Horizontal, Diagonal2], incl_king: true },
            ],
        }),
        "khv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![King, Horizontal, Vertical], incl_king: false },
            ],
        }),
        _ => panic!("Unknown NNUE model feature set: {}", name),
    }
}
