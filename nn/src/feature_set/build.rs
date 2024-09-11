use super::FeatureSet;

pub fn build_feature_set(name: &str) -> Box<dyn FeatureSet> {
    use crate::feature_set::fs_axes::Axes::*;
    use crate::feature_set::fs_axes::*;

    match name {
        "h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
            ],
        }),
        "d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Diagonal1], incl_king: true },
                AxesBlock { axes: vec![Diagonal2], incl_king: true }
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
        // Piece
        "hv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true }],
        }),
        "hv+h+v" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true },
                AxesBlock { axes: vec![Horizontal], incl_king: true },
                AxesBlock { axes: vec![Vertical], incl_king: true },
            ],
        }),
        "hv+d1+d2" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![Horizontal, Vertical], incl_king: true },
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
        "khv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![King, Horizontal, Vertical], incl_king: false },
            ],
        }),
        _ => panic!("Unknown NNUE model feature set: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use crate::feature_set::checks::fs_correctness_checks;

    use super::*;

    macro_rules! fs_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                fs_correctness_checks(build_feature_set($value).as_ref());
            }
        )*
        }
    }

    fs_tests! {
        h_v: "h+v",
        d1_d2: "d1+d2",
        h_v_d1_d2: "h+v+d1+d2",
        hv: "hv",
        hv_h_v: "hv+h+v",
        hv_d1_d2: "hv+d1+d2",
        hv_h_v_d1_d2: "hv+h+v+d1+d2",
        khv: "khv",
    }
}
