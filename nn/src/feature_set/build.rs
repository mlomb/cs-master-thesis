use crate::feature_set::blocks::FeatureBlocks;

use super::FeatureSet;

pub fn build_feature_set(name: &str) -> FeatureSet {
    use crate::feature_set::axis::Axis;
    use crate::feature_set::blocks::axes::*;

    match name {
        "h+v" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
        ]),
        "d1+d2" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal1)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal2)),
        ]),
        "h+v+d1+d2" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal1)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal2)),
        ]),
        "hv" => FeatureSet::sum_of(vec![
            // Piece
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
        ]),
        "hv+h+v" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
        ]),
        "hv+d1+d2" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal1)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal2)),
        ]),
        "hv+h+v+d1+d2" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal1)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal2)),
        ]),
        /*
        "khv" => Box::new(AxesFeatureSet {
            #[rustfmt::skip]
            blocks: vec![
                AxesBlock { axes: vec![King, Horizontal, Vertical], incl_king: false },
            ],
        }),
        */
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
                fs_correctness_checks(&build_feature_set($value));
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
    }
}
