use crate::feature_set::blocks::{king::KingBlock, pairwise::PairwiseBlock, FeatureBlocks};

use super::FeatureSet;

/// Build a feature set from its name
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
        "hv+ph" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
        ]),
        "hv+pv" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        ]),
        "h+v+ph+pv" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
            FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        ]),
        "hv+ph+pv" => FeatureSet::sum_of(vec![
            FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        ]),
        //
        "kr" => FeatureSet::sum_of(vec![
            // -
            FeatureBlocks::KingBlock(KingBlock::new()),
        ]),
        "kr+pv+ph" => FeatureSet::sum_of(vec![
            // -
            FeatureBlocks::KingBlock(KingBlock::new()),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
            FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        ]),
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

        hv_ph: "hv+ph",
        hv_pv: "hv+pv",
        h_v_ph_pv: "h+v+ph+pv",
        hv_ph_pv: "hv+ph+pv",

        kr: "kr",
    }
}
