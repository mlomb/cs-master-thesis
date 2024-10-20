use super::FeatureSet;
use crate::feature_set::blocks::{
    all::AllBlock,
    mobility::{MobilityBitsetBlock, MobilityCountsBlock},
    pairwise::PairwiseBlock,
    FeatureBlocks,
};

/// Build a feature set from its name.
/// Feature set names are a list of feature block names separated by '+'.
pub fn build_feature_set(name: &str) -> FeatureSet {
    // split name by +
    // each is a block
    FeatureSet::sum_of(name.split('+').map(|block| get_block(block)).collect())
}

/// Get a feature block from its name
fn get_block(name: &str) -> FeatureBlocks {
    use crate::feature_set::axis::Axis;
    use crate::feature_set::blocks::axes::*;

    match name {
        // all
        "hv" => FeatureBlocks::AllBlock(AllBlock::new()), // legacy name
        "all" => FeatureBlocks::AllBlock(AllBlock::new()),
        // axes
        "h" => FeatureBlocks::AxesBlock(AxesBlock::new(Axis::Horizontal)),
        "v" => FeatureBlocks::AxesBlock(AxesBlock::new(Axis::Vertical)),
        "d1" => FeatureBlocks::AxesBlock(AxesBlock::new(Axis::Diagonal1)),
        "d2" => FeatureBlocks::AxesBlock(AxesBlock::new(Axis::Diagonal2)),
        // pairwise
        "ph" => FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
        "pv" => FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        // mobility
        "mb" => FeatureBlocks::MobilityBitsetBlock(MobilityBitsetBlock::new()),
        "mc" => FeatureBlocks::MobilityCountsBlock(MobilityCountsBlock::new()),

        _ => panic!("Unknown NNUE model feature block: {}", name),
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
        all: "all",
        all_h_v: "all+h+v",
        all_d1_d2: "all+d1+d2",
        all_h_v_d1_d2: "all+h+v+d1+d2",

        all_ph: "all+ph",
        all_pv: "all+pv",
        h_v_ph_pv: "h+v+ph+pv",
        all_ph_pv: "all+ph+pv",

        mb: "mb",
        mc: "mc",
        all_mb: "all+mb",
        all_mc: "all+mc",
    }
}
