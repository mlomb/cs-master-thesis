use super::FeatureSet;
use crate::feature_set::blocks::{mobility::MobilityBlock, pairwise::PairwiseBlock, FeatureBlocks};

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
        // axes
        "h" => FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Horizontal)),
        "v" => FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Vertical)),
        "d1" => FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal1)),
        "d2" => FeatureBlocks::AxesBlock(AxesBlock::single(Axis::Diagonal2)),
        "hv" => FeatureBlocks::AxesBlock(AxesBlock::product(Axis::Horizontal, Axis::Vertical)),
        // pairwise
        "ph" => FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Horizontal)),
        "pv" => FeatureBlocks::PairwiseBlock(PairwiseBlock::new(Axis::Vertical)),
        // mobility
        "mb" => FeatureBlocks::MobilityBlock(MobilityBlock::new()),

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
        hv: "hv",
        hv_h_v: "hv+h+v",
        hv_d1_d2: "hv+d1+d2",
        hv_h_v_d1_d2: "hv+h+v+d1+d2",

        hv_ph: "hv+ph",
        hv_pv: "hv+pv",
        h_v_ph_pv: "h+v+ph+pv",
        hv_ph_pv: "hv+ph+pv",

        mb: "mb",
        hv_mb: "hv+mb",
    }
}
