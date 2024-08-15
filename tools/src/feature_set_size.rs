use clap::Args;
use nn::feature_set::build::build_feature_set;
use std::error::Error;

#[derive(Args)]
pub struct FeatureSetSizeCommand {
    /// The feature set
    #[arg(long, value_name = "feature-set")]
    feature_set: String,
}

pub fn feature_set_size(cmd: FeatureSetSizeCommand) -> Result<(), Box<dyn Error>> {
    let feature_set = build_feature_set(&cmd.feature_set);

    println!("{}", feature_set.num_features());

    Ok(())
}
