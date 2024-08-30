use crate::plain_format::{PlainReader, PlainWriter};
use clap::Args;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use std::error::Error;

#[derive(Args)]
pub struct ConvertCommand {
    /// Input file
    #[arg(long, required = true)]
    input: String,

    /// Output file (will be overwritten)
    #[arg(long, required = true)]
    output: String,
}

pub fn convert(cmd: ConvertCommand) -> Result<(), Box<dyn Error>> {
    println!("Input file: {}", cmd.input);
    println!("Output file: {}", cmd.output);

    let mut reader = PlainReader::open(&cmd.input).expect("can't open input file");
    let mut writer = PlainWriter::open(&cmd.output).expect("can't open output file");

    let bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Positions {human_pos} @ {per_sec}] {msg}",
        )
        .unwrap());

    while let Ok(Some(samples)) = reader.read_samples_line() {
        bar.inc(samples.len() as u64);

        for sample in samples {
            writer.write_sample(&sample)?;

            if bar.position() % 100_000 == 0 {
                bar.set_message(format!(
                    "[Read {} Written {}]",
                    HumanBytes(reader.bytes_read().unwrap()),
                    HumanBytes(writer.bytes_written().unwrap())
                ));
            }
        }
    }

    bar.set_message(format!(
        "[Read {} Written {}]",
        HumanBytes(reader.bytes_read().unwrap()),
        HumanBytes(writer.bytes_written().unwrap())
    ));
    bar.finish();

    Ok(())
}
