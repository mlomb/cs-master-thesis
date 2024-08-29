use crate::plain_format::{PlainReader, PlainWriter};
use clap::Args;
use indicatif::{HumanBytes, HumanCount, ProgressBar, ProgressStyle};
use std::{error::Error, fs::File, io::BufWriter};

#[derive(Args)]
pub struct ConvertCommand {
    #[arg(long, required = true)]
    input: String,

    #[arg(long, required = true)]
    output: String,
}

pub fn convert(cmd: ConvertCommand) -> Result<(), Box<dyn Error>> {
    println!("Input file: {}", cmd.input);
    println!("Output file: {}", cmd.output);

    let mut count = 0;
    let mut reader = PlainReader::new(File::open(&cmd.input).expect("can't open input file"));
    let mut writer = PlainWriter::new(File::create(cmd.output).expect("can't open output file"));

    let bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Positions {human_pos} @ {per_sec}] {msg}",
        )
        .unwrap());

    while let Ok(Some(samples)) = reader.read_samples_line() {
        bar.inc(samples.len() as u64);

        for sample in samples {
            writer.write_sample(sample)?;

            count += 1;
            if count % 100_000 == 0 {
                bar.set_message(format!(
                    "[Read {} Written {}]",
                    HumanBytes(reader.bytes_read()),
                    HumanBytes(writer.bytes_written().unwrap())
                ));
            }
        }
    }

    bar.set_message(format!(
        "[Read {} Written {}]",
        HumanBytes(reader.bytes_read()),
        HumanBytes(writer.bytes_written().unwrap())
    ));
    bar.finish();

    Ok(())
}
