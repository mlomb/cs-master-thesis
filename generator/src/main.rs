pub mod sample;
pub mod visitor;

use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::BufferedReader;
use std::{
    fs::File,
    io::{self, Seek},
};
use visitor::GameVisitor;

fn main() -> io::Result<()> {
    let files = vec![
        //"D:/lichess/lichess_db_standard_rated_2023-12.pgn.zst",
        "D:/lichess/lichess_db_standard_rated_2023-11.pgn.zst",
        "D:/lichess/lichess_db_standard_rated_2023-10.pgn.zst",
        "D:/lichess/lichess_db_standard_rated_2023-09.pgn.zst",
    ];
    const MAX_OUT_SIZE: u64 = 1024 * 1024 * 512; // 512 MB

    let bar_style = ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Parsed {human_pos} @ {per_sec}] {msg} {prefix}",
        )
        .unwrap();

    let mut count = 0;
    let mut out_part = 0;
    let mut out_file: Option<File> = None;

    for path in files {
        let file = File::open(&path)?;
        let uncompressed: Box<dyn io::Read> = if path.ends_with(".zst") {
            Box::new(zstd::Decoder::new(file)?)
        } else {
            Box::new(file)
        };

        let bar = ProgressBar::new_spinner()
            .with_style(bar_style.clone())
            .with_prefix(path);

        let mut reader = BufferedReader::new(uncompressed);
        let mut visitor = GameVisitor::new();

        while let Ok(Some(sample)) = reader.read_game(&mut visitor) {
            bar.inc(1);

            if let Some(sample) = sample {
                if out_file.is_none() || out_file.as_ref().unwrap().stream_position()? > MAX_OUT_SIZE {
                    out_part += 1;
                    out_file = Some(File::create(format!("../data/dataset/{}.bin", out_part))?);
                }
                sample.write_to(&mut out_file.as_mut().unwrap())?;

                count += 1;
                bar.set_message(format!("[Accepted {}] [Writing {}.bin]", count, out_part));
            }
        }
        bar.finish();
    }

    Ok(())
}
