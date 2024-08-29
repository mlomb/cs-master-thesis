use crate::format::plain::PlainReader;
use crate::method::pqr::PQR;
use crate::method::{eval::EvalRead, SampleEncoder};
use clap::{Args, ValueEnum};
use core::slice::SlicePattern;
use memmap2::MmapOptions;
use nn::feature_set::build::build_feature_set;
use shared_memory::ShmemConf;
use std::io::SeekFrom;
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, Cursor, Read, Seek, Write},
};

#[derive(ValueEnum, Clone)]
pub enum Method {
    /// Given a transition P → Q in a game, R is selected from a legal move from P while R != Q.
    PQR,
    /// Score evaluations (most likely from Stockfish) are used as target
    Eval,
}

#[derive(Args, Clone)]
pub struct BatchLoaderCommand {
    /// Method to use
    #[arg(long)]
    method: Method,

    /// The feature set to use
    #[arg(long)]
    feature_set: String,

    /// Number of samples in one batch
    #[arg(long, default_value = "8192")]
    batch_size: usize,

    /// The shared memory file to write the samples.
    /// Must have the correct size, otherwise it will panic
    #[arg(long)]
    shmem: String,

    /// List of .csv files to read samples from.
    /// CSVs must have the format: `fen,score,bestmove`
    #[arg(long, required = true)]
    input: String,

    /// Loop the input file indefinitely
    #[arg(long, default_value = "true")]
    input_loop: bool,

    // Offset to start reading from.
    // The first sample will be read after skipping a line from this offset.
    // So if the offset points to the middle of a sample, it will be skipped
    #[arg(long, default_value = "0")]
    input_offset: u64,

    // Length to read from the input file, starting from the offset.
    // If 0, it will read until the end of the file.
    // Length restricts where to start reading a sample from, this means
    // that if a sample start before length and end after length, it will be read (past length)
    #[arg(long, default_value = "0")]
    input_length: u64,
}

pub fn batch_loader(cmd: BatchLoaderCommand) -> Result<(), Box<dyn Error>> {
    let feature_set = build_feature_set(&cmd.feature_set);
    let method = build_method(&cmd);

    // initialize batch cursors
    // this is where the batches are written to
    // before being copied to shmem
    let x_batch_size = cmd.batch_size * method.x_size(&feature_set);
    let y_batch_size = cmd.batch_size * method.y_size();
    let mut x_cursor = Cursor::new(vec![0u8; x_batch_size]);
    let mut y_cursor = Cursor::new(vec![0u8; y_batch_size]);

    // open shared memory buffer
    let mut shmem = ShmemConf::new().os_id(&cmd.shmem).open()?;
    let shmem_slice = unsafe { shmem.as_slice_mut() };

    assert!(
        shmem_slice.len() == x_batch_size + y_batch_size,
        "unexpected shared memory size"
    );

    // loop to write batches
    loop {
        let file = File::open(&cmd.input).expect("can't open input file");
        let mut reader = PlainReader::new(file);
        eprintln!("Opened file {}", cmd.input);

        while let Ok(Some(samples)) = reader.read_samples_line() {
            for sample in samples {
                // add sample to batch
                method.write_sample(&sample, &mut x_cursor, &mut y_cursor, &feature_set);

                // is batch full?
                if x_cursor.position() >= x_batch_size as u64 {
                    assert!(x_cursor.position() == x_batch_size as u64);
                    assert!(y_cursor.position() == y_batch_size as u64);

                    // wait for the reader to signal that it has finished copying the data of the last batch (1 byte)
                    if let Err(_) = io::stdin().read_exact(&mut [0]) {
                        // reader disconnected
                        return Ok(());
                    }

                    // copy new batch to shared memory
                    // this means rewind the cursors, then read to shmem
                    x_cursor.rewind().unwrap();
                    y_cursor.rewind().unwrap();
                    x_cursor.read_exact(&mut shmem_slice[..x_batch_size])?;
                    y_cursor.read_exact(&mut shmem_slice[x_batch_size..])?;

                    // send a signal to the consumer (1 byte)
                    io::stdout().write_all(&[64])?;
                    io::stdout().flush()?;

                    // reset buffers
                    x_cursor.rewind().unwrap();
                    y_cursor.rewind().unwrap();
                }
            }
        }
    }
}

fn build_method(cmd: &BatchLoaderCommand) -> Box<dyn SampleEncoder> {
    match cmd.method {
        Method::PQR => Box::new(PQR {}),
        Method::Eval => Box::new(EvalRead {}),
    }
}

/*

    assert!(
        cmd.input_offset + cmd.input_length <= mmap.len() as u64,
        "offset + length is out of bounds"
    );

let pos = reader.position();

// break in EOF
if !cmd.input_loop && pos >= cmd.input_offset + cmd.input_length {
    // no more samples to read
    // discard the current batch (aka drop_last)
    // exiting the thread will disconnect the channel
    eprintln!("Thread finished");
    return Ok(());
}

// seek to the correct position
// in the first iteration or if we are out of bounds (and did not hit EOF above)
if pos < cmd.input_offset || pos >= cmd.input_offset + cmd.input_length {
    reader.seek(SeekFrom::Start(cmd.input_offset)).unwrap();

    // make sure we skip to the first valid sample
    reader.skip_until(b'\n').unwrap();
}

// random sample skipping
if rand::random::<f64>() < 0.3 {
    // skip a line (sample)
    reader.skip_until(b'\n').unwrap();
} else {
    // read a sample
    method.read_sample(&mut reader, &mut x_cursor, &mut y_cursor, &feature_set);
}

*/
