use crate::method::pqr::PQREncoding;
use crate::method::Sample;
use crate::method::{eval::EvalEncoding, SampleEncoder};
use crate::plain_format::PlainReader;
use clap::{Args, ValueEnum};
use crossbeam::channel::{bounded, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use nn::feature_set::build::build_feature_set;
use rand::seq::SliceRandom;
use rand::Rng;
use shakmaty::Position;
use shared_memory::ShmemConf;
use std::fs::metadata;
use std::thread;
use std::{
    error::Error,
    io::{self, Cursor, Read, Seek, Write},
};

/// A single batch, passed between threads
struct BatchData {
    x: Vec<u8>,
    y: Vec<u8>,
}

#[derive(ValueEnum, Clone)]
pub enum Method {
    /// Score evaluations (most likely from Stockfish) are used as target
    Eval,
    /// Given a transition P → Q in a game, R is selected from a legal move from P while R != Q
    PQR,
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
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// The shared memory file to write the samples. Must have the correct size, otherwise it will panic.
    /// If it is not provided, it will measure the performance of the batch loader without writing to shared memory
    /// (with a progress bar). Useful for benchmarking.
    #[arg(long)]
    shmem: Option<String>,

    /// Input .plain file to read samples from
    #[arg(long, required = true)]
    input: String,

    /// Loop the input file indefinitely. Used for training
    #[arg(long)]
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

    /// Number of batch threads to use
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Random skipping. Probability of skipping a sample.
    #[arg(long, default_value = "0.0")]
    random_skipping: f32,
}

pub fn batch_loader(cmd: BatchLoaderCommand) -> Result<(), Box<dyn Error>> {
    // true length of the file
    let file_length = metadata(&cmd.input)
        .expect("Unable to query input size")
        .len();

    assert!(
        cmd.input_offset + cmd.input_length <= file_length,
        "Length is out of bounds"
    );

    // length of the subfile to read from
    let readable_length = if cmd.input_length == 0 {
        file_length - cmd.input_offset
    } else {
        cmd.input_length
    };

    // actual length to read from each thread
    let thread_length = ((readable_length as f64) / cmd.threads as f64).floor() as u64;

    // open shared memory buffer
    let mut shmem = if let Some(shmem) = &cmd.shmem {
        Some(ShmemConf::new().os_id(&shmem).open()?)
    } else {
        None
    };

    // perf testing bar
    let bar = if shmem.is_none() {
        ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_spinner()
        .template(
            "{spinner:.green} [Elapsed {elapsed_precise}] [Batches {human_pos} @ {per_sec}] {msg}",
        )
        .unwrap())
    } else {
        ProgressBar::hidden()
    };

    // keep up to 8GB worth of batches ready in memory (this is WAY too much, but I have too much memory)
    let batch_buffer = if let Some(shmem) = &shmem {
        8 * 1024 * 1024 * 1024 / (shmem.len() as usize)
    } else {
        256
    };
    let (batch_sender, batch_receiver) = bounded(batch_buffer);

    // start batch threads
    for i in 0..cmd.threads {
        let cmd = cmd.clone();
        let offset = cmd.input_offset + i as u64 * thread_length;
        let batch_sender = batch_sender.clone();

        thread::spawn(move || build_samples_thread(cmd, offset, thread_length, batch_sender));
    }

    drop(batch_sender);

    // loop to write batches
    loop {
        // receive a batch from another thread
        if let Ok(batch) = batch_receiver.recv() {
            bar.inc(1);

            let x_batch_size = batch.x.len();
            let y_batch_size = batch.y.len();

            if let Some(shmem) = &mut shmem {
                assert!(
                    shmem.len() == x_batch_size + y_batch_size,
                    "unexpected shared memory size"
                );

                // wait for the reader to signal that it has finished copying the data of the last batch (1 byte)
                io::stdin().read_exact(&mut [0])?;

                // copy new batch to shared memory
                // this means rewind the cursors, then read to shmem
                let shmem_slice = unsafe { shmem.as_slice_mut() };
                shmem_slice[..x_batch_size].copy_from_slice(&batch.x);
                shmem_slice[x_batch_size..].copy_from_slice(&batch.y);

                // send a signal to the consumer (1 byte)
                io::stdout().write_all(&[64])?;
                io::stdout().flush()?;
            }
        } else {
            // wait for the last signal
            io::stdin().read_exact(&mut [0])?;

            break;
        }
    }

    bar.finish();

    Ok(())
}

fn build_samples_thread(
    cmd: BatchLoaderCommand,
    offset: u64,
    length: u64,
    batch_sender: Sender<BatchData>,
) {
    let mut rng = rand::thread_rng();
    let feature_set = build_feature_set(&cmd.feature_set);
    let method = build_method(&cmd);

    let mut samples_buffer = Vec::<Sample>::with_capacity(256 * 256 * 16); // 1 M

    let x_batch_size = cmd.batch_size * method.x_size(&feature_set);
    let y_batch_size = cmd.batch_size * method.y_size();

    let mut x_cursor = Cursor::new(vec![0u8; x_batch_size]);
    let mut y_cursor = Cursor::new(vec![0u8; y_batch_size]);

    // loop input file
    loop {
        let mut reader = PlainReader::open_with_limits(&cmd.input, offset, length)
            .expect("can't open input file");

        // loop file once
        loop {
            // loop to read samples
            while let Ok(Some(samples)) = reader.read_samples_line() {
                for sample in samples {
                    // smart fen skipping
                    if smart_filter(&sample, &cmd.method) {
                        continue;
                    }

                    // random fen skipping
                    if cmd.random_skipping > 0.0 && rng.gen::<f32>() < cmd.random_skipping {
                        continue;
                    }

                    // worst case we discard a line, no biggie
                    if samples_buffer.len() < samples_buffer.capacity() {
                        samples_buffer.push(sample);
                    }
                }

                // break out
                if samples_buffer.len() >= samples_buffer.capacity() - 100 {
                    break;
                }
            }

            if samples_buffer.is_empty() {
                // EOF
                break;
            }

            // shuffle!
            samples_buffer.shuffle(&mut rng);

            // loop to write batches
            while let Some(sample) = samples_buffer.pop() {
                method.write_sample(&sample, &mut x_cursor, &mut y_cursor, &feature_set);

                // is batch full?
                if x_cursor.position() >= x_batch_size as u64 {
                    assert!(x_cursor.position() == x_batch_size as u64);
                    assert!(y_cursor.position() == y_batch_size as u64);

                    // send batch to main thread
                    batch_sender
                        .send(BatchData {
                            x: x_cursor.clone().into_inner(),
                            y: y_cursor.clone().into_inner(),
                        })
                        .unwrap();

                    // reset buffers
                    x_cursor.rewind().unwrap();
                    y_cursor.rewind().unwrap();
                }
            }

            // Note: unused samples are kept in the buffer
        }

        if !cmd.input_loop {
            break;
        }
    }
}

fn smart_filter(sample: &Sample, method: &Method) -> bool {
    match method {
        Method::Eval => {
            sample.score.abs() > 5000 || // skip very extreme scores
            sample.bestmove.is_capture() || // skip capture moves
            sample.position.is_check() // skip check positions
        }
        Method::PQR => sample.score.abs() > 5000,
    }
}

fn build_method(cmd: &BatchLoaderCommand) -> Box<dyn SampleEncoder> {
    match cmd.method {
        Method::PQR => Box::new(PQREncoding {}),
        Method::Eval => Box::new(EvalEncoding {}),
    }
}
