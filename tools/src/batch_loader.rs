use crate::method::pqr::PQR;
use crate::method::{eval::EvalRead, ReadSample};
use clap::{Args, ValueEnum};
use crossbeam::channel::{bounded, Sender};
use nn::feature_set::build::build_feature_set;
use shared_memory::ShmemConf;
use std::fs::metadata;
use std::io::SeekFrom;
use std::time::Duration;
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, Cursor, Read, Seek, Write},
};

struct BatchData {
    x: Vec<u8>,
    y: Vec<u8>,
}

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

    /// List of .csv files to read samples from.
    /// CSVs must have the format: `fen,score,bestmove`
    #[arg(long, required = true)]
    input: String,

    /// Loop the input file indefinitely
    #[arg(long, default_value = "true")]
    input_loop: bool,

    // Offset to start reading from
    #[arg(long, default_value = "0")]
    input_offset: u64,

    // Length to read from the input file.
    // If 0, it will read until the end of the file
    #[arg(long, default_value = "0")]
    input_length: u64,

    /// The shared memory file to write the samples.
    /// Must have the correct size, otherwise it will panic
    #[arg(long)]
    shmem: String,

    /// The feature set to use
    #[arg(long)]
    feature_set: String,

    /// Number of samples in one batch
    #[arg(long, default_value = "8192")]
    batch_size: usize,

    /// Number of batch threads to use
    #[arg(long, default_value = "10")]
    threads: usize,
}

pub fn batch_loader(cmd: BatchLoaderCommand) -> Result<(), Box<dyn Error>> {
    let (batch_sender, batch_receiver) = bounded(8192); // keep up to X batches ready in memory

    // True length of the file
    let file_length = metadata(&cmd.input)
        .expect("Unable to query input size")
        .len();

    assert!(
        cmd.input_offset + cmd.input_length <= file_length,
        "Length is out of bounds"
    );

    // Length of the subfile to read from
    let readable_length = if cmd.input_length == 0 {
        file_length - cmd.input_offset
    } else {
        cmd.input_length
    };

    // Actual length to read from each thread
    let thread_length = ((readable_length as f64) / cmd.threads as f64).ceil() as u64;

    let feature_set = build_feature_set(&cmd.feature_set);
    let method = build_method(&cmd);

    // open shared memory file
    let mut shmem = ShmemConf::new().os_id(&cmd.shmem).open()?;
    let shmem_slice = unsafe { shmem.as_slice_mut() };

    let x_batch_size = cmd.batch_size * method.x_size(&feature_set);
    let y_batch_size = cmd.batch_size * method.y_size();
    assert!(
        shmem_slice.len() == x_batch_size + y_batch_size,
        "Unexpected shared memory size"
    );

    // batch_sender is moved into the scope, sent to the threads and later dropped
    // this means that when all threads are done, the channel will be disconnected
    // start batch threads
    let mut join_handles = vec![];

    for i in 0..cmd.threads {
        let batch_sender = batch_sender.clone();
        let cmd = cmd.clone();
        let offset = cmd.input_offset + i as u64 * thread_length;

        join_handles.push(std::thread::spawn(move || {
            build_samples_thread(
                cmd,
                offset,
                thread_length,
                x_batch_size,
                y_batch_size,
                batch_sender,
            )
        }));
    }

    // loop to write batches
    loop {
        // receive a batch from another thread
        if let Ok(batch) = batch_receiver.recv_timeout(Duration::from_millis(10)) {
            assert!(batch.x.len() == x_batch_size);
            assert!(batch.y.len() == y_batch_size);

            // wait for the reader to signal that it has finished copying the data of the last batch (1 byte)
            io::stdin().read_exact(&mut [0])?;

            // copy new batch to shared memory
            shmem_slice[..x_batch_size].copy_from_slice(&batch.x);
            shmem_slice[x_batch_size..].copy_from_slice(&batch.y);

            // send a signal to the consumer (1 byte)
            io::stdout().write_all(&[64])?;
            io::stdout().flush()?;
        } else {
            // After a timeout, check:
            // - There are no more batches to read
            // - All threads are done
            if batch_receiver.is_empty() && join_handles.iter().all(|h| h.is_finished()) {
                return Ok(());
            }
        }
    }
}

fn build_samples_thread(
    cmd: BatchLoaderCommand,
    offset: u64,
    length: u64,
    x_batch_size: usize,
    y_batch_size: usize,
    batch_sender: Sender<BatchData>,
) {
    let feature_set = build_feature_set(&cmd.feature_set);
    let mut method = build_method(&cmd);

    let mut x_cursor = Cursor::new(vec![0u8; x_batch_size]);
    let mut y_cursor = Cursor::new(vec![0u8; y_batch_size]);

    let file = File::open(&cmd.input).expect("Open input file");
    let mut reader = BufReader::with_capacity(256 * 1024 * 1024, file); // 256 MB

    eprintln!("Thread started offset={} length={}", offset, length);

    loop {
        while x_cursor.position() < x_batch_size as u64 {
            // read one more sample from the file
            let pos = reader.stream_position().unwrap();

            // break in EOF
            if !cmd.input_loop && pos >= offset + length {
                // no more samples to read
                // discard the batch
                // exiting the thread will disconnect the channel
                eprintln!("Thread finished");
                return;
            }

            if pos < offset || pos >= offset + length {
                // reset to offset if we are out of bounds
                reader.seek(SeekFrom::Start(offset)).unwrap();
                // make sure we skip to the first valid sample
                reader.read_until(b'\n', &mut vec![]).unwrap();
            }

            method.read_sample(&mut reader, &mut x_cursor, &mut y_cursor, &feature_set);
        }

        assert!(x_cursor.position() == x_batch_size as u64);
        assert!(y_cursor.position() == y_batch_size as u64);

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

fn build_method(cmd: &BatchLoaderCommand) -> Box<dyn ReadSample> {
    match cmd.method {
        Method::PQR => Box::new(PQR {}),
        Method::Eval => Box::new(EvalRead {}),
    }
}
