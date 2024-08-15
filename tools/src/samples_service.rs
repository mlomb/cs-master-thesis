use crate::method::pqr::PQR;
use crate::method::{eval::EvalRead, ReadSample};
use clap::{Args, Subcommand};
use crossbeam::channel::{bounded, Receiver, Sender};
use nn::feature_set::build::build_feature_set;
use shared_memory::ShmemConf;
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, Cursor, Read, Seek, Write},
};

#[derive(Args, Clone)]
pub struct SamplesServiceCommand {
    /// List of .csv or .csv.zst files to read samples.
    /// CSVs must be generated using the `build-dataset` command
    #[arg(long, value_name = "input", required = true)]
    inputs: Vec<String>,

    /// The shared memory file to write the samples. Must have the correct size, otherwise it will panic
    #[arg(long, value_name = "shmem")]
    shmem: String,

    /// The feature set to use
    #[arg(long, value_name = "feature-set")]
    feature_set: String,

    /// Batch size
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Method to use
    #[command(subcommand)]
    subcommand: MethodSubcommand,

    /// Number of batch threads to use
    #[arg(long, default_value = "4")]
    batch_threads: usize,
}

#[derive(Subcommand, Clone)]
pub enum MethodSubcommand {
    /// Given a transition P → Q in a game, R is selected from a legal move from P while R != Q.
    PQR,
    ///
    Eval,
}

pub fn samples_service(cmd: SamplesServiceCommand) -> Result<(), Box<dyn Error>> {
    let (line_sender, line_receiver) = bounded(1_000_000); // keep up to X sample lines in memory
    let (batch_sender, batch_receiver) = bounded(128); // keep up to X batches ready in memory

    // start line thread
    let inputs = cmd.inputs.clone();
    std::thread::spawn(move || read_lines_thread(inputs, line_sender).unwrap());

    // start batch threads
    for _ in 0..cmd.batch_threads {
        let line_receiver = line_receiver.clone();
        let batch_sender = batch_sender.clone();
        let cmd = cmd.clone();
        std::thread::spawn(move || build_samples_thread(cmd, line_receiver, batch_sender));
    }

    let feature_set = build_feature_set(&cmd.feature_set);
    let method = build_method(&cmd);

    // open shared memory file
    let mut shmem = ShmemConf::new().os_id(cmd.shmem).open()?;
    let shmem_slice = unsafe { shmem.as_slice_mut() };

    let x_batch_size = cmd.batch_size * method.x_size(&feature_set);
    let y_batch_size = cmd.batch_size * method.y_size();
    assert!(
        shmem_slice.len() == x_batch_size + y_batch_size,
        "Unexpected shared memory size"
    );

    // loop to write batches
    loop {
        let batch: BatchReady = batch_receiver.recv().unwrap();
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
    }
}

struct BatchReady {
    x: Vec<u8>,
    y: Vec<u8>,
}

fn read_lines_thread(
    inputs: Vec<String>,
    line_sender: Sender<Vec<u8>>,
) -> Result<(), Box<dyn Error>> {
    // loop over the dataset indefinitely
    loop {
        // loop over every input file
        for filename in &inputs {
            let file = File::open(filename)?;

            // decompress if necessary
            let reader: Box<dyn io::Read> = if filename.ends_with(".zst") {
                Box::new(zstd::Decoder::new(file)?)
            } else {
                Box::new(file)
            };

            let reader = BufReader::with_capacity(50_000_000, reader);

            // send every line
            reader
                .split('\n' as u8)
                .for_each(|line| line_sender.send(line.unwrap()).unwrap());
        }
    }
}

fn build_samples_thread(
    cmd: SamplesServiceCommand,
    line_receiver: Receiver<Vec<u8>>,
    batch_sender: Sender<BatchReady>,
) {
    let feature_set = build_feature_set(&cmd.feature_set);
    let mut method = build_method(&cmd);

    let x_batch_size = cmd.batch_size * method.x_size(&feature_set);
    let y_batch_size = cmd.batch_size * method.y_size();

    let mut x_cursor = Cursor::new(vec![0u8; x_batch_size]);
    let mut y_cursor = Cursor::new(vec![0u8; y_batch_size]);

    loop {
        while x_cursor.position() < x_batch_size as u64 {
            let line = line_receiver.recv().unwrap();

            method.read_sample(
                &mut line.as_slice(),
                &mut x_cursor,
                &mut y_cursor,
                &feature_set,
            );
        }

        assert!(x_cursor.position() == x_batch_size as u64);
        assert!(y_cursor.position() == y_batch_size as u64);

        batch_sender
            .send(BatchReady {
                x: x_cursor.clone().into_inner(),
                y: y_cursor.clone().into_inner(),
            })
            .unwrap();

        // reset buffers
        x_cursor.rewind().unwrap();
        y_cursor.rewind().unwrap();
    }
}

fn build_method(cmd: &SamplesServiceCommand) -> Box<dyn ReadSample> {
    match cmd.subcommand {
        MethodSubcommand::PQR => Box::new(PQR {}),
        MethodSubcommand::Eval => Box::new(EvalRead {}),
    }
}
