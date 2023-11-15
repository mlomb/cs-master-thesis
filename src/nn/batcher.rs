// https://github.com/epwalsh/batched-fn/blob/main/src/lib.rs

use flume::{unbounded, Sender};
use ort::Session;
use std::{thread::JoinHandle, time::Duration};

enum Message {
    /// Marks the end of the batcher thread
    Stop,

    /// A new batch has been requested
    Sample(usize, Sender<usize>),
}

pub struct Batcher {
    /// The channel to send the data to the batch thread
    tx: Sender<Message>,

    /// The batcher thread handle to be able to join it after stopping
    handle: JoinHandle<()>,
}

impl Batcher {
    pub fn spawn_with(session: Session, max_batch_size: usize, max_wait_time: Duration) -> Self {
        let (tx, rx) = unbounded();

        let handle = std::thread::spawn(move || {
            let mut batch_inputs = Vec::with_capacity(max_batch_size);
            let mut batch_txs = Vec::with_capacity(max_batch_size);

            // Wait for the first input in the batch
            while let Ok(Message::Sample(input, result_tx)) = rx.recv() {
                // Start building batch
                batch_inputs.push(input);
                batch_txs.push(result_tx);

                let deadline = std::time::Instant::now() + max_wait_time;

                // Wait for more inputs to come in
                while batch_inputs.len() < max_batch_size {
                    if let Ok(message) = rx.recv_deadline(deadline) {
                        match message {
                            Message::Stop => {
                                // Abort batch
                                return;
                            }
                            Message::Sample(input, result_tx) => {
                                // Add the input to the batch
                                batch_inputs.push(input);
                                batch_txs.push(result_tx);
                            }
                        }
                    } else {
                        // timed out, no new inputs
                        break;
                    }
                }

                // Run the batch
                batch_txs.iter().for_each(|tx| tx.send(0).unwrap());

                // Cleanup
                batch_inputs.clear();
                batch_txs.clear();
            }
        });

        Self { tx, handle }
    }

    pub fn stop(self) {
        // Notify the batch thread to stop
        self.tx.send(Message::Stop).unwrap();

        // Wait for it to stop
        self.handle.join().unwrap();
    }

    pub fn run(&self, input: usize) -> usize {
        // Create a channel to get the data back
        let (result_tx, result_rx) = unbounded();

        // Send the input to the batch thread
        self.tx.send(Message::Sample(input, result_tx)).unwrap();

        // Wait for the result
        result_rx.recv().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Batcher;
    use ort::{Environment, Session, SessionBuilder};
    use std::time::Duration;

    fn load_test_model() -> ort::Result<Session> {
        let environment = Environment::builder().build()?.into_arc();
        let session = SessionBuilder::new(&environment)?
            .with_intra_threads(1)?
            .with_model_from_memory(include_bytes!("../../tests/test_model.onnx"))?;

        Ok(session)
    }

    #[test]
    fn single_thread() {
        let batcher = Batcher::spawn_with(load_test_model().unwrap(), 7, Duration::from_millis(50));

        std::thread::scope(|s| {
            for _ in 0..20 {
                s.spawn({
                    let batcher = &batcher;
                    move || {
                        for key in [1, 2, 3, 4, 5, 6, 7, 8, 9] {
                            std::thread::yield_now();
                            let v = batcher.run(key);
                            println!("{}: {}", key, v);
                        }
                    }
                });
            }
            println!("all threads spawned");
        });

        println!("Stopping batcher");
        batcher.stop();
        println!("Batcher stopped");
    }
}
