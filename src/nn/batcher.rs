// https://github.com/epwalsh/batched-fn/blob/main/src/lib.rs

use flume::{unbounded, Sender};
use ndarray::{ArrayD, Axis, Slice};
use ort::{Session, Value};
use std::{thread::JoinHandle, time::Duration};

enum Message {
    /// Marks the end of the batcher thread
    Stop,

    /// A new batch has been requested
    Sample(Vec<ArrayD<f32>>, Sender<Vec<ArrayD<f32>>>),
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
            let mut batch_txs = Vec::with_capacity(max_batch_size);
            let mut batch_inputs: Vec<ArrayD<f32>> = session
                .inputs
                .iter()
                .map(|input| {
                    input
                        .input_type
                        .tensor_dimensions()
                        .expect("input tensor with dimensions")
                        .clone()
                })
                .map(|shape| {
                    assert!(shape[0] == -1, "model must support dynamic batch size");
                    assert!(shape[1..].iter().all(|&d| d != -1), "only one dynamic");

                    // i64 -> usize
                    let mut shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();

                    // reserve size for the biggest batch
                    shape[0] = max_batch_size;
                    shape
                })
                .map(|shape| ArrayD::zeros(shape))
                .collect();

            // Wait for the first input in the batch
            while let Ok(Message::Sample(inputs, result_tx)) = rx.recv() {
                // Start building batch
                batch_txs.push(result_tx);
                batch_inputs
                    .iter_mut()
                    .zip(inputs)
                    .for_each(|(batch_input, input)| {
                        // Note that this will panic if the input is not the right shape
                        batch_input.index_axis_mut(Axis(0), 0).assign(&input)
                    });

                let deadline = std::time::Instant::now() + max_wait_time;

                // Wait for more inputs to come in
                while batch_txs.len() < max_batch_size {
                    if let Ok(message) = rx.recv_deadline(deadline) {
                        match message {
                            Message::Stop => {
                                // Abort batch
                                return;
                            }
                            Message::Sample(inputs, result_tx) => {
                                let sample_index = batch_txs.len();

                                // Add the input to the batch
                                batch_txs.push(result_tx);
                                batch_inputs.iter_mut().zip(inputs).for_each(
                                    |(batch_input, input)| {
                                        // Note that this will panic if the input is not the right shape
                                        batch_input
                                            .index_axis_mut(Axis(0), sample_index)
                                            .assign(&input)
                                    },
                                );
                            }
                        }
                    } else {
                        // timed out, no new inputs
                        break;
                    }
                }

                // Convert the inputs into ort::Value-s
                let values: Vec<Value> = batch_inputs
                    .iter()
                    .map(|input| {
                        Value::from_array(
                            // Only keep the samples that were added to the batch
                            input.slice_axis(Axis(0), Slice::from(0..batch_txs.len())),
                        )
                        .unwrap()
                    })
                    .collect();

                // Run the batch
                println!("Running batch for {} samples", batch_txs.len());
                let result = session.run(values);

                match result {
                    Ok(outputs) => {
                        println!("Inference OK {:?}", outputs[0].extract_tensor::<f32>());

                        for (index, tx) in batch_txs.iter().enumerate() {
                            let arrays: Vec<ArrayD<f32>> = outputs
                                .iter()
                                .map(|(_, value)| {
                                    value
                                        .extract_tensor()
                                        .unwrap()
                                        .view()
                                        // extract the result from that sample
                                        .index_axis(Axis(0), index)
                                        .to_owned()
                                })
                                .collect();
                            tx.send(arrays).unwrap();
                        }
                    }
                    // TODO: handle properly
                    Err(err) => panic!("Error running inference"),
                }

                // Cleanup
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

    pub fn run(&self, inputs: Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        // Create a channel to get the data back
        let (result_tx, result_rx) = unbounded();

        // Send the input to the batch thread
        self.tx.send(Message::Sample(inputs, result_tx)).unwrap();

        // Wait for the result
        result_rx.recv().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Batcher;
    use ndarray::ArrayD;
    use ort::{Environment, Session, SessionBuilder};
    use std::time::Duration;

    fn load_test_model() -> ort::Result<Session> {
        let environment = Environment::builder().build()?.into_arc();
        let session = SessionBuilder::new(&environment)?
            .with_intra_threads(1)?
            .with_model_from_memory(include_bytes!("../../tests/test_model.onnx"))?;
        //.with_model_downloaded(ImageClassification::MobileNet)?;

        Ok(session)
    }

    #[test]
    fn single_thread() {
        let batcher =
            Batcher::spawn_with(load_test_model().unwrap(), 16, Duration::from_millis(50));

        std::thread::scope(|s| {
            for _ in 0..20 {
                s.spawn({
                    let batcher = &batcher;
                    move || {
                        let mut rng = rand::thread_rng();
                        for key in [1, 2, 3, 4, 5, 6, 7, 8, 9] {
                            std::thread::yield_now();
                            std::thread::sleep(Duration::from_millis(
                                // random
                                rand::Rng::gen_range(&mut rng, 0..200),
                            ));
                            let inputs = vec![ArrayD::zeros(vec![7, 6, 4])];
                            let out = batcher.run(inputs);
                            assert_eq!(out, vec![ArrayD::from_elem(vec![2], 0.5)]);
                            //println!("{}: {}", key, v);
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
