import subprocess
import numpy as np
# import tensorflow as tf
import torch
from multiprocessing.shared_memory import SharedMemory


class SamplesService:
    def __init__(self, batch_size):
        """
        Initializes the SamplesService class.

        Args:
            batch_size: The batch size to use.
        """

        
        size = batch_size * 3 * 12 * 8

        # Create the shared memory file.
        self.shmem = SharedMemory(create=True, size=size)

        # Start the program subprocess.
        args = [
            "../tools/target/release/tools",
            "samples-service",
            "--inputs=../tools/pqr.csv",
            "--shmem=" + self.shmem.name,
            "--batch-size=" + str(batch_size),
            "--feature-set=basic",
            "pqr"
        ]
        self.program = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        # Initialize the numpy array using the shared memory as buffer.
        self.data = np.frombuffer(
            buffer=self.shmem.buf,
            dtype=np.int64
        ).reshape((batch_size, 3 * 12))

    def next_batch(self):
        """
        Gets the next batch of samples.

        Returns:
            A TensorFlow tensor containing the next batch of samples.
        """

        # Wait until there is a byte in the stdout of the program (meaning data is ready).
        self.program.stdout.read(1)

        # Create a TensorFlow tensor using the numpy array.
        #with tf.device('/GPU:0'):
        #    tensor = tf.constant(self.data, dtype=tf.uint64)
        #    assert "GPU" in tensor.device
        
        # Create a PyTorch tensor using the numpy array.
        tensor = torch.tensor(self.data, dtype=torch.int64, device='cuda')
        assert tensor.is_cuda

        # Write a byte into the program's stdin, so it can start working on the next batch.
        self.program.stdin.write(b'\x00')
        self.program.stdin.flush()

        # Return the TensorFlow tensor.
        return tensor

    def __del__(self):
        """
        Destructor.
        """
        print("Samples service cleanup")

        # Kill the subprocess and close the shared memory.
        self.data = None
        self.program.kill()
        self.program.wait()
        self.shmem.close()
