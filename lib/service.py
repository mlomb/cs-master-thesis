import math
import torch
import numpy as np
import subprocess
from multiprocessing.shared_memory import SharedMemory

class SamplesService:
    def __init__(self, batch_shape: tuple, feature_set: str, inputs: list[str]):
        """
        Initializes the SamplesService class.

        Args:
            batch_size: The batch size to use.
        """
        batch_size = batch_shape[0]
        buffer_size = math.prod(batch_shape) * 8 # Method for now is PQR

        # Create the shared memory file.
        self.shmem = SharedMemory(create=True, size=buffer_size)

        # Start the program subprocess.
        args = [
            "../tools/target/release/tools",
            "samples-service",
        ] + [f"--inputs={i}" for i in inputs] + [
            "--shmem=" + self.shmem.name,
            "--batch-size=" + str(batch_size),
            "--feature-set=" + feature_set,
            "pqr"
        ]
        self.program = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        # Initialize the numpy array using the shared memory as buffer.
        self.data = np.frombuffer(buffer=self.shmem.buf, dtype=np.int64).reshape(batch_shape)

        # allow the generator to write the first batch
        self.notify_ready_for_next()

    def wait_until_ready(self):
        """
        Waits until the generator has written the next batch of samples into the shared memory.
        """
        self.program.stdout.read(1)

    def notify_ready_for_next(self):
        """
        Notifies the generator that the shared memory is no longer being used (data already copied).
        The generator can write a new batch into the shared memory.
        """
        self.program.stdin.write(b'\x00')
        self.program.stdin.flush()

    def next_batch(self):
        """
        Gets the next batch of samples.

        Returns:
            A TensorFlow tensor containing the next batch of samples.
        """
        # Wait until batch is ready
        self.wait_until_ready()

        # Create a PyTorch tensor using the numpy array.
        # This will copy the data into the device, so after this line we don't care about self.data
        tensor = torch.tensor(self.data, dtype=torch.int64, device='cuda')

        # Liberate the shared memory for the generator to use.
        self.notify_ready_for_next()

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
