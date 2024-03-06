import math
import torch
import numpy as np
import subprocess
from multiprocessing.shared_memory import SharedMemory

class SamplesService:
    def __init__(self, x_shape: tuple, y_shape: tuple, inputs: list[str], feature_set: str, method: str):
        batch_size = x_shape[0]
        x_size = math.prod(x_shape) * 8 # 8 bytes per int64
        y_size = math.prod(y_shape) * 4 # 4 bytes per float32

        # Create the shared memory file.
        self.shmem = SharedMemory(create=True, size=x_size + y_size)

        # Start the program subprocess.
        args = [
            "../tools/target/release/tools",
            "samples-service",
        ] + [f"--inputs={i}" for i in inputs] + [
            "--shmem=" + self.shmem.name,
            "--batch-size=" + str(batch_size),
            "--feature-set=" + feature_set,
            method
        ]
        self.program = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        # Initialize the numpy array using the shared memory as buffer.
        self.data = np.frombuffer(buffer=self.shmem.buf, dtype=np.int8)
        r = np.split(self.data, np.array([
            x_size,
            x_size + y_size
        ]))

        self.x = r[0].view(dtype=np.int64).reshape(x_shape)
        self.y = r[1].view(dtype=np.float32).reshape(y_shape)

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

        # Create PyTorch tensors using the numpy arrays.
        # This will copy the data into the device, so after this line we don't care about self.data/x/y
        x_tensor = torch.tensor(self.x, dtype=torch.int64, device='cuda')
        y_tensor = torch.tensor(self.y, dtype=torch.float32, device='cuda')

        # Liberate the shared memory for the generator to use.
        self.notify_ready_for_next()

        # Return the TensorFlow tensors.
        return x_tensor, y_tensor

    def __del__(self):
        """
        Destructor.
        """
        print("Samples service cleanup")

        # Kill the subprocess and close the shared memory.
        self.program.kill()
        self.program.wait()
        self.x = None
        self.y = None
        self.data = None
        self.shmem.close()
