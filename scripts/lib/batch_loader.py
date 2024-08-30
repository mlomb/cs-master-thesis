import math
import torch
import numpy as np
import subprocess
from multiprocessing.shared_memory import SharedMemory
import os

TOOLS_BIN = os.path.join(os.path.dirname(__file__), "../../tools/target/release/tools")


def get_feature_set_size(name: str):
    """
    Get the number of features in a feature set by name
    """
    return int(
        subprocess.check_output(
            [
                TOOLS_BIN,
                "feature-set-size",
                "--feature-set=" + name,
            ]
        )
    )


class BatchLoader:
    """
    A class that loads batches of samples from the Rust tool binary
    """

    def __init__(
        self,
        batch_size: int,
        feature_set: str,
        method: str,
        input: str,
        input_offset: int = 0,
        input_length: int = 0,
        input_loop: bool = False,
        batch_threads: int = 1,
    ):
        num_features = get_feature_set_size(feature_set)

        if method == "pqr":
            x_shape = (batch_size, 3, 2, math.ceil(num_features / 64))
            y_shape = (batch_size, 0)
        elif method == "eval":
            x_shape = (batch_size, 2, math.ceil(num_features / 64))
            y_shape = (batch_size, 1)

        x_size = math.prod(x_shape) * 8  # 8 bytes per int64
        y_size = math.prod(y_shape) * 4  # 4 bytes per float32

        # Create the shared memory file.
        self.shmem = SharedMemory(create=True, size=x_size + y_size)

        # Start the program subprocess.
        args = [
            TOOLS_BIN,
            "batch-loader",
            "--method=" + method,
            "--input=" + input,
            "--input-offset=" + str(input_offset),
            "--input-length=" + str(input_length),
            "--shmem=" + self.shmem.name,
            "--batch-size=" + str(batch_size),
            "--feature-set=" + feature_set,
            "--threads=" + str(batch_threads),
        ]
        if input_loop:
            args.append("--input-loop")

        self.program = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        )

        # Initialize the numpy array using the shared memory as buffer.
        self.data = np.frombuffer(buffer=self.shmem.buf, dtype=np.int8)
        r = np.split(self.data, np.array([x_size, x_size + y_size]))

        self.x = r[0].view(dtype=np.int64).reshape(x_shape)
        self.y = r[1].view(dtype=np.float32).reshape(y_shape)

        # allow the generator to write the first batch
        self.notify_ready_for_next()

    def wait_until_ready(self):
        """
        Waits until the generator has written the next batch of samples into the shared memory.
        """
        read = self.program.stdout.read(1)
        if len(read) == 0:
            raise Exception("Generator process has terminated")

    def notify_ready_for_next(self):
        """
        Notifies the generator that the shared memory is no longer being used (data already copied).
        The generator can write a new batch into the shared memory.
        """
        self.program.stdin.write(b"\x00")
        self.program.stdin.flush()

    def next_batch(self):
        """
        Gets the next batch of samples.

        Returns:
            A Pytorch tensor containing the next batch of samples.
        """
        # Wait until batch is ready
        try:
            self.wait_until_ready()
        except Exception as e:
            print("process terminated1")
            return None

        # Create PyTorch tensors using the numpy arrays.
        # This will copy the data into the device, so after this line we don't care about self.data/x/y
        x_tensor = torch.tensor(self.x, dtype=torch.int64)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)

        # Release the shared memory for the generator to use.
        try:
            self.notify_ready_for_next()
        except:
            print("process terminated2")
            pass

        return x_tensor, y_tensor

    def cleanup(self):
        """
        Kill the subprocess and close the shared memory. Idempotent.
        """
        self.x = None
        self.y = None
        self.data = None

        if self.program is not None:
            self.program.kill()
            self.program.wait()
            self.program = None

        if self.shmem is not None:
            self.shmem.unlink()
            self.shmem.close()
            self.shmem = None

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.cleanup()

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    # Performance test

    import torch.utils
    from tqdm import tqdm

    off = 0

    for threads in [2, 4, 6, 8, 10, 12, 14, 16]:
        bl = BatchLoader(
            batch_size=16384,
            feature_set="hv",
            method="eval",
            input="/mnt/c/datasets/raw/all.plain",
            input_loop=True,
            input_offset=off,
            batch_threads=threads
        )
        off += 100 * 1024 * 1024 # 100 GB

        for i in tqdm(range(1_000), desc=f"batch_threads={threads}"):
            X, y = bl.next_batch()
            X, y = X.cuda(), y.cuda()
