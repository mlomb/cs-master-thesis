import sys
from multiprocessing import shared_memory, resource_tracker

def open_shm(name):
    try:
        # map shared memory with the Rust side
        shm = shared_memory.SharedMemory(name)
        # avoid disposing the memory block on exit
        # see https://github.com/python/cpython/pull/15989
        if sys.platform != 'win32':
            # TODO: check why this is not working right :(
            resource_tracker.register(shm._name, 'shared_memory')

        return shm
    except FileNotFoundError as e:
        print(f"The shared memory block '{name}' does not exist. Please run the Rust side first.")
        exit(1)

shm_signal = open_shm("deep_cmp_shmem-signal")
shm_inputs = open_shm("deep_cmp_shmem-inputs")
shm_outputs = open_shm("deep_cmp_shmem-outputs")

import time
import os
import numpy as np
import onnx
import tf2onnx
import tensorflow.keras as keras

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    write_graph=False,
)

while True:
    time.sleep(0.1)

    print(shm_signal.buf[:10].tolist())
    