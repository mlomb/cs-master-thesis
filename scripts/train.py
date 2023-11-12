import sys
from multiprocessing import shared_memory, resource_tracker

def open_shm(name):
    try:
        # map shared memory with the Rust side
        shm = shared_memory.SharedMemory(name, create=False)
        # avoid disposing the memory block on exit
        # see https://github.com/python/cpython/pull/15989
        if sys.platform != 'win32':
            # TODO: check why this is not working right :(
            resource_tracker.unregister(shm._name, 'shared_memory')

        return shm
    except FileNotFoundError as e:
        print(f"The shared memory block '{name}' does not exist. Please run the Rust side first.")
        exit(1)

shm_signal = open_shm("deepcmp-signal")
shm_inputs = open_shm("deepcmp-inputs")
shm_outputs = open_shm("deepcmp-outputs")

import time
import numpy as np
import os
import onnx
import tf2onnx
import tensorflow.keras as keras

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    write_graph=False,
)

inputs = np.asarray(shm_inputs.buf).view(np.float32)
inputs = inputs.reshape((600, 7, 6, 4))

for i in range(0, 7):
    for j in range(0, 6):
        for k in range(0, 4):
            print("inputs[0][{}][{}][{}] = {}".format(i, j, k, inputs[0][i][j][k]))


while True:
    time.sleep(0.1)

    # print(shm_signal.buf[:10].tolist())
