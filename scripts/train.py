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

shm_status = open_shm("deepcmp-status")
shm_inputs = open_shm("deepcmp-inputs")
shm_outputs = open_shm("deepcmp-outputs")

import time
import numpy as np
import os
import onnx
import tf2onnx
import tensorflow.keras as keras
import math

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    write_graph=False,
)

model_path = "../models/best"
status = np.asarray(shm_status.buf).view(np.int32)
inputs = np.asarray(shm_inputs.buf).view(np.float32)
outputs = np.asarray(shm_outputs.buf).view(np.float32)

model = keras.models.load_model(model_path)

def train_iteration():

    batch_size = inputs.size // math.prod(model.inputs[0].shape.as_list()[1:])
    inputs_shape = (batch_size,) + model.inputs[0].shape[1:]
    outputs_shape = (batch_size,) + model.outputs[0].shape[1:]
    
    x = inputs.reshape(inputs_shape)
    y = outputs.reshape(outputs_shape)
    
    # fit model
    model.fit(
        x=x,
        y=y,
        shuffle=False, # already shuffled
        batch_size=batch_size,
        #initial_epoch=version,
        #epochs=version+1,
        callbacks=[tensorboard]
    )

while True:
    # wait until ready
    while status[0] == 0:
        time.sleep(0.1)

    train_iteration()

    # mark as done
    status[0] = 0
