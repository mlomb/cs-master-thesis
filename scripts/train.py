import sys
import math
import time
import os
import numpy as np
from pathlib import Path
from multiprocessing import shared_memory, resource_tracker

def open_shm(name: str):
    try:
        # map shared memory with the Rust side
        shm = shared_memory.SharedMemory(name, create=False)
        # avoid disposing the memory block on exit
        # see https://github.com/python/cpython/pull/15989
        if sys.platform != 'win32':
            # TODO: check why this is not working right :(
            resource_tracker.unregister(shm._name, 'shared_memory')

        return shm
    except FileNotFoundError:
        print(f"The shared memory block '{name}' does not exist. Please run the Rust side first.")
        exit(1)

shm_status = open_shm("deepcmp-status")
shm_inputs = open_shm("deepcmp-inputs")
shm_outputs = open_shm("deepcmp-outputs")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence tensorflow

import onnx
import tf2onnx
import tensorflow.keras as keras

if len(sys.argv) < 2:
    print("Missing models folder path argument. Defaulting to '../models'")
    models_path = Path("../models")
else:
    models_path = Path(sys.argv[1])

status = np.asarray(shm_status.buf).view(np.int32)
inputs = np.asarray(shm_inputs.buf).view(np.float32)
outputs = np.asarray(shm_outputs.buf).view(np.float32)

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    write_graph=False,
)

def train_iteration(version: int):
    model = keras.models.load_model(models_path / "best")

    # infer shapes using the model and shmem size
    batch_size = inputs.size // math.prod(model.inputs[0].shape.as_list()[1:])
    inputs_shape = (batch_size,) + model.inputs[0].shape[1:]
    outputs_shape = (batch_size,) + model.outputs[0].shape[1:]
    
    x = inputs.reshape(inputs_shape)
    y = outputs.reshape(outputs_shape)
    
    model.fit(
        x=x,
        y=y,
        shuffle=True,
        batch_size=batch_size,
        initial_epoch=version,
        epochs=version+1,
        callbacks=[tensorboard]
    )

    # convert to onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # save the new candidate model
    candidate_path = models_path / "candidate"
    os.makedirs(candidate_path, exist_ok=True)
    model.save(candidate_path)
    onnx.save_model(onnx_model, candidate_path / "onnx_model.onnx")

while True:
    # wait until ready
    while status[0] == 0:
        time.sleep(0.1)

    train_iteration(
        version=status[1]
    )

    # mark as done
    status[0] = 0
