from tensorflow.keras import losses
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

import onnx
import tf2onnx

input = Input(shape=(7,6,4), name='input')

x = Flatten()(input)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax', name='output')(x)

model = Model(inputs=[input], outputs=[x])
model.compile(loss=[losses.CategoricalCrossentropy()], optimizer='adam', metrics=['accuracy'])
model.summary()
model.save("../models/initial")

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, f"../models/initial/onnx_model.onnx")
