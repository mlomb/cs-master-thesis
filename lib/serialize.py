import torch

class NNWriter:
    def __init__(self, model):
        self.buf = bytearray()

        self.write_linear(
            model.ft,
            weightType=torch.int16,
            weightScale=model.quantized_one,
            biasType=torch.int16,
            biasScale=model.quantized_one
        )

        for layer in [model.linear1, model.linear2]:
            break
            self.write_linear(
                layer,
                weightType=torch.int8,
                weightScale=model.quantized_one,
                biasType=torch.int32,
                biasScale=model.quantized_one * model.weight_scale
            )

    def write_linear(self, layer, weightType, weightScale, biasType, biasScale):
        weight = layer.weight.data
        weight = weight.mul(weightScale).round().to(weightType)
        self.write_tensor(weight)

        bias = layer.bias.data
        bias = bias.mul(biasScale).round().to(biasType)
        self.write_tensor(bias)

    def write_tensor(self, tensor):
        print(tensor.shape)
        self.buf.extend(tensor.flatten().numpy().T.tobytes())

if __name__ == "__main__":
    import numpy as np
    from model import ChessModel

    model = ChessModel(768)
    model.clip_weights()

    writer = NNWriter(model)
    with open("../test_model.nn", "wb") as f:
        f.write(writer.buf)

    np.random.seed(42)
    np.set_printoptions(threshold=np.inf)

    features = np.random.choice(np.arange(0, 768), size=123, replace=False)
    sample_input = torch.zeros(768)
    sample_input[features] = 1

    print("model length:", len(writer.buf))
    print("features:", features.tolist())
    print("ft:", torch.round(model.ft(sample_input) * model.quantized_one))


