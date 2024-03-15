import torch

class NNWriter:
    def __init__(self, model):
        self.buf = bytearray()

        self.write_linear(
            model.ft,
            weightType=torch.int16,
            weightScale=model.quantized_one,
            weightOrder='F', # column-major
            biasType=torch.int16,
            biasScale=model.quantized_one
        )

        for layer in [model.linear1, model.linear2]:
            self.write_linear(
                layer,
                weightType=torch.int8,
                weightScale=model.weight_scale,
                weightOrder='C', # row-major
                biasType=torch.int32,
                biasScale=model.quantized_one * model.weight_scale
            )

    def write_linear(self, layer, weightType, weightScale, weightOrder, biasType, biasScale):
        weight = layer.weight.data
        weight = weight.mul(weightScale).round().to(weightType)
        self.write_tensor(weight, weightOrder)

        bias = layer.bias.data
        bias = bias.mul(biasScale).round().to(biasType)
        self.write_tensor(bias)

    def write_tensor(self, tensor, order='C'):
        self.buf.extend(tensor.numpy().tobytes(order))

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

    # ---------------------------------------------------------
    # This part is to compare the quantized implementation side
    # ---------------------------------------------------------

    print("model length:", len(writer.buf))
    print("features:", features.tolist())

    ft = model.ft(sample_input)
    ft_crelu = torch.clamp(ft, 0, 1)
    linear1 = model.linear1(torch.concat([ft_crelu, ft_crelu], dim=0))

    print("ft+crelu:", torch.round(ft_crelu * model.quantized_one))
    print("linear1:", torch.round(linear1 * model.quantized_one))

