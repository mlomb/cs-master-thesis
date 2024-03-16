import torch
import numpy as np

class NnueWriter:
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
                weightScale=model.weight_scale_hidden,
                weightOrder='C', # row-major
                biasType=torch.int32,
                biasScale=model.weight_scale_hidden * model.quantized_one
            )
        
        self.write_linear(
            model.output,
            weightType=torch.int8,
            weightScale=model.weight_scale_output * model.nnue2score / model.quantized_one,
            weightOrder='C', # (does not matter, a dimension is 1)
            biasType=torch.int32,
            biasScale=model.weight_scale_output * model.nnue2score
        )

    def write_linear(self, layer, weightType, weightScale, weightOrder, biasType, biasScale):
        weight = layer.weight.data
        weight = weight.mul(weightScale).round().to(weightType)
        self.write_tensor(weight, weightOrder)

        bias = layer.bias.data
        bias = bias.mul(biasScale).round().to(biasType)
        self.write_tensor(bias)

    def write_tensor(self, tensor, order='C'):
        self.buf.extend(tensor.cpu().numpy().tobytes(order))

if __name__ == "__main__":
    from model import NnueModel

    model = NnueModel(768)
    model.clip_weights()
    model.load_state_dict(torch.load('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/notebooks/runs/20240314_000247_eval_basic_4096/models/0.pth'))

    writer = NnueWriter(model)
    with open("../test_model.nn", "wb") as f:
        f.write(writer.buf)

    np.random.seed(42)
    np.set_printoptions(threshold=np.inf)

    features = np.random.choice(np.arange(0, 768), size=123, replace=False)
    sample_input = torch.zeros(768)
    sample_input[features] = 1

    # --------------------------------------------------------------
    # Following code is to compare with the quantized implementation
    # --------------------------------------------------------------

    print("model length:", len(writer.buf))
    print("features:", features.tolist())

    ft = model.ft(sample_input)
    ft_crelu = torch.clamp(ft, 0, 1)
    linear1 = model.linear1(torch.concat([ft_crelu, ft_crelu], dim=0))
    linear1_crelu = torch.clamp(linear1, 0, 1)
    linear2 = model.linear2(linear1_crelu)
    linear2_crelu = torch.clamp(linear2, 0, 1)
    linear_out = model.output(linear2_crelu)

    assert linear_out.item() - model(torch.concat([sample_input, sample_input], dim=0).reshape((2, 768))) < 1e-6

    print("ft+crelu:", torch.round(ft_crelu * model.quantized_one))
    print("linear1+crelu:", torch.round(linear1_crelu * model.quantized_one))
    print("linear2+crelu:", torch.round(linear2_crelu * model.quantized_one))
    print("output:", linear_out, torch.round(linear_out * model.nnue2score))
