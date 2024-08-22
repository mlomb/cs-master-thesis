import torch
import numpy as np

class NnueWriter:
    def __init__(self, model, feature_set_name):
        self.buf = bytearray()

        self.buf.extend(bytes(feature_set_name, 'utf-8'))
        self.buf.append(0) # null-terminated string

        for k in [
            model.num_features,
            model.l1_size,
            model.l2_size,
        ]:
            # number of neurons
            self.buf.extend(k.to_bytes(4, byteorder='little', signed=False))

        self.write_linear(
            model.l1,
            weightType=torch.int16,
            weightScale=model.quantized_one,
            weightOrder='F', # column-major
            biasType=torch.int16,
            biasScale=model.quantized_one
        )

        self.write_linear(
            model.l2,
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
    from model import NnueModel, decode_int64_bitset

    model = NnueModel(768)
    model.load_state_dict(torch.load('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/notebooks/runs/20240316_151919_eval_basic_4096/models/350.pth'))
    model.clip_weights()

    writer = NnueWriter(model, "basic")
    with open("../../test_model.nn", "wb") as f:
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

    ft = model.l1(sample_input)
    ft_crelu = torch.clamp(ft, 0, 1)
    linear1 = model.l2(torch.concat([ft_crelu, ft_crelu], dim=0))
    linear1_crelu = torch.clamp(linear1, 0, 1)
    linear2 = model.l3(linear1_crelu)
    linear2_crelu = torch.clamp(linear2, 0, 1)
    linear_out = model.output(linear2_crelu)

    assert linear_out.item() - model(torch.concat([sample_input, sample_input], dim=0).reshape((2, 768))) < 1e-6

    print("ft+crelu:", torch.round(ft_crelu * model.quantized_one))
    print("linear1+crelu:", torch.round(linear1_crelu * model.quantized_one))
    print("linear2+crelu:", torch.round(linear2_crelu * model.quantized_one))
    print("output:", linear_out, torch.round(linear_out * model.nnue2score))

    # ----------------------------------------------------------------
    # Following code is to check that indices match with training data
    # ----------------------------------------------------------------

    # this buffer is the basic feature set for FEN "4nrk1/3q1pp1/2n1p1p1/8/1P2Q3/7P/PB1N1PP1/2R3K1 w - - 5 26"
    buffer = np.array([0, 97, 128, 2, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 96, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 80, 96, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 97, 128, 2, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    buffer = buffer.view(np.int64).reshape(-1, 2, 12)

    one_hot = decode_int64_bitset(torch.tensor(buffer)).reshape(-1, 2, 768)
    indices = torch.where(one_hot.flatten() > 0.0)

    print("===============")
    print("indices:", indices[0].tolist())
    print("expected model output:", model(one_hot))
