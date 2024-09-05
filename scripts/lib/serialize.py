import torch


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
