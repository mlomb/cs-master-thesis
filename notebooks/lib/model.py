import torch
from torch import nn

def decode_int64_bitset(x: torch.Tensor):
    """
    Convert a int64 into a 64-element float32 tensor
    """
    masks = 2 ** torch.arange(64, dtype=torch.int64, device=x.device)
    expanded = torch.bitwise_and(x.unsqueeze(-1), masks).ne(0).to(torch.float32)
    return expanded

class NnueModel(nn.Module):
    def __init__(self, num_features: int = 768, num_ft: int = 256, num_l1: int = 32, num_l2: int = 32):
        super(NnueModel, self).__init__()

        self.quantized_one = 127
        self.weight_scale_hidden = 64
        self.weight_scale_output = 16
        self.nnue2score = 600

        self.num_features = num_features
        self.num_ft = num_ft
        self.num_l1 = num_l1
        self.num_l2 = num_l2

        self.ft = nn.Linear(num_features, num_ft)
        self.linear1 = nn.Linear(num_ft * 2, num_l1)
        self.linear2 = nn.Linear(num_l1, num_l2)
        self.output = nn.Linear(num_l2, 1)

    def forward(self, x):
        x = self.ft(x)
        x = x.view(-1, self.num_ft * 2)
        x = torch.clamp(x, 0.0, 1.0) # Clipped ReLU

        x = self.linear1(x)
        x = torch.clamp(x, 0.0, 1.0) # Clipped ReLU
        x = self.linear2(x)
        x = torch.clamp(x, 0.0, 1.0) # Clipped ReLU

        return self.output(x) * self.nnue2score

    def clip_weights(self):
        # ft weights are NOT clamped, since they are stored with 16 bits
        # and we don't expect them to be very large

        hidden_clip = self.quantized_one / self.weight_scale_hidden
        output_clip = (self.quantized_one * self.quantized_one) * self.weight_scale_output

        self.linear1.weight.data.clamp_(-hidden_clip, hidden_clip)
        self.linear2.weight.data.clamp_(-hidden_clip, hidden_clip)
        self.output.weight.data.clamp_(-output_clip, output_clip)
