import torch

SCALING = 356.0

class PQRLoss(torch.nn.Module):
    def __init__(self):
        super(PQRLoss, self).__init__()

    def forward(self, output, _target):
        output = output.reshape(-1, 3)
        
        p = output[:,0] / SCALING
        q = output[:,1] / SCALING
        r = output[:,2] / SCALING
        
        a = -torch.mean(torch.log(torch.sigmoid(r - q)))
        b = torch.mean(torch.square(p + q))

        loss = a + b

        return loss

class EvalLoss(torch.nn.Module):
    def __init__(self):
        super(EvalLoss, self).__init__()

    def forward(self, output, target):

        # go from UCI cp to Stockfish's internal engine units
        # https://github.com/official-stockfish/Stockfish/blob/fb07281f5590bc216ecbacd468aa0d06fdead70c/src/uci.cpp#L341
        target = target * SCALING / 100.0

        # targets are in CP-space change it to WDL-space [0, 1]
        wdl_model = torch.sigmoid(output / SCALING)
        wdl_target = torch.sigmoid(target / SCALING)

        loss = torch.pow(torch.abs(wdl_model - wdl_target), 2.5)

        return loss.mean()
