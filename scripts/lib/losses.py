import torch

# https://discord.com/channels/435943710472011776/718853716266188890/821010651756363818 (Stockfish Discord)
in_scaling = 410
out_scaling = 361


class EvalLoss(torch.nn.Module):
    def __init__(self):
        super(EvalLoss, self).__init__()

    def forward(self, output, target):
        # Specific commit before doing a more complex refinement
        # https://github.com/official-stockfish/nnue-pytorch/blob/50eed1cea1f0c36dd595672005434a99116ff525/model.py#L295-L307

        # CP-space to WDL-space [0, 1]
        q = (output / out_scaling).sigmoid()
        p = (target / in_scaling).sigmoid()

        loss = torch.pow(torch.abs(p - q), 2.6).mean()

        return loss

class PQRLoss(torch.nn.Module):
    def __init__(self):
        super(PQRLoss, self).__init__()

    def forward(self, output, _target):
        output = output.reshape(-1, 3)

        kappa = 1

        p = output[:,0] / out_scaling
        q = output[:,1] / out_scaling
        r = output[:,2] / out_scaling

        a = -torch.log(torch.sigmoid(q - r)).mean()
        b = -torch.log(torch.sigmoid(kappa * (p + q))).mean()
        c = -torch.log(torch.sigmoid(kappa * (-q - p))).mean()

        loss = a + b + c
        
        return loss
