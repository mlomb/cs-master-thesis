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
        # to WDL [0, 0.5, 1]
        output = (output / out_scaling).sigmoid()

        # to [-1, 0, 1]
        output = 2 * output - 1

        output = output.reshape(-1, 3)

        #torch.set_printoptions(threshold=30, edgeitems=30, sci_mode=False)
        #print(output)

        kappa = 1

        p = output[:,0]
        q = output[:,1]
        r = output[:,2]

        a = -torch.log(torch.sigmoid(r - q)).mean()
        b = -kappa * torch.log(torch.sigmoid(( p + q))).mean()
        c = -kappa * torch.log(torch.sigmoid((-p - q))).mean()
        
        loss = a + b + c
        
        return loss
