import torch
def rmsd_torch(X, Y):
    """ Assumes x,y are both (B * D * N). See below for wrapper. """
    X, Y = X.float(), Y.float()
    return torch.sqrt(torch.mean((X - Y) ** 2, axis=(-1, -2))).item()