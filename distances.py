import torch
def euclidean_distance(tensor1, tensor2):
    return torch.norm(torch.tensor(tensor1) - torch.tensor(tensor2)).item()

def cosine_distance(tensor1, tensor2):
    tensor1 = torch.tensor(tensor1)
    tensor2 = torch.tensor(tensor2)
    cos = torch.nn.CosineSimilarity(dim=0)
    return 1 - cos(tensor1, tensor2).item()