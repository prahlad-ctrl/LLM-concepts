import torch
from torch import nn

vocab_size = 10
embed_dim = 5

embedding = nn.Embedding(vocab_size, embed_dim)

token = torch.tensor([1, 2, 3]) #imagine these are 3 words from prev like fast, fatser, fastest and getting converted to tokens
vectors = embedding(token)

print(vectors)

'''
tensor([[-0.2629, -0.2316, -0.4087,  1.3661, -0.6996],
        [ 0.3767,  0.0207,  0.1415, -0.4387,  1.3595],
        [ 2.0481,  1.1911, -1.7518,  0.2941, -0.5838]],
       grad_fn=<EmbeddingBackward0>)
'''

cos = torch.nn.functional.cosine_similarity #cosine similarity measures cosine of an angle between two vectors in a multidimensional space
print(cos(vectors[0], vectors[1], dim=0))

'''
tensor(-0.6575, grad_fn=<SumBackward1>)
'''

' right now its just some random garbage values but in actual architechture, the nn train them to be related to other words'