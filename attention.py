import torch

Q = torch.rand(3, 4)
K = torch.rand(3, 4)
V = torch.rand(3, 4)

scores = Q @ K.T/(4**0.5)
weights = torch.softmax(scores, dim=-1)
out = weights @ V

print(out)

'''
tensor([[0.5193, 0.4645, 0.5240, 0.3527],
        [0.5161, 0.4880, 0.5512, 0.3681],
        [0.5413, 0.4629, 0.5344, 0.3526]])
'''

# Masked Attention

mask = torch.tril(torch.ones(3, 3)) #keeps lower traingle partt so everything else 0
scores = scores.masked_fill(mask ==0, -1e9) #softmax makes -ve inf to 0

weights = torch.softmax(scores, dim=-1)
out = weights @ V

print(out)

'''
tensor([[0.8613, 0.9544, 0.4970, 0.0542],
        [0.5708, 0.8062, 0.3244, 0.2821],
        [0.3833, 0.5729, 0.4935, 0.2750]])
'''


''' 
Normal attention is used in encoder part to maybe tasks like reading some text whereas,
Masked attention is used in decoder part to generate text so it cant look into the future while generating
'''