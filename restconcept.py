import torch
from torch import nn

# layer normalization

x = torch.randn(5, 10)
ln = nn.LayerNorm(10)

print(x.mean(dim=-1))
print(ln(x).mean(dim=-1))

'''
tensor([ 0.1923,  0.4134, -0.2226,  0.3897, -0.3004])
tensor([-1.1921e-08,  5.9605e-09, -2.5332e-08,  0.0000e+00,  3.5763e-08],
       grad_fn=<MeanBackward1>)
'''

# dropout

drop = nn.Dropout(p=0.5)
x = torch.ones(10)

print(drop(x))

'''
tensor([0., 2., 2., 0., 0., 2., 2., 0., 0., 0.])
'''

# temp scaling (new to me for now)

logits = torch.tensor([2.0, 1.0, 0.1])

def sample(logits, temp):
    probs = torch.softmax(logits/temp, dim=-1)
    return torch.multinomial(probs, 1)

print(sample(logits, 0.2)) #confident
print(sample(logits, 1.5)) #creative

'''
tensor([0])
tensor([0])
'''

# pretraining

inputs = torch.tensor([[1,2,3,4]])
targets = torch.tensor([[2,3,4,5]])

criterion = nn.CrossEntropyLoss()
logits = torch.randn(1, 4, 10)

loss = criterion(logits.view(-1, 10), targets.view(-1))
print(loss)

'''
tensor(3.2290)
'''