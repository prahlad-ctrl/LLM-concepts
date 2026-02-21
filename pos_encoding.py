import math
import torch

def positional_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i] = math.sin(pos/ (10000**(i/dim)))
            pe[pos, i+1] = math.cos(pos/ (10000**(i/dim)))
    return pe

print(positional_encoding(6, 6))

'''
tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000],
        [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000],
        [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  1.0000],
        [-0.7568, -0.6536,  0.1846,  0.9828,  0.0086,  1.0000],
        [-0.9589,  0.2837,  0.2300,  0.9732,  0.0108,  0.9999]])
'''

'''
why sine and cosine? Using interlocking sine and cosine waves of different frequencies creates a unique, continuous pattern. 
The model can easily learn to attend to relative positions because for any fixed offset k, PE{pos+k} can be represented 
as a linear function of PE{pos}
'''