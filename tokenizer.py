' Byte Pair Encoding '
' taking raw data breaking into subword units '

from collections import Counter

text = "slow slower slowest fast faster fastest"
words = text.split()

vocab = Counter()
for w in words:
    vocab[tuple(w)+ ("</w>",)]+= 1 # end of word token. this stores like 's','l','o','w','</w>'
    
def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bi_gram = pair
    for word in vocab:
        new_word = []
        i = 0
        while i< len(word):
            if i< len(word)-1 and (word[i], word[i+1])== bi_gram:
                new_word.append(word[i]+word[i+1])
                i+=2
            else:
                new_word.append(word[i])
                i+=1
        new_vocab[tuple(new_word)] = vocab[word]
    return new_vocab

for i in range(10):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    
print(vocab)

'''
After training 5:
{('slow', '</w>'): 1, ('slow', 'e', 'r', '</w>'): 1, ('slow', 'e', 'st</w>'): 1, ('f', 'a', 'st</w>'): 1, ('f', 'a', 'st', 'e', 'r', '</w>'): 1, ('f', 'a', 'st', 'e', 'st</w>'): 1}

After training 10:
{('slow', '</w>'): 1, ('slowe', 'r</w>'): 1, ('slowe', 'st</w>'): 1, ('fa', 'st</w>'): 1, ('faste', 'r</w>'): 1, ('faste', 'st</w>'): 1}
'''