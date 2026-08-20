import torch
import torch.nn as nn

# Our vocabulary has 26 possible tokens
vocab_size = 26

# Each token will be represented by 4 learned numbers
embedding_size = 4

# Create the embedding table
embedding = nn.Embedding(vocab_size, embedding_size)

# Our first training example:
# "The cat "
tokens = torch.tensor([3, 11, 8, 1, 6, 4, 21, 1], dtype=torch.long)

# Look up the embedding vector for every token
embedded_tokens = embedding(tokens)

print("Token IDs:")
print(tokens)

print("\nShape before embedding:")
print(tokens.shape)

print("\nEmbedded tokens:")
print(embedded_tokens)

print("\nShape after embedding:")
print(embedded_tokens.shape)
