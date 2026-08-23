import torch
import torch.nn as nn

# We currently have 26 unique tokens
vocab_size = 26

# Each token is represented by 4 numbers
embedding_size = 4

# Our current context length
context_length = 8

# Create the token embedding table
token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

# Create the position embedding table
position_embedding = nn.Embedding(
    context_length,
    embedding_size
)

# Our example: "The cat "
tokens = torch.tensor(
    [3, 11, 8, 1, 6, 4, 21, 1],
    dtype=torch.long
)

# Create position numbers for the 8 tokens
positions = torch.arange(context_length)

# Look up the vector for each token
token_vectors = token_embedding(tokens)

# Look up the vector for each position
position_vectors = position_embedding(positions)

# Combine token and position information
combined_vectors = token_vectors + position_vectors

print("Token IDs:")
print(tokens)

print("\nToken shape:")
print(tokens.shape)

print("\nPositions:")
print(positions)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)

print("\nCombined embedding shape:")
print(combined_vectors.shape)

print("\nCombined vectors:")
print(combined_vectors)