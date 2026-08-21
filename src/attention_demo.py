import torch
import torch.nn as nn

# Make our random results reproducible
torch.manual_seed(42)

# Our vocabulary and embedding settings
vocab_size = 26
embedding_size = 4
context_length = 8

# Token embedding table
token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

# Position embedding table
position_embedding = nn.Embedding(
    context_length,
    embedding_size
)

# Example input: "The cat "
tokens = torch.tensor(
    [3, 11, 8, 1, 6, 4, 21, 1],
    dtype=torch.long
)

# Position numbers: [0, 1, 2, 3, 4, 5, 6, 7]
positions = torch.arange(context_length)

# Create the combined token + position representations
x = token_embedding(tokens) + position_embedding(positions)

# Create three learned transformations
query_layer = nn.Linear(embedding_size, embedding_size, bias=False)
key_layer = nn.Linear(embedding_size, embedding_size, bias=False)
value_layer = nn.Linear(embedding_size, embedding_size, bias=False)

# Transform every input vector into a Query, Key, and Value
Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("Input shape:", x.shape)
print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)

print("\nFirst Query vector:")
print(Q[0])

print("\nFirst Key vector:")
print(K[0])

print("\nFirst Value vector:")
print(V[0])

# Calculate how strongly each Query matches each Key
attention_scores = Q @ K.transpose(-2, -1)

print("\nAttention score shape:")
print(attention_scores.shape)

print("\nAttention scores:")
print(attention_scores)

# Scale the attention scores
scale = embedding_size ** 0.5
scaled_scores = attention_scores / scale

print("\nScale factor:")
print(scale)

print("\nScaled attention scores:")
print(scaled_scores)

# Create a causal mask
mask = torch.tril(
    torch.ones(context_length, context_length)
)

print("\nCausal mask:")
print(mask)

# Prevent tokens from attending to future positions
masked_scores = scaled_scores.masked_fill(
    mask == 0,
    float("-inf")
)

print("\nMasked attention scores:")
print(masked_scores)

# Convert masked scores into attention weights
attention_weights = torch.softmax(masked_scores, dim=-1)

print("\nAttention weight shape:")
print(attention_weights.shape)

print("\nAttention weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))# Use the attention weights to combine the Value vectors
attention_output = attention_weights @ V

print("\nAttention output shape:")
print(attention_output.shape)

print("\nAttention output:")
print(attention_output)