import torch
import torch.nn as nn

from model import MiniGPT


torch.manual_seed(42)

# Model settings
vocab_size = 26
embedding_size = 4
context_length = 8
num_layers = 2


# Create the model
model = MiniGPT(
    vocab_size,
    embedding_size,
    context_length,
    num_layers
)


# Input: "The cat "
input_tokens = torch.tensor(
    [3, 11, 8, 1, 6, 4, 21, 1],
    dtype=torch.long
)


# Correct next token for each position
#
# Input:  T h e _ c a t _
# Target: h e _ c a t _ s
#
target_tokens = torch.tensor(
    [11, 8, 1, 6, 4, 21, 1, 20],
    dtype=torch.long
)


# Run the input through the model
logits = model(input_tokens)


print("Input tokens:")
print(input_tokens)

print("\nTarget tokens:")
print(target_tokens)

print("\nLogits shape:")
print(logits.shape)


# Create the loss function
loss_function = nn.CrossEntropyLoss()


# Calculate loss
loss = loss_function(
    logits,
    target_tokens
)


print("\nLoss:")
print(loss.item())


# See what the model predicted
predictions = torch.argmax(
    logits,
    dim=-1
)

print("\nPredicted token IDs:")
print(predictions)