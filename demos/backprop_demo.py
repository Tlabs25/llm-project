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


# Target: "he cat s"
target_tokens = torch.tensor(
    [11, 8, 1, 6, 4, 21, 1, 20],
    dtype=torch.long
)


# Loss function
loss_function = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)


# -----------------------------
# BEFORE TRAINING
# -----------------------------

logits = model(input_tokens)

loss = loss_function(
    logits,
    target_tokens
)

print("Loss before training:")
print(loss.item())


# -----------------------------
# BACKPROPAGATION
# -----------------------------

# Clear any old gradients
optimizer.zero_grad()

# Calculate gradients
loss.backward()


# Look at one gradient
print("\nExample gradient:")
print(model.output_layer.weight.grad[0])


# Update the model's parameters
optimizer.step()


# -----------------------------
# AFTER ONE UPDATE
# -----------------------------

logits = model(input_tokens)

new_loss = loss_function(
    logits,
    target_tokens
)

print("\nLoss after one training update:")
print(new_loss.item())


print("\nPredicted token IDs after training:")
print(torch.argmax(logits, dim=-1))