import torch
import torch.nn as nn

from model import MiniGPT


# -----------------------------
# SETTINGS
# -----------------------------

torch.manual_seed(42)

context_length = 8
batch_size = 4
embedding_size = 32
num_layers = 2
learning_rate = 0.001
training_steps = 2000


# -----------------------------
# LOAD TRAINING DATA
# -----------------------------

with open(
    "data/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Create the vocabulary
characters = sorted(list(set(text)))

vocab_size = len(characters)

character_to_id = {
    character: index
    for index, character in enumerate(characters)
}

id_to_character = {
    index: character
    for character, index in character_to_id.items()
}


# Convert the entire text into token IDs
data = torch.tensor(
    [character_to_id[character] for character in text],
    dtype=torch.long
)


# -----------------------------
# CREATE BATCHES
# -----------------------------

def get_batch():
    # Choose random starting positions
    starts = torch.randint(
        len(data) - context_length,
        (batch_size,)
    )

    # Create inputs and targets
    x = torch.stack([
        data[start:start + context_length]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + context_length + 1]
        for start in starts
    ])

    return x, y


# -----------------------------
# CREATE MODEL
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    context_length=context_length,
    num_layers=num_layers
)

model = model.to(device)


# -----------------------------
# LOSS AND OPTIMIZER
# -----------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -----------------------------
# TRAINING LOOP
# -----------------------------

for step in range(training_steps):

    # Get a random batch
    x, y = get_batch()

    # Move batch to GPU
    x = x.to(device)
    y = y.to(device)

    # Clear old gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(x)

    # Reshape for CrossEntropyLoss
    #
    # Current:
    # logits = [batch_size, context_length, vocab_size]
    # targets = [batch_size, context_length]
    #
    # CrossEntropyLoss expects:
    # logits = [examples, vocab_size]
    # targets = [examples]

    logits = logits.reshape(
        batch_size * context_length,
        vocab_size
    )

    y = y.reshape(
        batch_size * context_length
    )

    # Calculate loss
    loss = loss_function(
        logits,
        y
    )

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if step % 100 == 0:
        print(
            f"Step {step}: "
            f"Loss = {loss.item():.4f}"
        )


# -----------------------------
# SAVE TRAINED MODEL
# -----------------------------

torch.save(
    model.state_dict(),
    "model_weights.pth"
)

print("\nTraining complete!")
print("Model saved as model_weights.pth")