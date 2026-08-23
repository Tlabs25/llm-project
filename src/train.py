import torch
import torch.nn as nn

from model import MiniGPT
from dataset import TextDataset


# -----------------------------
# SETTINGS
# -----------------------------

torch.manual_seed(42)

context_length = 128
batch_size = 32

embedding_size = 128
num_heads = 4
num_layers = 4

learning_rate = 0.001

training_steps = 5000
evaluation_interval = 100
evaluation_batches = 20

train_split = 0.90


# -----------------------------
# LOAD TEXT
# -----------------------------

with open(
    "data/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# -----------------------------
# CREATE DATASET
# -----------------------------

dataset = TextDataset(
    text=text,
    context_length=context_length,
    train_split=train_split
)


print("Total characters:")
print(len(dataset.data))

print("\nVocabulary size:")
print(dataset.tokenizer.vocab_size)

print("\nTraining characters:")
print(len(dataset.train_data))

print("\nValidation characters:")
print(len(dataset.validation_data))


# -----------------------------
# DEVICE
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nUsing device:")
print(device)


# -----------------------------
# CREATE MODEL
# -----------------------------

model = MiniGPT(
    vocab_size=dataset.tokenizer.vocab_size,
    embedding_size=embedding_size,
    context_length=context_length,
    num_layers=num_layers,
    num_heads=num_heads
)

model = model.to(device)


# -----------------------------
# PARAMETER COUNT
# -----------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:")
print(f"{total_parameters:,}")


# -----------------------------
# LOSS FUNCTION
# -----------------------------

loss_function = nn.CrossEntropyLoss()


# -----------------------------
# OPTIMIZER
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -----------------------------
# EVALUATION
# -----------------------------

def calculate_average_loss(split):

    # Put model into evaluation mode
    model.eval()

    losses = []

    with torch.no_grad():

        for _ in range(evaluation_batches):

            x, y = dataset.get_batch(
                batch_size=batch_size,
                split=split
            )

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Flatten predictions
            logits = logits.reshape(
                batch_size * context_length,
                dataset.tokenizer.vocab_size
            )

            # Flatten targets
            y = y.reshape(
                batch_size * context_length
            )

            loss = loss_function(
                logits,
                y
            )

            losses.append(
                loss.item()
            )

    # Return model to training mode
    model.train()

    return sum(losses) / len(losses)


# -----------------------------
# TRAINING LOOP
# -----------------------------

model.train()


for step in range(training_steps):

    # -----------------------------
    # GET TRAINING BATCH
    # -----------------------------

    x, y = dataset.get_batch(
        batch_size=batch_size,
        split="train"
    )

    x = x.to(device)
    y = y.to(device)


    # -----------------------------
    # CLEAR OLD GRADIENTS
    # -----------------------------

    optimizer.zero_grad()


    # -----------------------------
    # FORWARD PASS
    # -----------------------------

    logits = model(x)


    # -----------------------------
    # RESHAPE FOR LOSS
    # -----------------------------

    logits = logits.reshape(
        batch_size * context_length,
        dataset.tokenizer.vocab_size
    )

    y = y.reshape(
        batch_size * context_length
    )


    # -----------------------------
    # CALCULATE LOSS
    # -----------------------------

    loss = loss_function(
        logits,
        y
    )


    # -----------------------------
    # BACKPROPAGATION
    # -----------------------------

    loss.backward()


    # -----------------------------
    # UPDATE PARAMETERS
    # -----------------------------

    optimizer.step()


    # -----------------------------
    # EVALUATION
    # -----------------------------

    if step % evaluation_interval == 0:

        train_loss = calculate_average_loss(
            split="train"
        )

        validation_loss = calculate_average_loss(
            split="validation"
        )

        print(
            f"Step {step}: "
            f"Train Loss = {train_loss:.4f} | "
            f"Validation Loss = {validation_loss:.4f}"
        )


# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(
    model.state_dict(),
    "model_weights.pth"
)


print("\nTraining complete!")
print("Model saved as model_weights.pth")