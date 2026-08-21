import torch

from model import MiniGPT


# -----------------------------
# SETTINGS
# -----------------------------

context_length = 8
embedding_size = 32
num_layers = 2
temperature = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD TRAINING DATA
# -----------------------------

with open(
    "data/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Recreate the same vocabulary used during training
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


# -----------------------------
# CREATE MODEL
# -----------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    context_length=context_length,
    num_layers=num_layers
)


# -----------------------------
# LOAD TRAINED WEIGHTS
# -----------------------------

model.load_state_dict(
    torch.load(
        "model_weights.pth",
        map_location=device
    )
)

model = model.to(device)


# Put the model into evaluation mode
model.eval()


# -----------------------------
# GENERATE TEXT
# -----------------------------

def generate(prompt, max_new_tokens, temperature):

    # Convert prompt characters into token IDs
    token_ids = [
        character_to_id[character]
        for character in prompt
    ]

    # Generate one character at a time
    for _ in range(max_new_tokens):

        # Only use the most recent context_length tokens
        context = token_ids[-context_length:]

        # Convert the context into a tensor
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        # Run the model without tracking gradients
        with torch.no_grad():
            logits = model(x)

        # Get predictions for the final position
        next_token_logits = logits[0, -1]

        # Adjust logits using temperature
        next_token_logits = next_token_logits / temperature

        # Convert logits into probabilities
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Randomly choose the next token based on probability
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        # Add the new token to the generated sequence
        token_ids.append(next_token_id)

    # Convert token IDs back into characters
    generated_text = "".join(
        id_to_character[token_id]
        for token_id in token_ids
    )

    return generated_text


# -----------------------------
# TEST GENERATION
# -----------------------------

prompt = "The cat "

generated_text = generate(
    prompt=prompt,
    max_new_tokens=100,
    temperature=temperature
)


print("Prompt:")
print(prompt)

print("\nTemperature:")
print(temperature)

print("\nGenerated text:")
print(generated_text)