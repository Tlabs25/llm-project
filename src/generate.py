import torch

from model import MiniGPT
from dataset import CharacterTokenizer


# -----------------------------
# SETTINGS
# -----------------------------

context_length = 128

embedding_size = 128
num_heads = 4
num_layers = 4

temperature = 0.8
top_k = 10

max_new_tokens = 500


# -----------------------------
# DEVICE
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:")
print(device)


# -----------------------------
# LOAD DATASET
# -----------------------------

with open(
    "data/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# -----------------------------
# CREATE TOKENIZER
# -----------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size

print("\nVocabulary size:")
print(vocab_size)


# -----------------------------
# CREATE MODEL
# -----------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    context_length=context_length,
    num_layers=num_layers,
    num_heads=num_heads
)

model = model.to(device)


# -----------------------------
# LOAD TRAINED WEIGHTS
# -----------------------------

model.load_state_dict(
    torch.load(
        "model_weights.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# GENERATE FUNCTION
# -----------------------------

def generate(
    prompt,
    max_new_tokens,
    temperature,
    top_k
):

    # -----------------------------
    # ENCODE PROMPT
    # -----------------------------

    token_ids = tokenizer.encode(
        prompt
    )


    # -----------------------------
    # GENERATION LOOP
    # -----------------------------

    for _ in range(max_new_tokens):

        # Only give the model the most
        # recent context window.
        context = token_ids[
            -context_length:
        ]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )


        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------

        with torch.no_grad():

            logits = model(x)


        # We only care about the prediction
        # at the final position.
        next_token_logits = logits[
            0,
            -1
        ]


        # -----------------------------
        # TEMPERATURE
        # -----------------------------

        next_token_logits = (
            next_token_logits
            / temperature
        )


        # -----------------------------
        # TOP-K
        # -----------------------------

        top_k_values, _ = torch.topk(
            next_token_logits,
            min(
                top_k,
                vocab_size
            )
        )

        minimum_top_k_value = (
            top_k_values[-1]
        )

        next_token_logits[
            next_token_logits
            < minimum_top_k_value
        ] = float("-inf")


        # -----------------------------
        # PROBABILITIES
        # -----------------------------

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )


        # -----------------------------
        # SAMPLE TOKEN
        # -----------------------------

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()


        # Add prediction to sequence
        token_ids.append(
            next_token_id
        )


    # -----------------------------
    # DECODE
    # -----------------------------

    return tokenizer.decode(
        token_ids
    )


# -----------------------------
# PROMPT
# -----------------------------

prompt = "Alice"


# -----------------------------
# GENERATE
# -----------------------------

generated_text = generate(
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k
)


# -----------------------------
# OUTPUT
# -----------------------------

print("\nPrompt:")
print(prompt)

print("\nTemperature:")
print(temperature)

print("\nTop-K:")
print(top_k)

print("\nGenerated text:\n")

print(generated_text)