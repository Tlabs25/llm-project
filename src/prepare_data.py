import torch

# Read the training text
with open("data/training.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Find all unique characters
characters = sorted(list(set(text)))

# Create the mapping from each character to its token ID
char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

# Convert every character in the training text into its token ID
encoded = [
    char_to_id[character]
    for character in text
]

# Convert the Python list into a PyTorch tensor
data = torch.tensor(encoded, dtype=torch.long)

print("Number of characters:", len(text))
print("Vocabulary size:", len(characters))
print("Tensor shape:", data.shape)

print("\nFirst 20 token IDs:")
print(data[:20])

print("\nThose tokens decode to:")
print(repr(text[:20]))

# Number of previous tokens the model will see
context_length = 8

print("\n--- Training Examples ---")

# Show the first 5 input/target pairs
for i in range(5):
    # Get 8 tokens as input
    input_tokens = data[i:i + context_length]

    # The token immediately after those 8 tokens is the target
    target_token = data[i + context_length]

    # Convert them back to characters so we can understand them
    input_text = "".join(
        characters[token.item()]
        for token in input_tokens
    )

    target_text = characters[target_token.item()]

    print(f"\nExample {i + 1}")
    print("Input:", repr(input_text))
    print("Target:", repr(target_text))