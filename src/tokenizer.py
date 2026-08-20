# Read our training text from the data folder
with open("data/training.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Find every unique character in the text and sort them
characters = sorted(list(set(text)))

# Create a vocabulary: each character gets an ID
char_to_id = {character: index for index, character in enumerate(characters)}

# Create the reverse vocabulary: each ID maps back to a character
id_to_char = {index: character for character, index in char_to_id.items()}

print("Training text:")
print(text)

print("\nVocabulary size:", len(characters))
print("\nCharacter to ID mapping:")
print(char_to_id)

# Convert the first 50 characters into token IDs
encoded = [char_to_id[character] for character in text[:50]]

print("\nFirst 50 characters:")
print(repr(text[:50]))

print("\nEncoded token IDs:")
print(encoded)

# Decode the token IDs back into text
decoded = "".join(id_to_char[token_id] for token_id in encoded)

print("\nDecoded text:")
print(decoded)