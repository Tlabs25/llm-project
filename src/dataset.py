import torch


class CharacterTokenizer:
    """
    Simple character-level tokenizer.

    Each unique character receives an integer ID.
    """

    def __init__(self, text):
        # Find every unique character
        characters = sorted(list(set(text)))

        self.characters = characters

        # Character → ID
        self.character_to_id = {
            character: index
            for index, character in enumerate(characters)
        }

        # ID → Character
        self.id_to_character = {
            index: character
            for character, index in self.character_to_id.items()
        }

        self.vocab_size = len(characters)

    def encode(self, text):
        """
        Convert text into token IDs.
        """

        return [
            self.character_to_id[character]
            for character in text
        ]

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """

        return "".join(
            self.id_to_character[token_id]
            for token_id in token_ids
        )


class TextDataset:
    """
    Stores encoded text and creates training batches.
    """

    def __init__(
        self,
        text,
        context_length,
        train_split=0.90
    ):

        self.context_length = context_length

        # Create tokenizer
        self.tokenizer = CharacterTokenizer(text)

        # Encode entire dataset
        encoded_text = self.tokenizer.encode(text)

        self.data = torch.tensor(
            encoded_text,
            dtype=torch.long
        )

        # Determine train/validation split
        split_index = int(
            train_split * len(self.data)
        )

        self.train_data = self.data[:split_index]

        self.validation_data = self.data[split_index:]

    def get_batch(
        self,
        batch_size,
        split="train"
    ):
        """
        Return a random batch of input/target sequences.
        """

        if split == "train":
            data = self.train_data

        elif split == "validation":
            data = self.validation_data

        else:
            raise ValueError(
                "split must be 'train' or 'validation'"
            )

        # Random starting positions
        starts = torch.randint(
            len(data) - self.context_length,
            (batch_size,)
        )

        # Input sequences
        x = torch.stack([
            data[
                start:start + self.context_length
            ]
            for start in starts
        ])

        # Target sequences
        #
        # Shifted one token forward
        y = torch.stack([
            data[
                start + 1:
                start + self.context_length + 1
            ]
            for start in starts
        ])

        return x, y


# -----------------------------
# TEST DATASET
# -----------------------------

if __name__ == "__main__":

    # Load training text
    with open(
        "data/training.txt",
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    # Create dataset
    dataset = TextDataset(
        text=text,
        context_length=8
    )

    print("Total characters:")
    print(len(dataset.data))

    print("\nVocabulary size:")
    print(dataset.tokenizer.vocab_size)

    print("\nTraining characters:")
    print(len(dataset.train_data))

    print("\nValidation characters:")
    print(len(dataset.validation_data))

    # Get a sample batch
    x, y = dataset.get_batch(
        batch_size=4,
        split="train"
    )

    print("\nInput batch shape:")
    print(x.shape)

    print("\nTarget batch shape:")
    print(y.shape)

    print("\nInput batch:")
    print(x)

    print("\nTarget batch:")
    print(y)

    # Decode the first example
    print("\nFirst input decoded:")
    print(dataset.tokenizer.decode(x[0].tolist()))

    print("\nFirst target decoded:")
    print(dataset.tokenizer.decode(y[0].tolist()))