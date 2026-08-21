import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        context_length,
        num_layers
    ):
        super().__init__()

        # Save model settings
        self.context_length = context_length

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_size
        )

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_size,
                    context_length
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            embedding_size
        )

        # Convert embeddings into vocabulary scores
        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):
        # token_ids shape:
        # [batch_size, sequence_length]

        batch_size, sequence_length = token_ids.shape

        # Create position IDs
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Convert token IDs into vectors
        #
        # [batch_size, sequence_length]
        #             ↓
        # [batch_size, sequence_length, embedding_size]
        token_vectors = self.token_embedding(token_ids)

        # Convert positions into vectors
        #
        # [sequence_length]
        #         ↓
        # [sequence_length, embedding_size]
        position_vectors = self.position_embedding(positions)

        # Add position information to every sequence
        # PyTorch broadcasts position_vectors across the batch
        x = token_vectors + position_vectors

        # Pass through Transformer blocks
        x = self.blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Convert into vocabulary scores
        #
        # [batch_size, sequence_length, embedding_size]
        #                    ↓
        # [batch_size, sequence_length, vocab_size]
        logits = self.output_layer(x)

        return logits


# Test the complete model
if __name__ == "__main__":
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

    # Create a batch of 3 sequences
    token_ids = torch.tensor(
        [
            [3, 11, 8, 1, 6, 4, 21, 1],
            [11, 8, 1, 6, 4, 21, 1, 20],
            [8, 1, 6, 4, 21, 1, 20, 4]
        ],
        dtype=torch.long
    )

    # Run the model
    logits = model(token_ids)

    print("Input token IDs:")
    print(token_ids)

    print("\nInput shape:")
    print(token_ids.shape)

    print("\nOutput logits shape:")
    print(logits.shape)

    print("\nVocabulary scores for the first token")
    print(logits[0, 0])

    print("\nPredicted token IDs at each position:")
    print(torch.argmax(logits, dim=-1))