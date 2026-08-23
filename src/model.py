import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        context_length,
        num_layers,
        num_heads
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_length = context_length

        # -----------------------------
        # TOKEN EMBEDDINGS
        # -----------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # -----------------------------
        # POSITION EMBEDDINGS
        # -----------------------------

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_size
        )

        # -----------------------------
        # TRANSFORMER BLOCKS
        # -----------------------------

        self.transformer_blocks = nn.ModuleList([

            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                context_length=context_length
            )

            for _ in range(num_layers)
        ])

        # -----------------------------
        # FINAL LAYER NORMALIZATION
        # -----------------------------

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # -----------------------------
        # OUTPUT LAYER
        # -----------------------------

        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )


    def forward(self, token_ids):

        # token_ids shape:
        #
        # [batch_size, sequence_length]

        batch_size, sequence_length = token_ids.shape

        # Prevent sequences longer than our
        # configured context window.
        if sequence_length > self.context_length:

            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds context length "
                f"{self.context_length}"
            )

        # -----------------------------
        # TOKEN EMBEDDINGS
        # -----------------------------

        token_embeddings = self.token_embedding(
            token_ids
        )

        # Shape:
        #
        # [batch, sequence, embedding]


        # -----------------------------
        # POSITION EMBEDDINGS
        # -----------------------------

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Shape:
        #
        # [sequence, embedding]


        # -----------------------------
        # COMBINE EMBEDDINGS
        # -----------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # -----------------------------
        # TRANSFORMER BLOCKS
        # -----------------------------

        for block in self.transformer_blocks:

            x = block(x)

        # -----------------------------
        # FINAL LAYER NORM
        # -----------------------------

        x = self.final_layer_norm(x)

        # -----------------------------
        # OUTPUT LOGITS
        # -----------------------------

        logits = self.output_layer(x)

        # Shape:
        #
        # [batch, sequence, vocabulary]

        return logits


# -----------------------------
# TEST MODEL
# -----------------------------

if __name__ == "__main__":

    torch.manual_seed(42)

    # New model configuration
    vocab_size = 71
    context_length = 128

    embedding_size = 128
    num_heads = 4
    num_layers = 4

    batch_size = 3

    # Create model
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # Create fake token data
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            context_length
        )
    )

    # Run model
    logits = model(token_ids)

    # -----------------------------
    # PARAMETER COUNT
    # -----------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------

    print("Input shape:")
    print(token_ids.shape)

    print("\nOutput logits shape:")
    print(logits.shape)

    print("\nModel configuration:")

    print("Vocabulary size:", vocab_size)
    print("Context length:", context_length)
    print("Embedding size:", embedding_size)
    print("Attention heads:", num_heads)
    print("Transformer layers:", num_layers)

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")

    print("\nTrainable parameters:")
    print(f"{trainable_parameters:,}")