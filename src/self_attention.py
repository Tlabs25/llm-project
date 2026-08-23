import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        context_length
    ):
        super().__init__()

        # Make sure the embedding can be divided
        # evenly among all attention heads.
        assert embedding_size % num_heads == 0

        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Example:
        #
        # embedding_size = 128
        # num_heads = 4
        #
        # head_size = 32
        self.head_size = embedding_size // num_heads

        # -----------------------------
        # QUERY, KEY, VALUE PROJECTIONS
        # -----------------------------

        self.query = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        # -----------------------------
        # OUTPUT PROJECTION
        # -----------------------------

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

        # -----------------------------
        # CAUSAL MASK
        # -----------------------------

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length
            )
        )

        self.register_buffer(
            "causal_mask",
            mask
        )


    def forward(self, x):

        # x shape:
        #
        # [batch_size, sequence_length, embedding_size]

        batch_size, sequence_length, embedding_size = x.shape

        # -----------------------------
        # CREATE Q, K, V
        # -----------------------------

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Current shape:
        #
        # [batch, sequence, embedding]

        # -----------------------------
        # SPLIT INTO HEADS
        # -----------------------------

        queries = queries.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )

        keys = keys.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )

        values = values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )

        # Current:
        #
        # [batch, sequence, heads, head_size]
        #
        # We want:
        #
        # [batch, heads, sequence, head_size]

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # -----------------------------
        # ATTENTION SCORES
        # -----------------------------

        attention_scores = (
            queries @ keys.transpose(-2, -1)
        )

        # Scale attention scores
        attention_scores = (
            attention_scores
            / (self.head_size ** 0.5)
        )

        # Shape:
        #
        # [batch, heads, sequence, sequence]

        # -----------------------------
        # CAUSAL MASK
        # -----------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # -----------------------------
        # SOFTMAX
        # -----------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        # -----------------------------
        # APPLY ATTENTION TO VALUES
        # -----------------------------

        output = attention_weights @ values

        # Shape:
        #
        # [batch, heads, sequence, head_size]

        # -----------------------------
        # COMBINE HEADS
        # -----------------------------

        output = output.transpose(1, 2)

        # Shape:
        #
        # [batch, sequence, heads, head_size]

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            embedding_size
        )

        # Back to:
        #
        # [batch, sequence, embedding]

        # -----------------------------
        # OUTPUT PROJECTION
        # -----------------------------

        output = self.output_projection(
            output
        )

        return output


# -----------------------------
# TEST
# -----------------------------

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 3
    sequence_length = 8

    embedding_size = 128
    num_heads = 4
    context_length = 128

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_size
    )

    attention = MultiHeadSelfAttention(
        embedding_size=embedding_size,
        num_heads=num_heads,
        context_length=context_length
    )

    output = attention(x)

    print("Input shape:")
    print(x.shape)

    print("\nNumber of heads:")
    print(num_heads)

    print("\nSize of each head:")
    print(attention.head_size)

    print("\nOutput shape:")
    print(output.shape)