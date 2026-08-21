import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, context_length):
        super().__init__()

        self.embedding_size = embedding_size

        # Create the Query, Key, and Value layers
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

        # Create and store the causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )

    def forward(self, x):
        # x shape:
        # [batch_size, sequence_length, embedding_size]

        batch_size, sequence_length, embedding_size = x.shape

        # Create Query, Key, and Value vectors
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Calculate attention scores
        #
        # queries:
        # [batch_size, sequence_length, embedding_size]
        #
        # keys.transpose(-2, -1):
        # [batch_size, embedding_size, sequence_length]
        #
        # scores:
        # [batch_size, sequence_length, sequence_length]
        scores = queries @ keys.transpose(-2, -1)

        # Scale the scores
        scale_factor = math.sqrt(embedding_size)
        scores = scores / scale_factor

        # Get the correct portion of the causal mask
        mask = self.mask[
            :sequence_length,
            :sequence_length
        ]

        # Apply causal mask
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Combine attention weights with Value vectors
        output = attention_weights @ values

        return output


# Test the SelfAttention module
if __name__ == "__main__":
    torch.manual_seed(42)

    embedding_size = 4
    context_length = 8
    batch_size = 3

    # Create example batched input
    x = torch.randn(
        batch_size,
        context_length,
        embedding_size
    )

    attention = SelfAttention(
        embedding_size,
        context_length
    )

    output = attention(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)