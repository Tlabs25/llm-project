import torch
import torch.nn as nn

from self_attention import SelfAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, context_length):
        super().__init__()

        # First normalization layer
        self.norm1 = nn.LayerNorm(embedding_size)

        # Self-attention layer
        self.attention = SelfAttention(
            embedding_size,
            context_length
        )

        # Second normalization layer
        self.norm2 = nn.LayerNorm(embedding_size)

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_size
        )

    def forward(self, x):
        # Normalize before attention
        attention_input = self.norm1(x)

        # Attention + residual connection
        x = x + self.attention(attention_input)

        # Normalize before feed-forward
        feed_forward_input = self.norm2(x)

        # Feed-forward + residual connection
        x = x + self.feed_forward(feed_forward_input)

        return x


# Test the TransformerBlock
if __name__ == "__main__":
    torch.manual_seed(42)

    embedding_size = 4
    context_length = 8

    # Create example input
    x = torch.randn(
        context_length,
        embedding_size
    )

    # Create the Transformer block
    transformer = TransformerBlock(
        embedding_size,
        context_length
    )

    # Pass input through the block
    output = transformer(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nTransformer output:")
    print(output)