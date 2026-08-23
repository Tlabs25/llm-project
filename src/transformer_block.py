import torch
import torch.nn as nn

from self_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        context_length
    ):
        super().__init__()

        # -----------------------------
        # LAYER NORMALIZATION
        # -----------------------------

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        # -----------------------------
        # MULTI-HEAD SELF-ATTENTION
        # -----------------------------

        self.attention = MultiHeadSelfAttention(
            embedding_size=embedding_size,
            num_heads=num_heads,
            context_length=context_length
        )

        # -----------------------------
        # FEED-FORWARD NETWORK
        # -----------------------------

        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            )
        )


    def forward(self, x):

        # -----------------------------
        # ATTENTION
        # -----------------------------

        normalized_x = self.layer_norm_1(x)

        attention_output = self.attention(
            normalized_x
        )

        # Residual connection
        x = x + attention_output

        # -----------------------------
        # FEED-FORWARD
        # -----------------------------

        normalized_x = self.layer_norm_2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual connection
        x = x + feed_forward_output

        return x


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

    transformer = TransformerBlock(
        embedding_size=embedding_size,
        num_heads=num_heads,
        context_length=context_length
    )

    output = transformer(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nNumber of attention heads:")
    print(num_heads)