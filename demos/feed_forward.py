import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        # Expand the embedding into a larger hidden representation
        self.linear1 = nn.Linear(
            embedding_size,
            embedding_size * 4
        )

        # Activation function
        self.relu = nn.ReLU()

        # Reduce it back to the original embedding size
        self.linear2 = nn.Linear(
            embedding_size * 4,
            embedding_size
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


# Test the FeedForward class
if __name__ == "__main__":
    torch.manual_seed(42)

    embedding_size = 4
    context_length = 8

    # Example input
    x = torch.randn(
        context_length,
        embedding_size
    )

    # Create the feed-forward network
    feed_forward = FeedForward(embedding_size)

    # Process the input
    output = feed_forward(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nFeed-forward output:")
    print(output)