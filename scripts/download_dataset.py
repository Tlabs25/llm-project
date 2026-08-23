import urllib.request
from pathlib import Path


# -----------------------------
# SETTINGS
# -----------------------------

url = (
    "https://raw.githubusercontent.com/"
    "GITenberg/Alice-s-Adventures-in-Wonderland_11/"
    "master/11.txt"
)

output_path = Path("data/training.txt")


# -----------------------------
# DOWNLOAD DATASET
# -----------------------------

print("Downloading dataset...")

urllib.request.urlretrieve(
    url,
    output_path
)

print("Download complete!")


# -----------------------------
# CHECK DATASET
# -----------------------------

text = output_path.read_text(
    encoding="utf-8"
)

print("\nDataset saved to:")
print(output_path)

print("\nNumber of characters:")
print(len(text))

print("\nFirst 500 characters:")
print(text[:500])