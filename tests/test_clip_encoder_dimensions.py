"""
Test script to verify CLIPEncoder output dimensions consistency.

This test checks if the CLIPEncoder produces consistent (B, 1, 512) outputs
using pooled [CLS] token embeddings.
"""

import sys

sys.path.insert(0, ".")

import torch
from src.utils.text_encoder import CLIPEncoder


def test_clip_encoder_dimensions():
    """Test CLIPEncoder output dimensions with various inputs."""

    print("=" * 60)
    print("CLIPEncoder Output Dimensions Test")
    print("=" * 60)

    # Initialize encoder
    encoder = CLIPEncoder()
    print(f"\nEmbedding dimension: {encoder.embedding_dim}")

    # Test cases
    test_cases = [
        # (name, input)
        ("Short text", "walk"),
        ("Medium text", "a person walks forward slowly"),
        (
            "Long text",
            "a person walks forward slowly and then turns around and walks back to the starting position",
        ),
        (
            "Very long text",
            "a person walks forward slowly and then turns around and walks back to the starting position and then sits down on a chair and stands up again",
        ),
        (
            "Extremely long text",
            "a person walks forward slowly and then turns around and walks back to the starting position and then sits down on a chair and stands up again and then jumps up and down multiple times before finally lying down on the ground",
        ),
    ]

    print("\n" + "-" * 60)
    print("Single Input Tests")
    print("-" * 60)

    shapes = []
    for name, text in test_cases:
        output = encoder(text)
        shape = output.shape
        shapes.append(shape)
        print(f"\n{name}:")
        print(
            f"  Input: '{text[:50]}...' ({len(text)} chars)"
            if len(text) > 50
            else f"  Input: '{text}' ({len(text)} chars)"
        )
        print(f"  Output shape: {shape}")
        print(f"  Expected: (1, 1, 512)")
        print(f"  Match: {shape == torch.Size([1, 1, 512])}")

    # Test batch inputs
    print("\n" + "-" * 60)
    print("Batch Input Tests")
    print("-" * 60)

    batch_tests = [
        ("Batch of 2 short texts", ["walk", "run"]),
        (
            "Batch of 2 medium texts",
            ["a person walks forward", "a person runs backward"],
        ),
        (
            "Batch of mixed lengths",
            ["walk", "a person walks forward slowly and then turns around"],
        ),
        ("Batch of 5 texts", ["walk", "run", "jump", "sit", "dance"]),
    ]

    for name, texts in batch_tests:
        output = encoder(texts)
        shape = output.shape
        shapes.append(shape)
        print(f"\n{name}:")
        print(f"  Inputs: {texts}")
        print(f"  Output shape: {shape}")
        expected = torch.Size([len(texts), 1, 512])
        print(f"  Expected: {expected}")
        print(f"  Match: {shape == expected}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    unique_shapes = set(shapes)
    print(f"\nUnique output shapes found: {unique_shapes}")

    expected_shape = torch.Size([1, 1, 512])
    all_match = all(s[1:] == expected_shape[1:] for s in shapes)

    if all_match:
        print("\n[SUCCESS] All outputs have consistent shape (B, 1, 512)")
        print("  Using pooled [CLS] token embeddings - no redundant padding!")
    else:
        print("\n[FAILURE] Output dimensions are inconsistent!")

    return all_match


if __name__ == "__main__":
    success = test_clip_encoder_dimensions()
    sys.exit(0 if success else 1)
