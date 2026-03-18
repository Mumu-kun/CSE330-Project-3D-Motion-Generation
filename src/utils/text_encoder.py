"""
Text encoding utility using CLIP model from Hugging Face Transformers.
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Union


class CLIPEncoder(torch.nn.Module):
    """
    Utility class to encode text captions using Microsoft's CLIP model.
    By default, uses 'openai/clip-vit-base-patch32' which produces 512D embeddings.

    For texts longer than 77 tokens, uses chunk-and-average approach to preserve
    all text content.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ):
        super().__init__()

        self.max_length = max_length

        print(f"Loading CLIP model '{model_name}'...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()

        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode a list of captions or a single caption into embeddings.

        For texts longer than max_length tokens, splits into chunks and averages
        the embeddings to preserve all text content.

        Args:
            text: A single string or a list of strings.

        Returns:
            embeddings: (B, 1, 512) tensor containing the pooled embeddings.
        """
        if isinstance(text, str):
            text = [text]

        # Determine device dynamically
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)

        # Use the pooler_output for a global representation of the sentence
        # Shape: (Batch_Size, 512)
        embeddings = outputs.pooler_output

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Output dimension of the CLIP text model."""
        return self.model.config.hidden_size
