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

        embeddings_list = []

        for caption in text:
            # Tokenize without padding/truncation to check length
            tokens = self.tokenizer(caption, return_tensors="pt", truncation=False)
            input_ids = tokens["input_ids"][0]
            seq_len = len(input_ids)

            if seq_len <= self.max_length:
                # Short text: encode directly
                inputs = self.tokenizer(
                    caption,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                output = self.model(**inputs)
                embedding = output.pooler_output  # (1, 512)
            else:
                # Long text: chunk and average
                chunk_embeddings = []

                # Split into overlapping chunks
                # Start from position 1 to skip BOS token for chunks
                stride = self.max_length - 2  # Leave room for BOS and EOS

                for start_idx in range(0, seq_len - 1, stride):
                    end_idx = min(start_idx + self.max_length - 1, seq_len - 1)

                    # Extract chunk tokens (keep BOS at start, EOS at end)
                    if start_idx == 0:
                        chunk_ids = input_ids[: end_idx + 1]
                    else:
                        # Add BOS token at the beginning
                        bos_token = torch.tensor([self.tokenizer.bos_token_id or 49406])
                        chunk_ids = torch.cat(
                            [bos_token, input_ids[start_idx : end_idx + 1]]
                        )

                    # Ensure EOS token at the end
                    if chunk_ids[-1] != self.tokenizer.eos_token_id:
                        eos_token = torch.tensor([self.tokenizer.eos_token_id or 49407])
                        chunk_ids = torch.cat([chunk_ids, eos_token])

                    # Truncate if still too long
                    if len(chunk_ids) > self.max_length:
                        chunk_ids = chunk_ids[: self.max_length - 1]
                        eos_token = torch.tensor([self.tokenizer.eos_token_id or 49407])
                        chunk_ids = torch.cat([chunk_ids, eos_token])

                    # Encode chunk
                    attention_mask = torch.ones_like(chunk_ids)
                    inputs = {
                        "input_ids": chunk_ids.unsqueeze(0).to(device),
                        "attention_mask": attention_mask.unsqueeze(0).to(device),
                    }
                    output = self.model(**inputs)
                    chunk_embeddings.append(output.pooler_output)

                # Average all chunk embeddings
                embedding = torch.stack(chunk_embeddings, dim=0).mean(dim=0)  # (1, 512)

            embeddings_list.append(embedding)

        # Stack all embeddings
        embeddings = torch.cat(embeddings_list, dim=0)  # (B, 512)

        return embeddings.unsqueeze(1)  # (B, 1, 512)

    @property
    def embedding_dim(self) -> int:
        """Output dimension of the CLIP text model."""
        return self.model.config.hidden_size
