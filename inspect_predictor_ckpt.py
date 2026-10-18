import torch
from pathlib import Path

ckpt_path = Path("./checkpoints/phase3/predictor/phase3_predictor_best_val_20261016_050607.pt")
print("Exists:", ckpt_path.exists())
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Keys in checkpoint:", list(ckpt.keys()))
    if "metadata" in ckpt:
        print("Metadata:", ckpt["metadata"])
    if "metrics" in ckpt:
        print("Metrics:", ckpt["metrics"])
    if "config" in ckpt:
        print("Config num_epochs:", getattr(ckpt["config"], "_num_epochs", None))
        print("Config lr:", getattr(ckpt["config"], "learning_rate", None))
        print("Config batch_size:", getattr(ckpt["config"], "batch_size", None))
    if "latent_stats" in ckpt:
        ls = ckpt["latent_stats"]
        print("Latent stats keys:", ls.keys() if isinstance(ls, dict) else dir(ls))
        if isinstance(ls, dict):
            for k, v in ls.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape {v.shape}, min {v.min().item():.4f}, max {v.max().item():.4f}, mean {v.mean().item():.4f}")
