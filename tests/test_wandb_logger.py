import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import wandb_logger as wandb_logger_module
from utils.wandb_logger import WandbLogger


def test_wandb_logger_passes_name_and_resume_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeWandb:
        def login(self, key: str | None = None) -> None:
            captured["login_key"] = key

        def init(self, **kwargs: object) -> SimpleNamespace:
            captured["init_kwargs"] = kwargs
            return SimpleNamespace(
                id=kwargs.get("id"),
                name=kwargs.get("name"),
                url="https://wandb.test/runs/example",
            )

    monkeypatch.setattr(wandb_logger_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(wandb_logger_module, "wandb", FakeWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    logger = WandbLogger(
        project="motion-generation",
        name="resume-run",
        config={"epochs": 12},
        resume_id="wandb-run-123",
        resume="allow",
    )

    assert logger.run is not None
    assert captured["login_key"] == "test-key"
    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["project"] == "motion-generation"
    assert init_kwargs["name"] == "resume-run"
    assert init_kwargs["id"] == "wandb-run-123"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["config"] == {"epochs": 12}
