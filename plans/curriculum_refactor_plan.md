# Curriculum Learning Refactoring Plan (Simplified)

## Objective
Change curriculum learning configuration from linear progression format to a flexible list-of-dicts format. No backward compatibility.

## Desired Format

```python
curriculum: Optional[list[dict[str, int]]] = [
    {"horizon": 5, "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 400},
]
```

- `horizon`: Maximum history length at this level
- `epochs`: Epoch at which to transition to this horizon

---

## Changes Required

### 1. config.py

**Replace lines 89-92:**
```python
# Curriculum learning settings
curriculum: Optional[list[dict[str, int]]] = [
    {"horizon": 5, "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 400},
]
```

**Remove these lines (deprecated):**
- `curriculum_start: Optional[int] = 5`
- `curriculum_step: int = 10`
- `curriculum_step_epochs: int = 80`

---

### 2. train_utils.py

#### 2.1 setup_training_environment() (~line 157)

**Replace:**
```python
curriculum_start = config.curriculum_start
curriculum_step = config.curriculum_step
curriculum_step_epochs = config.curriculum_step_epochs
```

**With:**
```python
curriculum = config.curriculum
```

**Update W&B config (~line 181):**
```python
"curriculum": curriculum,
```

#### 2.2 setup_curriculum_state() (~line 267)

**Replace entire function:**
```python
def setup_curriculum_state(
    curriculum: Optional[list[dict[str, int]]],
    checkpoint_state: Optional[dict] = None,
) -> dict:
    """
    Initialize curriculum learning state.
    """
    use_curriculum = curriculum is not None and len(curriculum) > 0
    
    if checkpoint_state and "current_horizon" in checkpoint_state:
        current_horizon = checkpoint_state["current_horizon"]
    elif use_curriculum:
        current_horizon = curriculum[0]["horizon"]
    else:
        current_horizon = curriculum[-1]["horizon"] if use_curriculum else 40
    
    return {
        "use_curriculum": use_curriculum,
        "current_horizon": current_horizon,
    }
```

#### 2.3 Training loop (~line 789)

**Replace:**
```python
curriculum_state = setup_curriculum_state(
    curriculum_start=config.curriculum_start,
    horizon=config.horizon,
    checkpoint_state=(
        training_state if "current_horizon" in training_state else None
    ),
)
```

**With:**
```python
curriculum_state = setup_curriculum_state(
    curriculum=config.curriculum,
    checkpoint_state=(
        training_state if "current_horizon" in training_state else None
    ),
)
```

#### 2.4 Horizon update logic (~line 816)

**Replace:**
```python
if (
    curriculum_state["use_curriculum"]
    and epoch > 0
    and epoch % curriculum_step_epochs == 0
):
    prev_horizon = curriculum_state["current_horizon"]
    curriculum_state["current_horizon"] = min(
        prev_horizon + curriculum_step, curriculum_state["max_horizon"]
    )
```

**With:**
```python
if curriculum_state["use_curriculum"]:
    for level in config.curriculum:
        if epoch >= level["epochs"]:
            curriculum_state["current_horizon"] = level["horizon"]
```

---

## Summary

| File           | Changes                                                                         |
| -------------- | ------------------------------------------------------------------------------- |
| config.py      | Replace 3 fields with `curriculum: list[dict]`                                  |
| train_utils.py | 4 locations: config extraction, curriculum setup, function call, horizon update |

---

## Usage Examples

```python
# Enable curriculum
config = Config(
    curriculum=[
        {"horizon": 5, "epochs": 100},
        {"horizon": 10, "epochs": 200},
        {"horizon": 20, "epochs": 300},
        {"horizon": 40, "epochs": 400},
    ]
)

# Disable curriculum (fixed horizon)
config = Config(
    curriculum=None
)
```
