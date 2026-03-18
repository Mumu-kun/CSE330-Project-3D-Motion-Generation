# Training Loop Refactoring Plan (Revised)

## Objective
Refactor the `train()` function in `src/utils/train_utils.py` to improve readability by extracting non-essential code into separate smaller functions, while **keeping the forward pass + loss computation inline** for understanding the core training mechanism.

## Refactoring Strategy

### EXTRACT to Helper Functions:
1. **Setup functions** - Before training loop
   - Curriculum state initialization
   - W&B logger initialization  
   - EMA models + optimizer setup
   - Mixed precision setup
   - Checkpoint resume logic

2. **Checkpoint saving** - Standalone function

3. **Batch preprocessing** - Before forward pass
   - Unpack batch (motion, text)
   - Sample window based on horizon

4. **Logging functions**
   - Batch-level W&B logging
   - Epoch-level W&B logging

5. **Checkpointing logic** - After each epoch

### KEEP INLINE (Core Training Mechanism):
1. **Forward pass** - Lines 394-455
   - Encoder call
   - Context expansion
   - Flow matching (t sampling, noise, x_t)
   - Predictor call
   
2. **Loss computation** - Lines 457-465
   - MSE loss for velocity field
   - Component losses (root_y, root_vel, root_rot, joints)

3. **Training step** - Lines 467-476
   - Backward pass
   - Gradient clipping
   - Optimizer step
   - EMA update

4. **Curriculum update** - Lines 324-331
   - Horizon increase at epoch boundaries

5. **Validation call** - Lines 536-571
   - Keep visible when validation runs

6. **W&B finish** - Lines 614-625

## Expected Result

The main `train()` function will show the **training flow clearly**:

```python
def train(...):
    # SETUP (~40 lines) - extracted to helpers
    device, save_dir = setup_directories(...)
    curriculum_state = setup_curriculum(...)
    wandb_logger = setup_wandb(...)
    encoder_ema, predictor_ema, optimizer, scaler = setup_training(...)
    start_epoch = load_checkpoint(...)
    
    # TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        # Update curriculum (INLINE - important for understanding)
        if use_curriculum and epoch % curriculum_step_epochs == 0:
            current_horizon = min(current_horizon + curriculum_step, max_horizon)
        
        for batch in dataloader:
            # PREPROCESS (~15 lines) - extracted
            motion, text = unpack_batch(batch, device, normalizer)
            hist, targets = sample_window(motion, lengths, current_horizon, device)
            
            # FORWARD PASS (INLINE - core mechanism!)
            text_input = apply_cfg_dropout(text, cfg_dropout)
            with autocast:
                contexts = encoder(hist, text_input)
                contexts = expand_contexts(contexts, num_frames)
                
                # Flow matching
                t = torch.rand(...)
                noise = torch.randn_like(targets)
                x_t = t * targets + (1-t) * noise
                
                pred = predictor(contexts, t, x_t, prev_features)
                
                # Loss computation (INLINE)
                loss = F.mse_loss(pred, targets - noise)
                loss_root_y = F.mse_loss(pred[:, 0], target_v[:, 0])
                ...
            
            # TRAINING STEP (INLINE)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            encoder_ema.update()
            predictor_ema.update()
            
            # LOGGING (extracted)
            log_batch_metrics(...)
        
        # VALIDATION (INLINE)
        if val_dataloader and epoch % val_interval == 0:
            val_metrics = validate(...)
        
        # CHECKPOINTING (extracted)
        handle_checkpointing(...)
    
    # FINISH (extracted)
    finish_training(wandb_logger, ...)
    
    return encoder_ema, predictor_ema
```

## Implementation Order

1. Create helper functions (setup, logging, checkpointing)
2. Refactor train() to use helpers
3. Keep forward pass + loss inline
4. Test that training still works correctly
