# Multi-GPU Training Guide

## Overview
The training script now supports **Distributed Data Parallel (DDP)** training across multiple GPUs, which will significantly speed up training when multiple GPUs are available.

## How It Works

### Automatic Detection
The `train.lsf` script automatically detects the number of available GPUs and:
- If **1 GPU**: Runs normal single-GPU training
- If **2+ GPUs**: Launches distributed training using `torchrun`

### Key Changes Made

1. **Distributed Training Support**: The code now uses PyTorch's DistributedDataParallel (DDP) to split batches across multiple GPUs
2. **Proper Synchronization**: Each GPU processes different data batches, and gradients are synchronized across all GPUs
3. **Efficient Scaling**: The effective batch size scales with the number of GPUs (e.g., 2 GPUs = 2x effective batch size)

## Performance Benefits

With **n GPUs**, you can expect:
- **~n× speedup** in training time (e.g., 2 GPUs ≈ 2× faster)
- **n× larger effective batch size** (improves gradient estimates)
- More efficient GPU utilization

## Usage

### LSF Job Submission
When submitting your job, request multiple GPUs:

```bash
# Request 2 GPUs
bsub -n 2 -gpu "num=2" < train.lsf

# Request 4 GPUs
bsub -n 4 -gpu "num=4" < train.lsf
```

### Important Notes

1. **Gradient Accumulation**: If using `gradient_accumulation_steps`, it will be automatically divided by the number of GPUs to maintain the same effective batch size
   
2. **Checkpoints**: Only the master process (rank 0) saves checkpoints to avoid conflicts

3. **Logging**: Only the master process prints logs and saves to wandb (if enabled)

4. **Seeds**: Each GPU gets a slightly different random seed to ensure data diversity

## Configuration

No changes to your config files are needed! The DDP setup is automatic when multiple GPUs are detected.

### Adjusting for Multi-GPU
If you want to maintain the same per-GPU batch size when scaling to more GPUs:
- The `batch_size` parameter in config is the **per-GPU batch size**
- Total effective batch size = `batch_size × num_gpus × gradient_accumulation_steps`

## Troubleshooting

### NCCL Errors
If you see NCCL backend errors, make sure:
- All GPUs are on the same node
- NCCL library is installed in your environment

### Out of Memory
If you run out of GPU memory with multiple GPUs:
- Reduce `batch_size` in your config
- Reduce `block_size` or model size parameters

### Different Training Results
Due to the distributed nature and different random seeds per GPU, results may vary slightly from single-GPU runs. This is normal and expected.

## Testing

To test multi-GPU training locally:
```bash
# Test with 2 GPUs
torchrun --standalone --nproc_per_node=2 train.py config/sample_config.py

# Test with single GPU (backward compatibility)
python train.py config/sample_config.py
```

## Technical Details

The implementation follows the nanoGPT DDP approach:
- Uses PyTorch's `DistributedDataParallel` wrapper
- Employs `DistributedSampler` to partition data across GPUs
- Synchronizes gradients automatically after each backward pass
- Only rank 0 process handles I/O operations (checkpointing, logging)
