# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository trains transformer models on arithmetic tasks (addition, subtraction, multiplication, sorting). The codebase supports different data formats (plain, reversed, scratchpad) and analyzes how transformers learn algorithmic reasoning.

## Common Commands

### Generate Data
```bash
python data_generate.py --task <task> --num_operands <n> --experiment_name <name> \
  --train_size <N> --test_size <N> --val_size <N> \
  --train_eval True --sample-size <N> --generate_reverse True
```

**Examples:**
```bash
# 4-operand addition
python data_generate.py --task addition --num_operands 4 --experiment_name 4_operands_0_to_999_uniform \
  --train_size 1000000 --test_size 10000 --val_size 10000 --train_eval True --sample-size 10000 --generate_reverse True

# 2-operand multiplication
python data_generate.py --task multiplication --experiment_name 2_operands_mul \
  --train_size 100000 --test_size 5000 --val_size 5000
```

### Train Model
```bash
python train.py <config_file>
```

**With batch mode:**
```bash
python train.py <config_file> --batch [per_example|slicing]
```

**Examples:**
```bash
python train.py 4_operands_addition_plain.txt
python train.py 2_operands_addition_reversed.txt --batch slicing
```

**Using --task and --experiment_name flags (loads default config for task):**
```bash
python train.py --task addition --experiment_name my_experiment
python train.py --task multiplication --experiment_name mul_experiment
```

## Architecture Overview

### Data Flow
1. **Data Generation** (`data_generate.py`): Dispatches to task-specific generators in `data_generation_script/individual_task_scripts/`
   - Creates `train.txt`, `test.txt`, `val.txt` in `data/<experiment_name>/`
   - Optionally creates `train_eval.txt` (sampled from training data)
   - Optionally creates reversed format files

2. **Configuration** (`configuration_files/*.txt`): Plain-text config files that override default training parameters
   - Loaded via `exec()` in `train.py`
   - Can specify data paths, model architecture, training hyperparameters

3. **Training** (`train.py`): Main training loop
   - Loads config file and overrides defaults
   - Two batch preparation modes:
     - `per_example`: Each example padded to block_size (uses Dataset/DataLoader)
     - `slicing`: Random token slices from concatenated data
   - Evaluates on test/val/train_eval at intervals
   - Saves checkpoints to `results/<experiment_name>/`

4. **Evaluation** (`evaluation.py`): Evaluates model accuracy
   - Can compute groundtruth (`mode="compute_gold"`) or read from files (`mode="read_gold_as_str"`)
   - Supports multiple test files
   - Produces CSV files with correct/incorrect examples

### Model Architecture (`model.py`)
- GPT-style decoder-only transformer
- Key components:
  - `GPTConfig`: Hyperparameter dataclass
  - `GPT`: Main model class with tied embeddings (`wte.weight` = `lm_head.weight`)
  - `CasualSelfAttention`: Supports Flash Attention (PyTorch 2.0+)
  - Padding-aware loss: Uses `ignore_index=pad_id` in cross-entropy

### Key Utilities
- `main_utilities.py`: Helper functions for data loading, metadata creation
- `statistical_measurements.py`: Mutual information calculations for analyzing model internals
- `result_analysis.py`: Post-training analysis of digitwise errors
- `configurator.py`: Legacy config parsing (train.py now handles this directly)

### Configuration System
Config files in `configuration_files/` override defaults in `train.py`. Critical parameters:

**Data parameters:**
- `data_dir`: Path to data directory (e.g., `'data/4_operands_0_to_999_uniform/'`)
- `train_data_name`, `val_data_name`, `test_file_name`: Data file names
- `train_data_test_name`: Sampled training data for eval (if `eval_addition_train=True`)
- `data_format`: `'plain'`, `'reverse'`, `'scratchpad'`, `'max'`, or `'sorting'`
- `operator`: `'+'`, `'-'`, or `'*'`
- `mode`: `"compute_gold"` (compute groundtruth) or `"read_gold_as_str"` (read from file)

**Model parameters:**
- `n_layer`, `n_head`, `n_embd`: Transformer architecture
- `block_size`: Context length
- `dropout`: Dropout rate
- `use_flash`: Enable Flash Attention

**Training parameters:**
- `batch_size`: Batch size
- `learning_rate`, `max_iters`, `warmup_iters`, `lr_decay_iters`
- `eval_interval`: Evaluate every N iterations
- `max_new_tokens`: Maximum tokens to generate during evaluation

**Output parameters:**
- `out_dir`: Checkpoint and results directory
- `wandb_log`, `wandb_project`, `wandb_run_name`: Weights & Biases logging

## Data Formats

The codebase supports multiple data representations:

1. **Plain format**: `123+456=579$`
2. **Reversed format**: `321+654=975$` (operands and result reversed)
3. **Scratchpad format**: Includes intermediate computation steps

Each line ends with `$` (EOS token). The tokenizer uses character-level encoding with a special `<pad>` token.

## Batch Preparation Modes

Controlled by `--batch` flag:

1. **per_example** (default):
   - Each training example is padded to `block_size`
   - Uses PyTorch Dataset/DataLoader
   - Better for tasks where examples shouldn't be concatenated

2. **slicing**:
   - Concatenates all training data into one sequence
   - Samples random windows of length `block_size`
   - More efficient for language modeling-style training

## Evaluation Modes

Controlled by `mode` config parameter:

1. **compute_gold**: Parses operands from test file and computes expected result
   - Used when test files contain only operands (e.g., `123+456=`)

2. **read_gold_as_str**: Reads expected result from test file
   - Used when test files contain groundtruth (e.g., `123+456=579$`)
   - Performs exact string matching

## Digitwise Error Analysis

When `operator` is `'+'`, `'-'`, or `'*'`, the training script automatically runs `result_analysis.analyze_csv()` after training to generate digit-by-digit error plots. These visualizations show which digit positions the model struggles with most.

## Statistical Measurements

The codebase includes mutual information (MI) measurement capabilities (`mi_measurement=True` in config):
- Measures input-output and output-output MI during training
- Only active for addition tasks (`operator='+'`)
- Requires specific dataset structure (4-operand addition with digit place tracking)
- Results saved to CSV files in results directory

## Directory Structure

```
.
├── data/                          # Generated datasets
│   └── <experiment_name>/         # Each experiment gets its own folder
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── train_eval.txt         # Optional
├── configuration_files/           # Config files for different tasks
├── data_generation_script/        # Data generation utilities
│   └── individual_task_scripts/   # Task-specific generators
│       ├── addition/
│       ├── multiplication/
│       └── sorting/
├── results/                       # Training outputs
│   └── <experiment_name>/
│       ├── ckpt_*.pt              # Checkpoints
│       ├── training_metrics.csv   # Loss/accuracy over time
│       ├── correct_examples.csv
│       ├── incorrect_examples.csv
│       └── digitwise_errors.png   # Error analysis plot
├── train.py                       # Main training script
├── model.py                       # GPT model implementation
├── evaluation.py                  # Evaluation utilities
├── data_generate.py               # Data generation dispatcher
└── main_utilities.py              # Helper functions
```

## Important Implementation Details

1. **Metadata and Tokenization**:
   - Vocabulary created from training data in `create_meta_for_addition()`
   - Always includes `<pad>` and `$` (EOS) tokens
   - Character-level tokenization by default

2. **Padding Handling**:
   - Model ignores padding tokens in loss via `ignore_index=pad_id`
   - Padding token is `<pad>`, EOS token is `$`

3. **Checkpoint Format**:
   - Checkpoints save: model state, optimizer state, config, metadata, iteration number, best metrics
   - Resume training with `init_from='resume'` in config

4. **Greedy Decoding**:
   - Recent commit switched to greedy decoding (`top_k=1`)
   - See `model.generate()` method

5. **Config Loading Priority**:
   - Defaults in `train.py` → Config file → Command-line args
   - Config file loaded via `exec()` which directly modifies globals
