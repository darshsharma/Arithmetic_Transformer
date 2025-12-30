#!/usr/bin/env python3
"""
Test pre-trained model (e.g., Pythia-1B) on addition task WITHOUT fine-tuning.

This script loads a pre-trained model and generates outputs for addition problems
to see what the model produces before any fine-tuning.

Usage:
    python test_pretrained_model.py --model EleutherAI/pythia-1b \
                                     --test_file data/4_operands_0_to_999_uniform/test.txt \
                                     --output pretrained_outputs.csv
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_test_line(line, mode="read_gold_as_str"):
    """
    Parse a test line to extract input (operands) and actual answer.

    Args:
        line: Test line like "123+456=579$" or "123+456="
        mode: "read_gold_as_str" if answer is in file, "compute_gold" if not

    Returns:
        tuple: (input_text, actual_answer)
               - input_text: "123+456=" (without answer or EOS)
               - actual_answer: "579" (without EOS) or None if mode="compute_gold"
    """
    line = line.strip()

    # Remove EOS token if present
    if line.endswith('$'):
        line = line[:-1]

    # Split by '='
    if '=' in line:
        parts = line.split('=')
        operands = parts[0] + '='

        if mode == "read_gold_as_str" and len(parts) > 1:
            actual = parts[1].strip()
        else:
            actual = None

        return operands, actual
    else:
        return line, None


def main():
    parser = argparse.ArgumentParser(
        description="Test pre-trained model on addition task"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-1b",
        help="HuggingFace model name (default: EleutherAI/pythia-1b)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test file (e.g., data/4_operands_0_to_999_uniform/test.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained_outputs.csv",
        help="Output CSV file (default: pretrained_outputs.csv)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to test (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="read_gold_as_str",
        choices=["read_gold_as_str", "compute_gold"],
        help="How to read ground truth (default: read_gold_as_str)"
    )

    args = parser.parse_args()

    # Validate test file
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        return 1

    print("=" * 70)
    print("Testing Pre-trained Model on Addition Task (NO Fine-tuning)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Test file: {args.test_file}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 70)

    # Load tokenizer (use original tokenizer, no modifications)
    print("\n[1/4] Loading original tokenizer (no modifications)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set pad token if not set (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Load model (use original pre-trained model, no adapters)
    print("\n[2/4] Loading pre-trained model (no fine-tuning, no modifications)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device
    )
    model.eval()  # Set to evaluation mode
    print(f"✓ Model loaded and moved to {args.device}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e9:.2f}B")

    # Read test file
    print("\n[3/4] Reading test file...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if args.max_examples:
        lines = lines[:args.max_examples]

    print(f"✓ Loaded {len(lines)} test examples")

    # Generate predictions
    print("\n[4/4] Generating predictions...")
    results = []

    for line in tqdm(lines, desc="Processing"):
        # Parse input and actual
        input_text, actual_answer = parse_test_line(line, mode=args.mode)

        # Tokenize input using original tokenizer
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)

        # Generate output using original model
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding (deterministic)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode full output
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Decode only the generated part (skip the input)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_part = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up generated part (remove extra whitespace)
        generated_part = generated_part.strip()

        results.append({
            'input': input_text,
            'actual': actual_answer if actual_answer else '',
            'predicted': generated_part,
            'full_output': full_output
        })

    # Save to CSV
    print(f"\n[5/5] Saving results to {args.output}...")
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"✓ Saved {len(results)} results")

    # Print sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (first 10 examples):")
    print("=" * 70)
    for i, row in df.head(10).iterrows():
        print(f"\nExample {i+1}:")
        print(f"  Input:     {row['input']}")
        print(f"  Actual:    {row['actual']}")
        print(f"  Predicted: {row['predicted']}")

    print("\n" + "=" * 70)
    print("✓ Done! Results saved to:", args.output)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
