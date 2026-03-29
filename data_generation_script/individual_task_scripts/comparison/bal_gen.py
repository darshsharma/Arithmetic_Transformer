#!/usr/bin/env python3
"""
Generate balanced train/val/test files for n-digit comparison.

Each example uses the format:
    <a>,<b>#<label>$

For `num_digits = n`, the generator balances across `n + 1` NCID groups:
  Group 0: no shared-prefix constraint
  Group 1: first digit shared
  Group 2: first 2 digits shared
  ...
  Group n-1: first n-1 digits shared
  Group n: identical numbers

Usage:
    python bal_gen.py [--outdir out] [--num_digits 4] [--train_size 45000] [--seed 123]
"""
import random
import argparse
from pathlib import Path

def random_digit(first=False):
    """Return a digit int. If first True, return 1..9, else 0..9."""
    return random.randint(1, 9) if first else random.randint(0, 9)

def digits_to_str(digits):
    return ''.join(str(digit) for digit in digits)

def sample_number_digits(num_digits):
    return [random_digit(first=(idx == 0)) for idx in range(num_digits)]

def sample_group(num_digits, shared_prefix_len):
    if shared_prefix_len < 0 or shared_prefix_len > num_digits:
        raise ValueError(f"shared_prefix_len must be in [0, {num_digits}]")

    if shared_prefix_len == num_digits:
        digits = sample_number_digits(num_digits)
        number = digits_to_str(digits)
        return number, number

    if shared_prefix_len == 0:
        return digits_to_str(sample_number_digits(num_digits)), digits_to_str(sample_number_digits(num_digits))

    shared_prefix = [
        random_digit(first=(idx == 0))
        for idx in range(shared_prefix_len)
    ]
    a_digits = shared_prefix + [random_digit(False) for _ in range(num_digits - shared_prefix_len)]
    b_digits = shared_prefix + [random_digit(False) for _ in range(num_digits - shared_prefix_len)]
    return digits_to_str(a_digits), digits_to_str(b_digits)

def make_example(num_digits, shared_prefix_len):
    a, b = sample_group(num_digits, shared_prefix_len)
    if a > b:
        comp = '>'
    elif a < b:
        comp = '<'
    else:
        comp = '='
    return f"{a},{b}#{comp}$"

def balanced_group_counts(total_examples, num_groups):
    counts = [total_examples // num_groups] * num_groups
    for idx in range(total_examples % num_groups):
        counts[idx] += 1
    return counts

def generate_file(path: Path, n_examples: int, num_digits: int):
    group_counts = balanced_group_counts(n_examples, num_digits + 1)
    examples = []
    for shared_prefix_len, count in enumerate(group_counts):
        examples.extend(
            make_example(num_digits=num_digits, shared_prefix_len=shared_prefix_len)
            for _ in range(count)
        )
    random.shuffle(examples)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example + "\n")

    return group_counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", "-o", default=".", help="output directory")
    p.add_argument("--num_digits", type=int, default=4, help="number of digits in each operand")
    p.add_argument("--train_size", type=int, default=45000, help="number of train examples")
    p.add_argument("--val_size", type=int, default=5000, help="number of validation examples")
    p.add_argument("--test_size", type=int, default=5000, help="number of test examples")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()

    if args.num_digits < 1:
        raise ValueError("--num_digits must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train.txt"
    val_path = outdir / "val.txt"
    test_path = outdir / "test.txt"

    print("Generating datasets...")
    train_group_counts = generate_file(train_path, args.train_size, args.num_digits)
    val_group_counts = generate_file(val_path, args.val_size, args.num_digits)
    test_group_counts = generate_file(test_path, args.test_size, args.num_digits)
    print(f"Saved: {train_path} ({train_path.stat().st_size} bytes)")
    print(f"Saved: {val_path}   ({val_path.stat().st_size} bytes)")
    print(f"Saved: {test_path}  ({test_path.stat().st_size} bytes)")
    # Optional brief counts of equality vs others
    def quick_stats(path):
        eq = 0
        gt = 0
        lt = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format: a,b#c$
                try:
                    tail = line.split("#", 1)[1]
                except IndexError:
                    continue
                if tail.startswith("=$"):
                    eq += 1
                elif tail.startswith(">$"):
                    gt += 1
                elif tail.startswith("<$"):
                    lt += 1
        return eq, gt, lt

    print(f"Balanced over {args.num_digits + 1} groups for {args.num_digits}-digit comparison.")
    print(f"train group counts: {train_group_counts}")
    print(f"val group counts:   {val_group_counts}")
    print(f"test group counts:  {test_group_counts}")

    for path in (train_path, val_path, test_path):
        eq, gt, lt = quick_stats(path)
        print(f"{path.name}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
