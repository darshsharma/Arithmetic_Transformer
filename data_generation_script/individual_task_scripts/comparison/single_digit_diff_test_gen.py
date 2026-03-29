#!/usr/bin/env python3
"""
Generate one `*_diff_only.txt` file per digit position for n-digit comparison.

For each file, exactly one digit position differs and all others match.
Examples:
  n=3 -> hundreds_diff_only, tens_diff_only, units_diff_only
  n=4 -> thousands_diff_only, hundreds_diff_only, tens_diff_only, units_diff_only
  n=5 -> ten_thousands_diff_only, thousands_diff_only, hundreds_diff_only, tens_diff_only, units_diff_only
"""
import random
import argparse
from pathlib import Path

PLACE_NAMES_FROM_RIGHT = [
    "units",
    "tens",
    "hundreds",
    "thousands",
    "ten_thousands",
    "hundred_thousands",
    "millions",
]

def rand_first():
    return random.randint(1, 9)

def rand_digit():
    return random.randint(0, 9)

def digits_to_str(digits):
    return ''.join(str(digit) for digit in digits)

def place_names(num_digits):
    if num_digits > len(PLACE_NAMES_FROM_RIGHT):
        raise ValueError(f"num_digits={num_digits} is not supported; max is {len(PLACE_NAMES_FROM_RIGHT)}")
    return list(reversed(PLACE_NAMES_FROM_RIGHT[:num_digits]))

def format_example(a, b):
    if a > b:
        cmp_sym = '>'
    elif a < b:
        cmp_sym = '<'
    else:
        cmp_sym = '='
    return f"{a},{b}#{cmp_sym}$"

def sample_diff_only(num_digits, diff_idx):
    a_digits = []
    b_digits = []
    for idx in range(num_digits):
        if idx == diff_idx:
            a_digit = rand_first() if idx == 0 else rand_digit()
            choices = [digit for digit in range(1 if idx == 0 else 0, 10) if digit != a_digit]
            b_digit = random.choice(choices)
        else:
            a_digit = rand_first() if idx == 0 else rand_digit()
            b_digit = a_digit
        a_digits.append(a_digit)
        b_digits.append(b_digit)
    return digits_to_str(a_digits), digits_to_str(b_digits)

def build_samplers(num_digits):
    samplers = {}
    for diff_idx, label in enumerate(place_names(num_digits)):
        samplers[f"{label}_diff_only.txt"] = (
            lambda diff_idx=diff_idx: sample_diff_only(num_digits=num_digits, diff_idx=diff_idx)
        )
    return samplers

def generate_file(path: Path, sampler, n=1000):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            a, b = sampler()
            f.write(format_example(a, b) + "\n")

def quick_stats(path: Path):
    eq = gt = lt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", "-o", default=".", help="output directory")
    parser.add_argument("--num_digits", type=int, default=4, help="number of digits in each operand")
    parser.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    parser.add_argument("--n", type=int, default=1000, help="examples per file (default 1000)")
    args = parser.parse_args()

    if args.num_digits < 1:
        raise ValueError("--num_digits must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    samplers = build_samplers(args.num_digits)

    for fname, sampler in samplers.items():
        generate_file(outdir / fname, sampler, n=args.n)

    print("Generated files:")
    for fname in samplers.keys():
        eq, gt, lt = quick_stats(outdir / fname)
        print(f"  {fname}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
