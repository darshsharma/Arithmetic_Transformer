#!/usr/bin/env python3
"""
Generate digitwise comparison test files for n-digit numbers.

For n digits, this creates:
  - one unconstrained file for each position name
  - one strict file where the first differing digit is forced at that position
  - one equal.txt file with identical numbers

Examples:
  n=3 -> hundreds, tens, units, plus strict variants, plus equal
  n=4 -> thousands, hundreds, tens, units, plus strict variants, plus equal
  n=5 -> ten_thousands, thousands, hundreds, tens, units, plus strict variants, plus equal
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
        cmp = '>'
    elif a < b:
        cmp = '<'
    else:
        cmp = '='
    return f"{a},{b}#{cmp}$"

def random_number_digits(num_digits):
    return [rand_first()] + [rand_digit() for _ in range(num_digits - 1)]

def sample_with_shared_prefix(num_digits, shared_prefix_len, force_diff_at=None):
    shared_prefix = []
    for idx in range(shared_prefix_len):
        shared_prefix.append(rand_first() if idx == 0 else rand_digit())

    a_digits = shared_prefix[:]
    b_digits = shared_prefix[:]

    for idx in range(shared_prefix_len, num_digits):
        if idx == force_diff_at:
            a_digit = rand_first() if idx == 0 else rand_digit()
            choices = [digit for digit in range(1 if idx == 0 else 0, 10) if digit != a_digit]
            b_digit = random.choice(choices)
            a_digits.append(a_digit)
            b_digits.append(b_digit)
        else:
            a_digits.append(rand_first() if idx == 0 else rand_digit())
            b_digits.append(rand_first() if idx == 0 else rand_digit())

    return digits_to_str(a_digits), digits_to_str(b_digits)

def sample_equal(num_digits):
    digits = random_number_digits(num_digits)
    number = digits_to_str(digits)
    return number, number

def build_samplers(num_digits):
    samplers = {}
    labels = place_names(num_digits)

    for idx, label in enumerate(labels):
        shared_prefix_len = idx
        samplers[f"{label}.txt"] = (
            lambda shared_prefix_len=shared_prefix_len: sample_with_shared_prefix(
                num_digits=num_digits,
                shared_prefix_len=shared_prefix_len,
            )
        )
        samplers[f"{label}_strict.txt"] = (
            lambda shared_prefix_len=shared_prefix_len: sample_with_shared_prefix(
                num_digits=num_digits,
                shared_prefix_len=shared_prefix_len,
                force_diff_at=shared_prefix_len,
            )
        )

    samplers["equal.txt"] = lambda: sample_equal(num_digits)
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
    parser.add_argument("--outdir", "-o", default=".", help="output directory for files")
    parser.add_argument("--num_digits", type=int, default=4, help="number of digits in each operand")
    parser.add_argument("--n", type=int, default=1000, help="examples per generated file")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    if args.num_digits < 1:
        raise ValueError("--num_digits must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    samplers = build_samplers(args.num_digits)

    print(f"Generating {len(samplers)} files ({args.n} examples each) for {args.num_digits}-digit comparison...")
    for fname, sampler in samplers.items():
        path = outdir / fname
        generate_file(path, sampler, n=args.n)

    print("Done. Quick stats:")
    for fname in samplers.keys():
        path = outdir / fname
        eq, gt, lt = quick_stats(path)
        print(f"{fname}: = {eq}, > {gt}, < {lt}")

if __name__ == "__main__":
    main()
