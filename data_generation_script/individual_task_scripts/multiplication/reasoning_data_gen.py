import argparse
import os


def _build_place_value_terms(number_str: str) -> list[int]:
    terms = []
    num_digits = len(number_str)
    for idx, ch in enumerate(number_str):
        digit = int(ch)
        if digit == 0:
            continue
        place_value = 10 ** (num_digits - idx - 1)
        terms.append(digit * place_value)
    return terms


def create_multiplication_reasoning_data(file_path: str, output_path: str) -> None:
    """
    Convert multiplication examples of the form ``a*b=result$`` into a scratchpad format.

    The function recomputes the true product from the operands, so it works whether the
    input file stores the result in plain order or reversed order.

    Example:
        3856*9=29904$
    becomes
        3856*9=(3000*9)+(800*9)+(50*9)+(6*9)=27000+7200+450+54=40743$
    """
    with open(file_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for raw_line in fin:
            clean_line = raw_line.strip().replace("$", "")
            if "=" not in clean_line or "*" not in clean_line:
                continue

            left_side, _ = clean_line.split("=", 1)
            a_str, b_str = left_side.split("*", 1)

            try:
                a_value = int(a_str)
                b_value = int(b_str)
            except ValueError:
                continue

            place_terms = _build_place_value_terms(a_str)
            if not place_terms:
                place_terms = [0]

            expanded_terms = [f"({term}*{b_value})" for term in place_terms]
            partial_products = [str(term * b_value) for term in place_terms]
            product = a_value * b_value
            reversed_product = str(product)[::-1]

            transformed_line = (
                f"{left_side}="
                f"{'+'.join(expanded_terms)}="
                f"{'+'.join(partial_products)}="
                f"{reversed_product}$"
            )
            fout.write(transformed_line + "\n")

    print(f"Transformed data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True)
    parser.add_argument("--output_file_path", required=True)
    args = parser.parse_args()

    in_path = os.path.abspath(args.input_file_path)
    out_path = os.path.abspath(args.output_file_path)

    if in_path == out_path:
        raise SystemExit(
            f"Error: output_file_path must be different from input_file_path (got {out_path}). "
            "Writing to the same file will truncate it."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    create_multiplication_reasoning_data(in_path, out_path)


if __name__ == "__main__":
    main()
