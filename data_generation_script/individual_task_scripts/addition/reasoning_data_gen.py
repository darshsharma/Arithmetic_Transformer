import argparse

def create_reasoning_data_1(file_path, output_path):
    """
    Transforms raw reasoning data into a format suitable for training.

    Args:
        file_path: Path to the raw data file
        output_path: Path to save the transformed data
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    transformed_data = []
    for line in lines:
        clean_line = line.strip().replace('$', '')
        if '=' in clean_line:
            left_side, result = clean_line.split('=')
            operands = left_side.split('+')
        # 2. Lists to hold the digits for each place value
        hundreds = []
        tens = []
        units = []

        for number in operands:
            n = int(number)
            
            # Extract digits mathematically
            h = (n // 100) % 10
            t = (n // 10) % 10
            u = n % 10
            
            # Add to our lists as strings so we can join them with '+' later
            hundreds.append(str(h))
            tens.append(str(t))
            units.append(str(u))
            # 3. Create the transformed line

        breakdown = (
            f"100({'+'.join(hundreds)})+"
            f"10({'+'.join(tens)})+"
            f"1({'+'.join(units)})"
        )
        transformed_line = f"{left_side}={breakdown}={result}$"    

        transformed_data.append(transformed_line)
    with open(output_path, 'w') as f:
        for item in transformed_data:
            f.write(f"{item}\n")
    print(f"Transformed data saved to {output_path}")


def create_reasoning_data_2(file_path, output_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    transformed_data = []
    for line in lines:
        clean_line = line.strip().replace('$', '')
        if '=' in clean_line:
            left_side, result = clean_line.split('=')
            operands = left_side.split('+')
        # 2. Lists to hold the digits for each place value
        hundreds = []
        tens = []
        units = []

        for number in operands:
            n = int(number)
            
            # Extract digits mathematically
            h = (n // 100) % 10
            t = (n // 10) % 10
            u = n % 10
            
            # Add to our lists as strings so we can join them with '+' later
            hundreds.append(str(h))
            tens.append(str(t))
            units.append(str(u))
            # 3. Create the transformed line

        breakdown = (
            f"100({'+'.join(hundreds)})+"
            f"10({'+'.join(tens)})+"
            f"1({'+'.join(units)})"
        )

        breakdown_2 = (f"100({sum([int(h) for h in hundreds])})+"
                        f"10({sum([int(t) for t in tens])})+"
                        f"1({sum([int(u) for u in units])})")

        transformed_line = f"{left_side}={breakdown}={breakdown_2}={result}$"    

        transformed_data.append(transformed_line)
    with open(output_path, 'w') as f:
        for item in transformed_data:
            f.write(f"{item}\n")
    print(f"Transformed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True)
    parser.add_argument("--output_file_path", required=True)
    parser.add_argument("--mode", default="plain")
    parsed_args = parser.parse_args()
    if parsed_args.mode == "plain":
        create_reasoning_data_1(parsed_args.input_file_path, parsed_args.output_file_path)
    elif parsed_args.mode == "plain_v2":
        create_reasoning_data_2(parsed_args.input_file_path, parsed_args.output_file_path)

if __name__ == "__main__":
    main()

    