#!/usr/bin/env python3
"""
Plot digit-wise error rates (0-1) from test results CSV.

Usage:
    python plot_error_rates.py <csv_path> [options]

Example:
    python plot_error_rates.py results/4_operands_0_to_999_uniform/pythia_out_plain/test_results.csv
    python plot_error_rates.py results/4_operands_0_to_999_uniform/pythia_out_plain/test_results.csv --step_size 2000
"""

import argparse
from pathlib import Path
import result_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Plot digit-wise error RATES (0-1) from test results CSV."
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV with 'actual' and 'pred_iter_<N>' columns"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Include every Nth iteration (default: 1 = all iterations)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start from this iteration (default: 0)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800000,
        help="Maximum iteration to include (default: 800000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: <csv_path>.digit_error_rates.png)"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save error rates to CSV file"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot instead of saving"
    )

    args = parser.parse_args()

    # Validate CSV path
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Set output path
    if args.output:
        fig_path = args.output
    else:
        fig_path = csv_path.with_name(csv_path.stem + "_digit_error_rates.png")

    print(f"Reading CSV: {csv_path}")
    print(f"Configuration:")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Offset: {args.offset}")
    print(f"  - Max steps: {args.max_steps}")

    # Analyze and plot
    try:
        df, rates_by_iter = result_analysis.analyze_csv_error_rates(
            csv_path=csv_path,
            step_size=args.step_size,
            offset=args.offset,
            max_steps=args.max_steps,
            actual_col="actual",
            save_fig=(not args.show),
            fig_path=fig_path if not args.show else None,
            save_rates_csv=args.save_csv
        )

        if not args.show:
            print(f"✓ Plot saved to: {fig_path}")

        if args.save_csv:
            csv_output = csv_path.with_name(csv_path.stem + "_digit_error_rates.csv")
            print(f"✓ Error rates CSV saved to: {csv_output}")

        # Print summary
        print(f"\nAnalyzed {len(df)} test examples across {len(rates_by_iter)} iterations")

        # Show latest error rates
        if rates_by_iter:
            latest_iter = max(rates_by_iter.keys())
            latest_rates = rates_by_iter[latest_iter]
            print(f"\nLatest error rates (iteration {latest_iter}):")
            for place, rate in sorted(latest_rates.items(), key=lambda x: int(x[0].rstrip('stndrh'))):
                place_name = {
                    "0th": "units",
                    "1st": "tens",
                    "2nd": "hundreds",
                    "3rd": "thousands"
                }.get(place, place)
                print(f"  {place_name:15s}: {rate:.4f} ({rate*100:.2f}%)")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
