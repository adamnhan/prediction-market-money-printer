#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


def aggregate_csvs(input_dir: Path, output_path: Path) -> int:
    csv_paths = sorted(p for p in input_dir.iterdir() if p.suffix == ".csv")
    if not csv_paths:
        print(f"No CSV files found in {input_dir}")
        return 1

    lines = []
    for csv_path in csv_paths:
        lines.append(f"=== {csv_path.name} ===")
        for row in read_csv_rows(csv_path):
            lines.append("\t".join(row))
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate analysis output CSVs into a single text file."
    )
    parser.add_argument(
        "--input-dir",
        default="market_maker/analysis_outputs",
        help="Directory containing analysis CSVs.",
    )
    parser.add_argument(
        "--output",
        default="market_maker/analysis_outputs/analysis_outputs.txt",
        help="Path to the aggregated text output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    return aggregate_csvs(input_dir, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
