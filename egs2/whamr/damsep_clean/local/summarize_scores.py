"""Aggregate standard ESPnet enhancement score jobs, without re-scoring audio."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    for metric in ("SI_SNR", "SDR", "SIR", "SAR", "STOI"):
        values = []
        for speaker in (1, 2):
            lines = []
            for j in range(1, args.jobs + 1):
                lines.extend(
                    (args.scoring_dir / f"output.{j}" / f"{metric}_spk{speaker}")
                    .read_text()
                    .splitlines()
                )
            lines.sort(key=lambda line: line.split()[0])
            ids = [line.split()[0] for line in lines]
            if len(ids) != len(set(ids)) or not lines:
                raise ValueError("Duplicate or empty score output")
            (args.scoring_dir / f"{metric}_spk{speaker}").write_text(
                "\n".join(lines) + "\n"
            )
            values.extend(float(line.split()[1]) for line in lines)
        result = sum(values) / len(values)
        (args.scoring_dir / f"result_{metric.lower()}.txt").write_text(
            f"{result:.2f}\n"
        )
        print(f"{metric}: {result:.2f}")


if __name__ == "__main__":
    main()
