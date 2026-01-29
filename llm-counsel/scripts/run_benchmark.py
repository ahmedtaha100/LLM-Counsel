import argparse
import asyncio
import json
from pathlib import Path

from evaluation.benchmarks import BenchmarkRunner

SAMPLE_DATASET = [
    {"query": "What is the capital of France?", "expected": "Paris"},
    {"query": "What is 15 * 7?", "expected": "105"},
    {"query": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
    {"query": "What is the chemical symbol for gold?", "expected": "Au"},
    {"query": "In what year did World War II end?", "expected": "1945"},
]


async def main(output: Path, dataset_path: Path | None) -> None:
    runner = BenchmarkRunner()

    if dataset_path and dataset_path.exists():
        dataset = runner.load_dataset(dataset_path)
    else:
        dataset = SAMPLE_DATASET

    print(f"Running benchmark on {len(dataset)} queries...")
    results = await runner.run_benchmark(dataset)

    output_data = {}
    for mode, summary in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Accuracy: {summary.accuracy:.2%}")
        print(f"  Total Cost: ${summary.total_cost:.4f}")
        print(f"  Avg Latency: {summary.avg_latency:.0f}ms")

        output_data[mode] = {
            "total": summary.total,
            "correct": summary.correct,
            "accuracy": summary.accuracy,
            "total_cost": summary.total_cost,
            "avg_latency": summary.avg_latency,
        }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    asyncio.run(main(args.output, args.dataset))
