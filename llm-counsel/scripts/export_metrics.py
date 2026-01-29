import argparse
import json
from pathlib import Path

import httpx


def export_metrics(api_url: str, output: Path) -> None:
    resp = httpx.get(f"{api_url}/analytics")
    resp.raise_for_status()
    data = resp.json()

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Metrics exported to {output}")
    print(f"Total queries: {data.get('total_queries', 0)}")
    print(f"Cache hit rate: {data.get('cache', {}).get('hit_rate', 0):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()

    export_metrics(args.api_url, args.output)
