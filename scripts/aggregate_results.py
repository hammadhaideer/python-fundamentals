import os
import json
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    result_files = glob(os.path.join(args.results_dir, "*.json"))
    all_results = {}

    for f in result_files:
        with open(f, "r") as fp:
            data = json.load(fp)
            name = os.path.basename(f).replace(".json", "")
            all_results[name] = data

    print("\n" + "="*60)
    print("MEDCLIP REPRODUCTION RESULTS")
    print("="*60)

    for name, res in all_results.items():
        print(f"\n{name}:")
        for k, v in res.items():
            print(f"  {k}: {v}")

    out_path = os.path.join(args.results_dir, "aggregated.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
