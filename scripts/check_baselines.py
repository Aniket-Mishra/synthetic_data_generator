"""Compare the current datasets against the recorded baseline."""

import sys
from synthetic_data_generator.hashing import hash_all_datasets, read_baseline


def compare(baseline: dict[str, str], current: dict[str, str]) -> list[str]:
    problems = []

    for dataset, expected in sorted(baseline.items()):
        actual = current.get(dataset)
        if actual is None:
            problems.append(f"Missing: {dataset}")
        elif actual != expected:
            problems.append(f"Changed: {dataset}")
        else:
            print(f"Good: {dataset}")

    for dataset in sorted(set(current) - set(baseline)):
        print(f"New: {dataset}")

    return problems


def main() -> int:
    problems = compare(read_baseline(), hash_all_datasets())

    if problems:
        print()
        for problem in problems:
            print(problem)
        print(f"{len(problems)} dataset(s) do not match the baseline.")
        return 1

    print("All datasets match baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
