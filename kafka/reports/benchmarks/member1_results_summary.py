import csv


CSV_FILE = "member1_experiment_results.csv"


def main():
    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    print()
    print("=" * 100)
    print("MEMBER 1 — KEY EXPERIMENT RESULTS")
    print("=" * 100)

    print(
        f"{'Experiment':<25}"
        f"{'Configuration':<30}"
        f"{'Success':>10}"
        f"{'Failed':>10}"
    )

    print("-" * 100)

    for row in rows:
        print(
            f"{row['Experiment']:<25}"
            f"{row['Configuration']:<30}"
            f"{row['Successful']:>10}"
            f"{row['Failed']:>10}"
        )

    print("=" * 100)
    print(f"Total experiment records: {len(rows)}")


if __name__ == "__main__":
    main()
