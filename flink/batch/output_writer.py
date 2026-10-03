from pathlib import Path

def write_features(df, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        path = output / "historical_features.parquet"
        df.to_parquet(path, index=False)
    except Exception:
        path = output / "historical_features.csv"
        df.to_csv(path, index=False)
    return path
