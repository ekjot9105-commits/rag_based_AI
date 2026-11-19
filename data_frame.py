import json
import os
import pandas as pd
import joblib

def create_dataframe_embeddings(json_path="embedded_chunks.json", output_joblib="embeddings.joblib"):
    if not os.path.exists(json_path):
        print(f"❌ Error: '{json_path}' not found. Run read_chunks1.py first to generate embeddings.")
        return None

    print(f"📂 Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔄 Converting {len(data)} items to pandas DataFrame...")
    df = pd.DataFrame(data)

    print(f"💾 Saving DataFrame to {output_joblib}...")
    joblib.dump(df, output_joblib)
    print(f"🎉 Successfully saved {output_joblib}! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    create_dataframe_embeddings()
