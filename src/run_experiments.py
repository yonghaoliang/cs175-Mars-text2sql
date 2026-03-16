import os
import warnings

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from config import DRIVE_DB_PATH, OUTPUT_DIR
from config import BASELINE_RAW, FEWSHOT_RAW, COT_RAW, REFINEMENT_RAW
from models import load_sql_model
from utils import get_schema_from_sqlite
from sql_methods import (
    generate_sql_baseline,
    generate_sql_fewshot,
    generate_sql_cot,
    generate_sql_refinement,
)

# Setup

def setup():
    """Load dataset and ensure output directory exists."""
    dataset = load_dataset("spider", split="validation")
    test_df = pd.DataFrame(dataset).copy()   # 1 034 validation questions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Setup complete! Data will be backed up to: {OUTPUT_DIR}")
    return test_df

# Experiment runners

def run_baseline(test_df, tokenizer, model):
    print("🚀 Starting Baseline evaluation...")
    results = []

    for i, row in test_df.iterrows():
        db_id      = row["db_id"]
        question   = row["question"]
        gold_sql   = row["query"]
        db_path    = f"{DRIVE_DB_PATH}/{db_id}/{db_id}.sqlite"

        schema  = get_schema_from_sqlite(db_path)
        pred_sql = generate_sql_baseline(question, schema, tokenizer, model)

        results.append({"ID": i + 1, "Question": question, "Gold_SQL": gold_sql,
                         "Predicted_SQL": pred_sql, "db_id": db_id})

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(BASELINE_RAW.replace("_final", "_partial"), index=False)
            print(f"   [Backup] Baseline progress saved at question {i + 1}.")

    pd.DataFrame(results).to_csv(BASELINE_RAW, index=False)
    print("✅ Baseline evaluation finished and saved!")


def run_fewshot(test_df, tokenizer, model):
    print("🚀 Starting Few-Shot evaluation...")
    results = []

    for i, row in test_df.iterrows():
        db_id    = row["db_id"]
        question = row["question"]
        gold_sql = row["query"]
        db_path  = f"{DRIVE_DB_PATH}/{db_id}/{db_id}.sqlite"

        schema   = get_schema_from_sqlite(db_path)
        pred_sql = generate_sql_fewshot(question, schema, tokenizer, model)

        results.append({"ID": i + 1, "Question": question, "Gold_SQL": gold_sql,
                         "Predicted_SQL": pred_sql, "db_id": db_id})

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(FEWSHOT_RAW.replace("_final", "_partial"), index=False)
            print(f"   [Backup] Few-Shot progress saved at question {i + 1}.")

    pd.DataFrame(results).to_csv(FEWSHOT_RAW, index=False)
    print("✅ Few-Shot evaluation finished and saved!")


def run_cot(test_df, tokenizer, model):
    print("🚀 Starting Chain-of-Thought evaluation...")
    results = []

    for i, row in test_df.iterrows():
        db_id    = row["db_id"]
        question = row["question"]
        gold_sql = row["query"]
        db_path  = f"{DRIVE_DB_PATH}/{db_id}/{db_id}.sqlite"

        schema   = get_schema_from_sqlite(db_path)
        pred_sql = generate_sql_cot(question, schema, tokenizer, model)

        results.append({"ID": i + 1, "Question": question, "Gold_SQL": gold_sql,
                         "Predicted_SQL": pred_sql, "db_id": db_id})

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(COT_RAW.replace("_final", "_partial"), index=False)
            print(f"   [Backup] CoT progress saved at question {i + 1}.")

    pd.DataFrame(results).to_csv(COT_RAW, index=False)
    print("✅ Chain-of-Thought evaluation finished and saved!")


def run_refinement(test_df, tokenizer, model):
    print("🚀 Starting Refinement evaluation...")
    results = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Refining SQL"):
        db_id    = row["db_id"]
        question = row["question"]
        gold_sql = row["query"]
        db_path  = f"{DRIVE_DB_PATH}/{db_id}/{db_id}.sqlite"

        schema      = get_schema_from_sqlite(db_path)
        sql_attempts = generate_sql_refinement(question, schema, db_path, tokenizer, model)

        results.append({
            "ID": i + 1,
            "Question": question,
            "Gold_SQL": gold_sql,
            "Attempt_1_SQL": sql_attempts[0],
            "Attempt_2_SQL": sql_attempts[1],
            "Attempt_3_SQL": sql_attempts[2],
            "Final_Predicted_SQL": sql_attempts[-1],
            "db_id": db_id,
        })

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(REFINEMENT_RAW.replace("_final", "_partial"), index=False)

    pd.DataFrame(results).to_csv(REFINEMENT_RAW, index=False)
    print(f"\n✅ Refinement evaluation complete. Results saved to {OUTPUT_DIR}")

# Entry point

if __name__ == "__main__":
    tokenizer, model = load_sql_model()
    test_df = setup()

    run_baseline(test_df, tokenizer, model)
    run_fewshot(test_df, tokenizer, model)
    run_cot(test_df, tokenizer, model)
    run_refinement(test_df, tokenizer, model)