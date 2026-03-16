import json
import re

import pandas as pd
import torch
from tqdm.auto import tqdm

from utils import execute_query

# Core judge

def execution_aware_llama_judge(
    question: str,
    gold_sql: str,
    pred_sql: str,
    db_path: str,
    judge_tokenizer,
    judge_model,
) -> tuple[int, str]:
    """
    Grade *pred_sql* against *gold_sql* on a 0-10 scale.

    Returns (score, reason).
    """
    if not pred_sql or str(pred_sql).strip() == "":
        return 0, "No SQL generated."

    gold_success, gold_res = execute_query(gold_sql, db_path)
    pred_success, pred_res = execute_query(pred_sql, db_path)

    gold_data_str = (
        str(gold_res.head(5).to_dict(orient="records"))
        if gold_success
        else f"Gold Error: {gold_res}"
    )
    pred_data_str = (
        str(pred_res.head(5).to_dict(orient="records"))
        if pred_success
        else f"SQLite Error: {pred_res}"
    )

    # Perfect match shortcut — skip LLM call
    if gold_success and pred_success and gold_res.equals(pred_res):
        return 10, "Execution results match perfectly."

    system_prompt = (
        "You are an expert SQL database administrator grading a student's SQL query.\n"
        "Your task is to grade the student's SQL on a scale of 0 to 10 based on its logic AND its execution results.\n"
        "- 10: Functionally equivalent to Gold Standard and returns the exact same correct data.\n"
        "- 6-9: Partial credit. Executed successfully but data is slightly off (e.g., missing ORDER BY).\n"
        "- 1-5: Major errors, execution failed with SQLite errors, or returns completely wrong data.\n"
        "- 0: Completely wrong or hallucinates tables/columns.\n\n"
        'You MUST output ONLY a valid JSON object in this exact format:\n'
        '{"score": 5, "reason": "Your reason here"}'
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Gold SQL: {gold_sql}\n"
        f"Gold Result (First 5 rows): {gold_data_str}\n\n"
        f"Student SQL: {pred_sql}\n"
        f"Student Execution Status: {'SUCCESS' if pred_success else 'FAILED'}\n"
        f"Student Result (First 5 rows or Error MSG): {pred_data_str}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = judge_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(judge_model.device)

    with torch.no_grad():
        outputs = judge_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    response = judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    try:
        match = re.search(r"\{.*?\}", response.replace("\n", ""), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return int(data.get("score", 0)), data.get("reason", "Parsed successfully")
        return 0, "Format Error: No JSON found."
    except Exception as exc:
        return 0, f"Parse Error: {exc}"


# Batch graders

def grade_csv_with_execution(
    input_csv: str,
    output_csv: str,
    drive_path: str,
    judge_tokenizer,
    judge_model,
) -> tuple[float, int, float]:
    """
    Grade every row in *input_csv*, write scored results to *output_csv*.

    Returns (total_points, max_points, accuracy_pct).
    """
    df = pd.read_csv(input_csv)
    scores, reasons = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Grading {input_csv.split('/')[-1]}"):
        db_full_path = f"{drive_path}/{row['db_id']}/{row['db_id']}.sqlite"
        score, reason = execution_aware_llama_judge(
            row["Question"],
            row["Gold_SQL"],
            row["Predicted_SQL"],
            db_full_path,
            judge_tokenizer,
            judge_model,
        )
        scores.append(score)
        reasons.append(reason)

    df["AI_Score"] = scores
    df["AI_Reason"] = reasons

    total_points = df["AI_Score"].sum()
    max_points = len(df) * 10
    df.to_csv(output_csv, index=False)

    return total_points, max_points, (total_points / max_points) * 100


def grade_multi_attempts(
    input_csv: str,
    output_csv: str,
    drive_path: str,
    judge_tokenizer,
    judge_model,
) -> tuple[float, float, float]:
    """
    Grade the three refinement attempts per question.

    Returns (avg_score_1, avg_score_2, avg_score_3).
    """
    df = pd.read_csv(input_csv)
    results_data = {
        "Score_1": [], "Reason_1": [],
        "Score_2": [], "Reason_2": [],
        "Score_3": [], "Reason_3": [],
    }

    print(f"🚀 Grading 3 attempts for {len(df)} questions...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grading All Attempts"):
        db_full_path = f"{drive_path}/{row['db_id']}/{row['db_id']}.sqlite"

        for i in range(1, 4):
            pred_sql = row[f"Attempt_{i}_SQL"]
            score, reason = execution_aware_llama_judge(
                row["Question"],
                row["Gold_SQL"],
                pred_sql,
                db_full_path,
                judge_tokenizer,
                judge_model,
            )
            results_data[f"Score_{i}"].append(score)
            results_data[f"Reason_{i}"].append(reason)

    for key, values in results_data.items():
        df[key] = values

    df.to_csv(output_csv, index=False)

    return df["Score_1"].mean(), df["Score_2"].mean(), df["Score_3"].mean()


# Accuracy helper

def calculate_accuracy(file_path: str) -> str:
    """Read a graded CSV and return a formatted accuracy string."""
    try:
        df = pd.read_csv(file_path)
        if "AI_Score" not in df.columns:
            return "Not graded yet (Missing AI_Score column)"
        total_points = df["AI_Score"].sum()
        max_points = len(df) * 10
        accuracy = (total_points / max_points) * 100
        return f"{accuracy:.2f}% ({total_points} / {max_points} points)"
    except Exception as exc:
        return f"Error reading file: {exc}"