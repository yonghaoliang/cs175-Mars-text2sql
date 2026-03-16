import re
import json
import torch
import pandas as pd
from tqdm.auto import tqdm
from Helpers import execute_query, get_db_path

def execution_aware_llama_judge(question, gold_sql, pred_sql, db_path):
    if not pred_sql or str(pred_sql).strip() == "":
        return 0, "No SQL generated."

    gold_success, gold_res = execute_query(gold_sql, db_path)
    pred_success, pred_res = execute_query(pred_sql, db_path)

    gold_data_str = str(gold_res.head(5).to_dict(orient='records')) if gold_success else f"Gold Error: {gold_res}"
    pred_data_str = str(pred_res.head(5).to_dict(orient='records')) if pred_success else f"SQLite Error: {pred_res}"

    if gold_success and pred_success:
        try:
            gold_vals = gold_res.reset_index(drop=True)
            pred_vals = pred_res.reset_index(drop=True)
            if gold_vals.shape == pred_vals.shape:
                pred_vals.columns = gold_vals.columns
            if gold_vals.equals(pred_vals):
                return 10, "Execution results match perfectly."
        except Exception:
            pass

    system_prompt = """You are an expert SQL database administrator grading a student's SQL query.
Your task is to grade the student's SQL on a scale of 0 to 10 based on its logic AND its execution results.
- 10: Functionally equivalent to Gold Standard and returns the exact same correct data.
- 6-9: Partial credit. Executed successfully but data is slightly off (e.g., missing ORDER BY).
- 1-5: Major errors, execution failed with SQLite errors, or returns completely wrong data.
- 0: Completely wrong or hallucinates tables/columns.

You MUST output ONLY a valid JSON object in this exact format:
{"score": 5, "reason": "Your reason here"}"""

    user_prompt = f"""Question: {question}

Gold SQL: {gold_sql}
Gold Result (First 5 rows): {gold_data_str}

Student SQL: {pred_sql}
Student Execution Status: {"SUCCESS" if pred_success else "FAILED"}
Student Result (First 5 rows or Error MSG): {pred_data_str}"""

    from Models import judge_model, judge_tokenizer

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    inputs = judge_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(judge_model.device)

    with torch.no_grad():
        outputs = judge_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id
        )

    response = judge_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    try:
        match = re.search(r'\{.*?\}', response.replace('\n', ''), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return int(data.get('score', 0)), data.get('reason', 'Parsed successfully')
        return 0, "Format Error: No JSON found."
    except Exception as e:
        return 0, f"Parse Error: {str(e)}"

def grade_csv_with_execution(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    scores, reasons = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Grading {input_csv.split('/')[-1]}"):
        db_full_path = get_db_path(row['db_id'])
        score, reason = execution_aware_llama_judge(
            row['Question'], row['Gold_SQL'], row['Predicted_SQL'], db_full_path
        )
        scores.append(score)
        reasons.append(reason)

    df['AI_Score']  = scores
    df['AI_Reason'] = reasons

    total_points = df['AI_Score'].sum()
    max_points   = len(df) * 10

    df.to_csv(output_csv, index=False)
    return total_points, max_points, (total_points / max_points) * 100

def calculate_accuracy(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'AI_Score' not in df.columns:
            return "Not graded yet (Missing AI_Score column)"
        total_points = df['AI_Score'].sum()
        max_points   = len(df) * 10
        accuracy     = (total_points / max_points) * 100
        return f"{accuracy:.2f}% ({total_points} / {max_points} points)"
    except Exception as e:
        return f"Error reading file: {e}"

def grade_multi_attempts(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results_data = {
        "Score_1": [], "Reason_1": [],
        "Score_2": [], "Reason_2": [],
        "Score_3": [], "Reason_3": []
    }

    print(f"🚀 Grading 3 attempts for {len(df)} questions...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grading All Attempts"):
        db_full_path = get_db_path(row['db_id'])
        for i in range(1, 4):
            pred_sql = row[f"Attempt_{i}_SQL"]
            score, reason = execution_aware_llama_judge(
                row['Question'], row['Gold_SQL'], pred_sql, db_full_path
            )
            results_data[f"Score_{i}"].append(score)
            results_data[f"Reason_{i}"].append(reason)

    for key, values in results_data.items():
        df[key] = values

    df.to_csv(output_csv, index=False)
    return df['Score_1'].mean(), df['Score_2'].mean(), df['Score_3'].mean()