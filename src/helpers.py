import re
import sqlite3
import threading

import pandas as pd
import torch

# SQL extraction

def extract_sql_from_output(text: str) -> str:
    """Parse a raw model output and return only the SQL statement."""
    if not isinstance(text, str):
        return ""
    if "[SQL]" in text:
        return text.split("[SQL]")[-1].strip()
    code_block = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    select_match = re.search(r"(SELECT .*?;)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    return text.strip()

# SQLite execution

def execute_query(sql: str, db_path: str, timeout_seconds: int = 5):
    """
    Execute *sql* against the SQLite database at *db_path*.

    Returns (success: bool, data: DataFrame | str).
    A background thread enforces the timeout so runaway cross-joins are killed.
    """
    result = {
        "success": False,
        "data": "Timeout: Query took too long to execute (likely a massive cross-join).",
    }

    def run_target():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            cols = [desc[0] for desc in cursor.description]
            data = cursor.fetchmany(10)          # cap at 10 rows to save memory
            conn.close()
            result["success"] = True
            result["data"] = pd.DataFrame(data, columns=cols)
        except Exception as exc:
            result["success"] = False
            result["data"] = str(exc)

    thread = threading.Thread(target=run_target)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return False, result["data"]

    return result["success"], result["data"]


# Schema extraction

def get_schema_from_sqlite(db_path: str) -> str:
    """Return the CREATE TABLE statements for every user-created table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = cursor.fetchall()
        conn.close()
        return "\n".join([t[0] + ";" for t in tables if t[0]])
    except Exception as exc:
        return f"Error reading schema: {exc}"


# Model inference wrapper

def run_inference(prompt: str, tokenizer, model, max_tokens: int = 300) -> str:
    """Generate text from *prompt* and return only the newly generated portion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):]