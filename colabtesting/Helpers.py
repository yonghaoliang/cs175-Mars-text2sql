import sqlite3
import re
import threading
import pandas as pd
import os

DB_MAP = {
    "concert_singer": "concert_singer",
    "flight_2":       "flight_2",
    "tvshow":         "tvshow",
}

DATA_ROOT = None

def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = path

def get_db_path(db_id):
    if DATA_ROOT is None:
        raise RuntimeError("DATA_ROOT not set. Call set_data_root(path) first.")
    folder = DB_MAP.get(db_id, db_id)
    return os.path.join(DATA_ROOT, folder, f"{folder}.sqlite")

def extract_sql_from_output(text):
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

def execute_query(sql, db_path, timeout_seconds=5):
    result = {"success": False, "data": "Timeout: Query took too long to execute (likely a massive cross-join)."}

    def run_target():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            cols = [desc[0] for desc in cursor.description]
            data = cursor.fetchmany(10)
            conn.close()
            result["success"] = True
            result["data"] = pd.DataFrame(data, columns=cols)
        except Exception as e:
            result["success"] = False
            result["data"] = str(e)

    thread = threading.Thread(target=run_target)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return False, result["data"]
    return result["success"], result["data"]

def get_schema_from_sqlite(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        conn.close()
        return "\n".join([t[0] + ";" for t in tables if t[0]])
    except Exception as e:
        return f"Error reading schema: {str(e)}"