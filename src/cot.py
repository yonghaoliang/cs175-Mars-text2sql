def generate_sql_cot(question: str, schema: str, tokenizer, model) -> str:
    prompt = f"""Think step-by-step to generate a valid SQLite query.
 
Schema:
{schema}
 
Question:
{question}
 
Reasoning:
1. Identify tables.
2. Identify columns.
3. Determine joins.
4. Apply filters.
5. Construct SQL.
 
SQL:
"""
    return extract_sql_from_output(run_inference(prompt, tokenizer, model, max_tokens=400))