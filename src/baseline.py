def generate_sql_baseline(question: str, schema: str, tokenizer, model) -> str:
    prompt = f"""### Task
Generate a SQL query.
 
### Database Schema
{schema}
 
### Question
{question}
 
### SQL
"""
    return extract_sql_from_output(run_inference(prompt, tokenizer, model))