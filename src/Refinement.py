def _ai_judge_diagnostic_parser(
    question: str,
    schema: str,
    failed_sql: str,
    execution_status: str,
    tokenizer,
    model,
) -> str:
    """Ask the SQL model to diagnose what went wrong and suggest a fix."""
    prompt = f"""You are an expert SQL Debugger and Logic Judge.
A model tried to answer this question: {question}
Using this schema: {schema}
It generated this SQL: {failed_sql}
Execution Status: {execution_status}
 
Analyze what is wrong. If the query perfectly answers the question and has no logical errors, start your response with the exact words "LOOKS GOOD".
Otherwise, provide a clear, one-sentence instruction for the model to fix the SQL.
 
Fix Instruction:"""
    return run_inference(prompt, tokenizer, model, max_tokens=150).strip()
 
 
def generate_sql_refinement(
    question: str,
    schema: str,
    db_path: str,
    tokenizer,
    model,
    max_attempts: int = 3,
) -> list[str]:
    """
    Iteratively refine the SQL using feedback from the judge.
 
    Returns a list of exactly *max_attempts* SQL strings (padded with the last
    value if the loop terminates early).
    """
    attempts = []
 
    # Attempt 1: initial few-shot generation
    current_sql = generate_sql_fewshot(question, schema, tokenizer, model)
    attempts.append(current_sql)
 
    for attempt in range(max_attempts):
        success, exec_result = execute_query(current_sql, db_path)
        judge_input_status = (
            "Execution Succeeded. Logic check required."
            if success
            else f"Execution Failed: {exec_result}"
        )
 
        advice = _ai_judge_diagnostic_parser(
            question, schema, current_sql, judge_input_status, tokenizer, model
        )
 
        # Early stop if judge is satisfied
        if success and ("LOOKS GOOD" in advice.upper() or "PERFECT" in advice.upper()):
            break
 
        # Don't refine after the last allowed attempt
        if attempt == max_attempts - 1:
            break
 
        backticks = "`" * 3
        refine_prompt = f"""The current SQL query needs improvement.
QUESTION: {question}
SCHEMA: {schema}
CURRENT SQL: {current_sql}
EXECUTION STATUS: {judge_input_status}
JUDGE OPINION: {advice}
 
Follow the Judge's advice and provide the improved SQL inside {backticks}sql ... {backticks} blocks.
Corrected SQL:"""
 
        current_sql = extract_sql_from_output(
            run_inference(refine_prompt, tokenizer, model)
        )
        attempts.append(current_sql)
 
    # Pad to exactly max_attempts entries
    while len(attempts) < max_attempts:
        attempts.append(attempts[-1])
 
    return attempts