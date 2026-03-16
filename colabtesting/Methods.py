from helpers import extract_sql_from_output, execute_query
from models  import run_inference

def generate_sql_baseline(question, schema):
    prompt = f"""### Task
Generate a SQL query.

### Database Schema
{schema}

### Question
{question}

### SQL
"""
    return extract_sql_from_output(run_inference(prompt))

def generate_sql_fewshot(question, schema):
    prompt = f"""Generate a strict SQLite query.
RULES:
1. ONLY use the exact table and column names from the schema provided. Do NOT invent or guess columns.
2. ONLY return the specific columns requested. You MAY include COUNT() or AVG() columns when the question explicitly asks for a count, number, or average.
3. Use simple logic. Use ORDER BY ... LIMIT 1 for maximums/minimums.

=== EXAMPLE 1 ===
Schema:
CREATE TABLE Singer (Singer_ID int, Name text, Country text, Age int);
CREATE TABLE Song (Song_ID int, Title text, Singer_ID int);
Question: List all song titles by singers above the average age.
SQL: SELECT Title FROM Song JOIN Singer ON Song.Singer_ID = Singer.Singer_ID WHERE Age > (SELECT AVG(Age) FROM Singer);

=== EXAMPLE 2 ===
Schema:
CREATE TABLE stadium (Stadium_ID int, Location text, Name text, Capacity int);
CREATE TABLE concert (concert_ID int, concert_Name text, Theme text, Stadium_ID text, Year text);
Question: Show the stadium names without any concert.
SQL: SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert);

=== EXAMPLE 3 ===
Schema:
CREATE TABLE student (StuID int, LName text, Fname text, Age int, Sex text);
CREATE TABLE pets (PetID int, PetType text, pet_age int, weight real);
CREATE TABLE has_pet (StuID int, PetID int);
Question: Find number of pets owned by students who are older than 20.
SQL: SELECT count(*) FROM student JOIN has_pet ON student.StuID = has_pet.StuID WHERE student.Age > 20;

=== YOUR TASK ===
Schema:
{schema}

Question: {question}
SQL:
"""
    return extract_sql_from_output(run_inference(prompt))

def generate_sql_cot(question, schema):
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
    return extract_sql_from_output(run_inference(prompt, max_tokens=400))

def ai_judge_diagnostic_parser(question, schema, failed_sql, execution_status):
    """Ask the SQLCoder model to diagnose and suggest a fix for a SQL query."""
    prompt = f"""You are an expert SQL Debugger and Logic Judge.
A model tried to answer this question: {question}
Using this schema: {schema}
It generated this SQL: {failed_sql}
Execution Status: {execution_status}

Analyze what is wrong. If the query perfectly answers the question and has no logical errors, start your response with the exact words "LOOKS GOOD".
Otherwise, provide a clear, one-sentence instruction for the model to fix the SQL.

Fix Instruction:"""
    return run_inference(prompt, max_tokens=150).strip()


def generate_sql_refinement(question, schema, db_path, max_attempts=3):
    """Generate SQL with up to 3 self-refinement attempts using the AI judge."""
    attempts = []
    current_sql = generate_sql_fewshot(question, schema)
    attempts.append(current_sql)

    for attempt in range(max_attempts):
        success, exec_result = execute_query(current_sql, db_path)
        judge_input_status = (
            "Execution Succeeded. Logic check required."
            if success else
            f"Execution Failed: {exec_result}"
        )

        advice = ai_judge_diagnostic_parser(question, schema, current_sql, judge_input_status)

        # Early stop if judge is satisfied
        if success and ("LOOKS GOOD" in advice.upper() or "PERFECT" in advice.upper()):
            break

        if attempt == max_attempts - 1:
            break

        b = "`" * 3
        refine_prompt = f"""The current SQL query needs improvement.
QUESTION: {question}
SCHEMA: {schema}
CURRENT SQL: {current_sql}
EXECUTION STATUS: {judge_input_status}
JUDGE OPINION: {advice}

Follow the Judge's advice and provide the improved SQL inside {b}sql ... {b} blocks.
Corrected SQL:"""

        current_sql = extract_sql_from_output(run_inference(refine_prompt))
        attempts.append(current_sql)

    # Pad to exactly 3 attempts if finished early
    while len(attempts) < 3:
        attempts.append(attempts[-1])

    return attempts