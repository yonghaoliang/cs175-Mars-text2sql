def generate_sql_fewshot(question: str, schema: str, tokenizer, model) -> str:
    prompt = f"""Generate a strict SQLite query.
RULES:
1. ONLY use the exact table and column names from the schema provided. Do NOT invent or guess columns.
2. ONLY return the specific columns requested. Do not return extra COUNT() or AVG() columns unless explicitly asked.
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
    return extract_sql_from_output(run_inference(prompt, tokenizer, model))