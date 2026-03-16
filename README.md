# cs175-Mars-text2sql

Project Name: Mars

Team Members: Yonghao Liang, Wenshuo Wang, Jacky Zhang

Mention the external libraries your team used


# Libraries used:

transformers (https://huggingface.co/docs/transformers)

bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

accelerate (https://huggingface.co/docs/accelerate)

datasets (https://huggingface.co/docs/datasets)

peft (https://huggingface.co/docs/peft)

pandas (https://pandas.pydata.org)

torch (https://pytorch.org)

tqdm (https://tqdm.github.io)


# List the publicly available code(s) you used in your project.

Publicly available codes used:

SQLCoder-7B-2 (https://huggingface.co/defog/sqlcoder-7b-2). Used unmodified as our base Text-to-SQL generation model.

Meta-Llama-3-8B-Instruct (https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct). Used unmodified as our AI Judge for scoring.

Spider Dataset (https://huggingface.co/datasets/spider). Used unmodified for validation questions and SQLite database testing.


# List the code(s) written entirely by your team.

Scripts/functions written by our team:

(Note: These Python files are located in the src/ directory )

Refinement.py: An iterative refinement loop for SQL generation (80 lines)

analysis.py: Analysis and plot the graph of the four methods (180 lines)

baseline.py: Baseline prompt to the SQLcoder (13 lines)

cot.py: COT prompt to the SQLcoder (20 lines)

few_shot.py: Few-shots prompt to the SQLcoder (37 lines)

grade_results.py: Add the points after finishing grading the SQL query (90 lines)

grading.py: Use llama as a ai judge to grade the SQL query (200 lines)

helpers.py: Helper functions that can reuse to save time (90 lines)

models.py: Functions to load the models (50 lines)

run_experiments.py: Function to let the SQLcoder to run all four methods (140 lines)
