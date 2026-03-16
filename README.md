# cs175-Mars-text2sql

Project Name: Mars
Team Members: Yonghao Liang, Wenshuo Wang, Jacky Zhang

Mention the external libraries your team used
Libraries used:
transformers (https://huggingface.co/docs/transformers)
bitsandbytes (https://github.com/TimDettmers/bitsandbytes)
accelerate (https://huggingface.co/docs/accelerate)
datasets (https://huggingface.co/docs/datasets)
peft (https://huggingface.co/docs/peft)
pandas (https://pandas.pydata.org)
torch (https://pytorch.org)
tqdm (https://tqdm.github.io)

List the publicly available code(s) you used in your project.
Publicly available codes used:
SQLCoder-7B-2 (https://huggingface.co/defog/sqlcoder-7b-2). Used unmodified as our base Text-to-SQL generation model.
Meta-Llama-3-8B-Instruct (https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct). Used unmodified as our AI Judge for scoring.
Spider Dataset (https://huggingface.co/datasets/spider). Used unmodified for validation questions and SQLite database testing.

List the code(s) written entirely by your team.
Scripts/functions written by our team:
(Note: These Python files are located in the src/ directory )

evaluate.py: Implements the Llama-3 execution-aware judge and grading loops for all experiment runs. (110 lines)
Helpers.py: Contains utility functions for threaded SQLite execution, SQL regex extraction, and schema retrieval. (70 lines)
Methods.py: Implements prompt templates and generation logic for Baseline, Few-Shot, CoT, and Agentic Refinement. (100 lines)
Models.py: Handles initialization and 4-bit NF4 quantization loading for both SQLCoder and Llama-3. (50 lines)
