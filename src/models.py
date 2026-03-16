import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import SQL_MODEL_NAME, JUDGE_MODEL_ID


def load_sql_model():
    """Load SQLCoder-7B in 4-bit quantization for SQL generation."""
    print(f"🚀 Loading {SQL_MODEL_NAME}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(SQL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        SQL_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("✅ SQL model loaded successfully!")
    return tokenizer, model


def load_judge_model():
    """Load Llama-3-8B-Instruct in 4-bit quantization as the AI judge."""
    print("📥 Loading Llama-3-8B-Instruct as the Final AI Judge...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    judge_tokenizer.pad_token = judge_tokenizer.eos_token

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
    )

    print("✅ Llama-3 Judge loaded successfully!")
    return judge_tokenizer, judge_model