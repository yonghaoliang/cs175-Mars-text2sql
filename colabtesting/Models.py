import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings('ignore')

model      = None
tokenizer  = None
judge_model     = None
judge_tokenizer = None

def load_sqlcoder(model_name="defog/sqlcoder-7b-2"):
    global model, tokenizer
    print(f"🚀 Loading {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("✅ SQLCoder loaded successfully!")
    return model, tokenizer

def load_judge(model_id="NousResearch/Meta-Llama-3-8B-Instruct"):
    global judge_model, judge_tokenizer
    print(f"📥 Loading {model_id} as AI Judge...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(model_id)
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )
    print("✅ Llama-3 Judge loaded successfully!")
    return judge_model, judge_tokenizer

def run_inference(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):]