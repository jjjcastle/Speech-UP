from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 🔧 설정값
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./llama3_lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ tokenizer 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ✅ 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ✅ base model 로딩 (4bit)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config
)

# ✅ LoRA adapter 로딩
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ✅ 외부에서 사용할 함수
def correct_sentence(raw_sentence: str) -> str:
    prompt = f"Correct the following sentence:\n{raw_sentence.strip()}\nCorrected sentence:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()

    if "Corrected sentence:" in generated:
        result = generated.split("Corrected sentence:")[1].strip().split("\n")[0]
    else:
        result = generated.strip().split("\n")[0]
    return result
