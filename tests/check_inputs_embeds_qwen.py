# 验证模型 forward、数据 pipeline、trainer prompt 注入接口是否正常

# tests/check_inputs_embeds_qwen.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = r"F:/github/Qwen3-1.7B"  # <- 请按实际路径修改

# optional: avoid HF version check warnings if needed
os.environ["DISABLE_VERSION_CHECK"] = "1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Loading model (AutoModelForCausalLM)...")
# use device_map="auto" if you have multiple GPUs or limited memory (requires accelerate)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded to {device}")

sample = "今天天气很好。"
print("Tokenizing sample:", sample)
inputs = tokenizer(sample, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

with torch.no_grad():
    emb = model.get_input_embeddings()(input_ids)  # (1, T, D)
    print("input_emb shape:", emb.shape, "dtype:", emb.dtype)

    # forward with inputs_embeds (many HF causalLMs support this)
    try:
        out = model(inputs_embeds=emb, attention_mask=attention_mask)
        print("Forward OK. output keys:", list(out.keys()))
    except TypeError as e:
        print("Forward with inputs_embeds failed:", e)

    # test generate (simple)
    print("Running generate (1 token)...")
    gen = model.generate(input_ids, max_new_tokens=20)
    print("Generated ids:", gen.shape)
    print("Decoded:", tokenizer.batch_decode(gen, skip_special_tokens=True))
