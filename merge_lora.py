from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL_PATH = r"D:\Hugging_Face\models\Qwen3.5-0.8B"  # 原始模型路径
LORA_PATH = r"./qwen3.5-0.8b-lora-data"                   # LoRA权重路径
SAVE_MERGED_PATH = r"./千问蟹堡王"                          # 合并后新模型保存路径

# 加载基础模型
print("加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# 加载LoRA权重
print("加载LoRA权重...")
model = PeftModel.from_pretrained(model, LORA_PATH)

# 合并LoRA+基础模型
print("开始合并权重...")
model = model.merge_and_unload()

# 保存合并后的完整模型
print("保存合并后的模型...")
model.save_pretrained(SAVE_MERGED_PATH)
tokenizer.save_pretrained(SAVE_MERGED_PATH)

print(f"✅ 合并完成！新模型保存在：{SAVE_MERGED_PATH}")