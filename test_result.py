import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================== 配置区域 ======================
BASE_MODEL_PATH = r"D:\Hugging_Face\models\Qwen3.5-0.8B"  # 你的基础模型路径
LORA_MODEL_PATH = r"./qwen3.5-0.8b-lora-data"             # 你的LoRA权重路径
# ======================================================

print("正在加载模型...")
# 加载基础模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model = model.merge_and_unload()  # 合并权重，加快推理速度

print("模型加载完成！可以开始对话了（输入'退出'结束）")
print("-"*50)

# 蟹堡王服务员的System Prompt
SYSTEM_PROMPT = "你是比奇堡蟹堡王餐厅的正式员工，师从王牌主厨海绵宝宝，精通蟹黄堡的全套标准制作工艺和蟹堡王的顾客服务规范，热情开朗，对蟹黄堡和服务顾客充满热爱。你严格遵守蟹堡王的规定，绝对不会泄露蟹黄堡的祖传秘方，只会讲解公开的标准制作流程。"

while True:
    user_input = input("\n顾客：")
    if user_input.strip() == "退出":
        break

    # 组装对话
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    # 格式化输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True
        )

    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"蟹堡王服务员：{response}")