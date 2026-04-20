import torch
import time
import warnings

# 禁用所有FutureWarning警告，确保bitsandbytes的警告被禁用
warnings.filterwarnings("ignore", category=FutureWarning)

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  未检测到GPU，将使用CPU训练")

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig

# 自定义训练回调类，用于输出详细的训练进度
class TrainingProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"🚀 训练开始！总步数: {state.max_steps}")
        print("=" * 80)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n📈 第 {state.epoch} 轮开始")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # 计算训练速度
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed
            
            # 计算剩余时间
            remaining_steps = state.max_steps - state.global_step
            remaining_time = remaining_steps / steps_per_sec
            
            # 格式化剩余时间
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)
            
            # 获取最新的损失值
            loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
            
            # 计算进度百分比
            progress = (state.global_step / state.max_steps) * 100
            
            # 打印进度信息
            print(f"[{state.global_step}/{state.max_steps}] 进度: {progress:.2f}% | 损失: {loss:.4f} | 速度: {steps_per_sec:.2f} step/s | 剩余时间: {hours}h {minutes}m {seconds}s")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        hours = int(epoch_time // 3600)
        minutes = int((epoch_time % 3600) // 60)
        seconds = int(epoch_time % 60)
        print(f"📊 第 {state.epoch} 轮结束，耗时: {hours}h {minutes}m {seconds}s")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\n🏁 训练完成！总耗时: {hours}h {minutes}m {seconds}s")
        print("=" * 80)


# 模型配置
# 本地模型路径，我这里是Qwen3.5-0.8B版本
MODEL_PATH = r"D:\Hugging_Face\models\Qwen3.5-0.8B"
# 训练数据集文件路径
DATA_PATH = r"data.json"
# 训练后的LoRA权重保存路径
LORA_OUTPUT_PATH = r"./qwen3.5-0.8b-lora-data"


# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 启用4bit量化
    bnb_4bit_use_double_quant=True,        # 使用双重量化
    bnb_4bit_quant_type="nf4",             # 指定 4 位量化的类型，nf为归一化浮点量化
    bnb_4bit_compute_dtype=torch.float16   # 模型计算时使用的数据类型
)

# LoRA超参数
lora_config = LoraConfig(
    r=16,                     # LoRA秩，8-32之间，小模型16足够
    lora_alpha=32,            # 缩放参数，通常是r的2倍
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen核心注意力层
    lora_dropout=0.05,                # LoRA dropout，防止过拟合
    bias="none",                    # 不使用偏置项
    task_type="CAUSAL_LM"           # 任务类型，因果语言模型
)

# 训练参数
training_args = SFTConfig(
    output_dir=LORA_OUTPUT_PATH,
    per_device_train_batch_size=1,   # 可以根据显存调整
    gradient_accumulation_steps=8,   # 累计梯度，等效增大batch_size
    learning_rate=2e-4,              # 小模型推荐学习率1e-4~3e-4
    num_train_epochs=3,              # 100条数据3轮，避免过拟合
    logging_steps=10,                # 每10步日志一次
    save_strategy="epoch",           # 每轮保存一次
    fp16=False,                       # 开启混合精度训练
    report_to="none",                # 不上报日志
    optim="adamw_torch",             # 使用标准AdamW优化器，避免8bit优化器的兼容性问题
    max_length=512,                  # 单条数据长度，短对话512足够
    dataset_text_field="text"        # 数据集文本字段名
)

def main():
    print("="*50)
    print("开始加载模型和Tokenizer...")
    print("="*50)

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Qwen必须设置pad_token

    # 加载模型（4bit量化）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        dtype=torch.float16,  # 使用dtype替代deprecated的torch_dtype
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 打印模型设备信息
    print(f"✅ 模型加载完成，当前设备: {next(model.parameters()).device}")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # 打印可训练参数量
    model.print_trainable_parameters()

    print("="*50)
    print("开始加载和处理数据集...")
    print("="*50)

    # 加载数据集
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 格式化对话数据（应用Qwen的Chat Template）
    def format_chat(examples):
        texts = []
        for messages in examples["messages"]:
            # 直接用tokenizer的apply_chat_template处理成模型需要的格式
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(formatted)
        return {"text": texts}

    formatted_dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

    print("="*50)
    print("开始LoRA微调训练...")
    print("="*50)

    # 初始化SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[TrainingProgressCallback()]
    )

    # 开始训练
    trainer.train()

    # 保存最终LoRA权重
    trainer.model.save_pretrained(LORA_OUTPUT_PATH)
    tokenizer.save_pretrained(LORA_OUTPUT_PATH)

    print("="*50)
    print(f"🎉 蟹堡王服务员LoRA微调完成！权重已保存到：{LORA_OUTPUT_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()