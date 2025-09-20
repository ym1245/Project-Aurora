import json
import os
import subprocess
import shutil
from llama_cpp import Llama
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import torch

# 1. 환경 설정
MODEL_PATHH = r"C:/Users/breadly1245/.cache/huggingface/hub/models--allganize--Llama-3-Alpha-Ko-8B-Instruct/snapshots/d294a56f7b3d1128178a75148f14a00964e961b1"
DATASET_PATH = r"/\LoRA finetuner\dataset.json"
OUTPUT_DIR = r"/\model"
LORA_ADAPTER_PATH = r"//LoRA finetuner/lora_adapter"
GGUF_LORA_PATH = r"//LoRA finetuner/lora_adapter.gguf/Lora_Adapter-F16-LoRA.gguf"
SEVEN_ZIP_EXEC = r"C:\Program Files\7-Zip\7z.exe"

# 2. JSON 데이터셋 로드 및 전처리
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    processed_data = {
        "input": [item["input"] for item in data],
        "output": [item["output"] for item in data]
    }
    return Dataset.from_dict(processed_data)

# 3. 토크나이저 로드
model_id = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. 데이터셋 전처리 함수
def preprocess_function(examples):
    inputs = [f"{item}" for item in examples["input"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 5. 데이터셋 로드 및 토큰화
dataset = load_json_dataset(DATASET_PATH)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 6. LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 7. Hugging Face 모델 로드 및 LoRA 적용
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 8. 훈련 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# 9. Trainer 초기화 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# 10. LoRA 어댑터 저장
model.save_pretrained(LORA_ADAPTER_PATH)
print(f"LoRA 어댑터가 저장되었습니다: {LORA_ADAPTER_PATH}")

#python "C:/Users/breadly1245/Desktop/Project Aurora/LoRA finetuner/llama.cpp/convert_lora_to_gguf.py" --base "C:/Users/breadly1245/.cache/huggingface/hub/models--allganize--Llama-3-Alpha-Ko-8B-Instruct/snapshots/d294a56f7b3d1128178a75148f14a00964e961b1" --outfile "C:/Users/breadly1245/Desktop/Project Aurora/LoRA finetuner/lora_adapter.gguf" "C:/Users/breadly1245/Desktop/Project Aurora/LoRA finetuner/lora_adapter"

# 11. LoRA 어댑터를 GGUF 형식으로 변환
convert_command = [
    "python",
    "C:/Users/breadly1245/Desktop/Project Aurora/LoRA finetuner/llama.cpp/convert_lora_to_gguf.py",
    "--base", MODEL_PATHH,
    "--outfile", GGUF_LORA_PATH,
    LORA_ADAPTER_PATH
]

subprocess.run(convert_command, check=True)
print(f"LoRA 어댑터가 GGUF 형식으로 변환되었습니다: {GGUF_LORA_PATH}")

# 12. 체크포인트 폴더들을 7z 형식으로 최대 압축률로 압축
checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint-")]
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(OUTPUT_DIR, checkpoint)
    archive_name = os.path.join(OUTPUT_DIR, f"{checkpoint}.7z")
    compress_command = [SEVEN_ZIP_EXEC, "a", "-mx9", archive_name, checkpoint_path]
    try:
        subprocess.run(compress_command, check=True)
        print(f"체크포인트 {checkpoint}가 {archive_name}으로 압축되었습니다.")
        shutil.rmtree(checkpoint_path)
        print(f"원본 체크포인트 폴더 {checkpoint}가 삭제되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"체크포인트 {checkpoint} 압축 실패: {e}")
        print(f"7-Zip 실행 파일({SEVEN_ZIP_EXEC})이 올바른지 확인하세요.")