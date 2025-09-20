from llama_cpp import Llama

#MODEL_PATH = r"C:\Users\breadly1245\Desktop\Project Aurora\model\qwen3_8b_192k Q8_0.gguf"
MODEL_PATH = r"/\model\Llama-3-Alpha-Ko-8B-Instruct.Q8_0.gguf"
GGUF_LORA_PATH = r"//LoRA finetuner/lora_adapter.gguf/Lora_Adapter-F16-LoRA.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    lora_path=GGUF_LORA_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    flash_attn=True,
)
prompt = "테나를 칭찬해봐"
output = llm(prompt, max_tokens=512)
print("추론 결과:")
print(output["choices"][0]["text"])