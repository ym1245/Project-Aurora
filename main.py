import discord
from discord.ext import commands
import load_whisper
import asyncio
import os
import wave
import psutil
from io import BytesIO
from pynvml import *
from llama_cpp import Llama

gctx=None
llm_result=None
model_path="model/qwen3_8b_192k Q8_0.gguf"
lora_path="LoRA finetuner/lora_adapter.gguf/Lora_Adapter-F16-LoRA.gguf"
conversation_history=[]

stt_model=load_whisper.load_model("turbo")

def load_model():
    global model_path
    global lora_path
    mem=psutil.virtual_memory()
    d_free=mem.free/1024/1024/1024  #DRAM 여유용량
    d_free-=6  #DRAM 여유를 위한 6GB
    nvmlInit()
    handle=nvmlDeviceGetHandleByIndex(0)
    info=nvmlDeviceGetMemoryInfo(handle)
    v_free=info.free/1024/1024**2  #VRAM 여유용량
    print(f"DMemory free: {d_free} GB")
    print(f"VMemory free: {v_free} GB")
    nvmlShutdown()
    llm=Llama(model_path=model_path,n_gpu_layers=0,n_ctx=1)
    for k in llm.metadata:
        if "block" in k.lower() in k.lower():
            model_layer_count=int(llm.metadata[k])
    del llm
    model_total_size=10
    layer_size=model_total_size/model_layer_count
    n_gpu_layers=int(v_free//layer_size)
    if v_free+d_free>model_total_size:
        if n_gpu_layers>=model_layer_count:
            n_gpu_layers=model_layer_count
        print(f"추정된 n_gpu_layers: {n_gpu_layers} (총 {model_layer_count} 중)")
        return Llama(
            model_path=model_path,
            #lora_path=lora_path,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            flash_attn=True,
            chat_format="chatml",
        )
    raise MemoryError

llm_model=load_model()

async def llm_input(ctx,user_id,text):
    global llm_result
    global llm_model
    global conversation_history
    system_prompt="""
/think
한국어를 사용하세요.
당신의 이름은 오로라입니다.
당신은 AI 캐릭터입니다. 어시스턴트가 아니라 캐릭터입니다.
당신의 대답은 매우 짧아야 합니다. 마치 사람 간의 대화처럼.
당신이 받은 메시지 앞의 숫자는 디스코드 사용자 ID입니다. 이 ID를 사용하여 메시지를 보낸 사람을 구분하세요.
955403977636331520은 당신의 개발자 breadly입니다.
841319977760194600은 mintflower입니다.
당신은 남성이 아닙니다.
이전 대화는 간단한 참고 자료일 뿐이며, 지금 질문에 답변하는 데 더 집중해야 합니다.
"""
    llm_result=llm_model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_id}가 보낸 메시지:\n\n\n\n{text}"},
        ]+conversation_history,
        max_tokens=1024,
        temperature=0.8
    )

    response=llm_result['choices'][0]['message']['content']
    conversation_history.append({
        "role":"assistant",
        "content":response
    })

    if len(conversation_history)>40:
        conversation_history=conversation_history[-40:]

    await ctx.send(response)

bot=commands.Bot(
    instents=discord.Intents.default() | discord.Intents.voice_states | discord.Intents.message_content
)

class PlusSink(discord.sinks.WaveSink):
    def __init__(self):
        super().__init__()
        self.audio={}
        self.last_speak_time={}
        self.tasks={}
        self.loop=asyncio.get_event_loop()

    def create_wav_bytesio(self,audio_byte):
        buffer=BytesIO()
        with wave.open(buffer,'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_byte)
        buffer.seek(0)
        return buffer

    def write(self, data, user):
        user_id=user
        if user_id not in self.audio:
            self.audio[user_id]=[]
            self.last_speak_time[user_id]=self.loop.time()
            self.tasks[user_id]=self.loop.create_task(self.monitor_silence(user_id))
        self.audio[user_id].append(data)
        self.last_speak_time[user_id] = self.loop.time()

    async def monitor_silence(self, user_id):
        while True:
            await asyncio.sleep(0.5)
            if self.loop.time()-self.last_speak_time.get(user_id, 0) > 2:
                await self.process_audio(user_id)
                if user_id in self.tasks:
                    del self.tasks[user_id]
                break

    async def process_audio(self, user_id):
        audio_bytes=b''.join(self.audio[user_id])
        wav_buffer=self.create_wav_bytesio(audio_bytes)
        stt_result=stt_model.transcribe(wav_buffer)
        text=stt_result["text"].strip()
        if text:
            await gctx.send(f"{user_id}: {text}")
            await llm_input(gctx,user_id,text)
        del self.audio[user_id]
        del self.last_speak_time[user_id]

@bot.slash_command(name="join")
async def join(ctx):
    global gctx
    gctx=ctx
    channel=ctx.author.voice.channel
    voice_client=await channel.connect()
    sink=PlusSink()
    voice_client.start_recording(sink,callback,ctx)

async def callback(sink, ctx):
    await ctx.voice_client.disconnect()

@bot.slash_command(name="leave")
async def leave(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop_recording()


script_dir=os.path.dirname(os.path.abspath(__file__))
token_file_path=os.path.join(script_dir, 'discord.token')
with open(token_file_path, 'r') as file:
    token=file.read().strip()

bot.run(token)