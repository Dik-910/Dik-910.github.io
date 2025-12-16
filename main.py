import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# --- 1. 模型加载与配置 ---
# !!! 重要：请确保下面的路径与你电脑上的实际路径完全一致 !!!
MODEL_NAME = "D:\\model_cache\\qwen\\Qwen1___5-1___8B-Chat"

print("正在从本地加载模型...")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 加载模型，强制使用CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    trust_remote_code=True
)

# 创建生成pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cpu"
)

print("模型加载完成！")

# --- 2. FastAPI 应用定义 ---
app = FastAPI(title="AI学习助手 API", description="基于Qwen模型的轻量级问答服务")


class Query(BaseModel):
    question: str


class Response(BaseModel):
    answer: str
    time_taken: float


# --- 3. API 端点定义 ---
@app.post("/generate", response_model=Response)
async def generate_text(query: Query):
    start_time = time.time()

    # Qwen模型的提示词模板
    messages = [
        {"role": "system", "content": "你是一个知识渊博的AI学习助手，请用简单易懂的语言回答问题。"},
        {"role": "user", "content": query.question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sequences = generator(
        text,
        max_new_tokens=512,#限制生成文本长度为128个标记（token），防止输出过长。‌
        do_sample=True,#启用随机采样，使输出更具多样性
        top_k=30,#仅从概率最高的50个候选标记中选择，平衡多样性与可控性
        top_p=0.95,#累积概率阈值，确保选择的标记总概率超过95%，避免极端低概率选项
        temperature=0.2,#降低温度值，使模型更倾向于高概率标记，输出更确定性
    )

    answer = sequences[0]['generated_text'].split("assistant")[-1].strip()

    end_time = time.time()
    time_taken = end_time - start_time

    return Response(answer=answer, time_taken=time_taken)


@app.get("/")
async def root():
    return {"message": "欢迎使用AI学习助手API，请访问 /docs 查看API文档"}


# --- 4. 静态文件服务 ---
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")
# --- 5. 启动服务 ---
import uvicorn

uvicorn.run(app, host='0.0.0.0', port=8000)