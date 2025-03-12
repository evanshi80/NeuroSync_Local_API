import asyncio
import io
import threading
import queue
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import librosa
import numpy as np
import torch
import uvicorn
import soundfile as sf

from utils.config import config
from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)

def warmup_librosa(original_sr=24000, target_sr=88200):
    t0 = time.time()
    # 创建一个简单的正弦波音频（0.1秒的440Hz音频）
    duration = 0.1  # 0.1秒
    t = np.linspace(0, duration, int(original_sr * duration), endpoint=False)
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 将生成的音频写入内存中的WAV文件（PCM16格式）
    buf = io.BytesIO()
    sf.write(buf, dummy_audio, original_sr, format='WAV', subtype='PCM_16')
    buf.seek(0)

    # 使用 librosa.load 加载音频，并上采样到目标采样率
    y, sr = librosa.load(buf, sr=target_sr)
    print("Librosa 预热完成，加载耗时:", time.time()-t0)



def process_data(data: bytes) -> str:
    audio_bytes = data
    generated_facial_data = generate_facial_data_from_bytes(audio_bytes, blendshape_model, device, config)
    generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

    return {'blendshapes': generated_facial_data_list}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")

    # 为当前连接创建一个线程安全的队列，用于存放收到的消息
    message_queue = queue.Queue()
    # 获取当前运行的事件循环，用于线程中调度异步发送
    loop = asyncio.get_running_loop()

    def worker():
        """
        后台工作线程，从队列中取出消息依次处理，并将处理结果发送回客户端。
        """
        while True:
            # 阻塞等待队列中的消息
            data = message_queue.get()
            if data is None:
                # 收到 None 表示退出信号
                break
            try:
                result = process_data(data)
                # 通过 run_coroutine_threadsafe 将发送任务提交到事件循环中
                asyncio.run_coroutine_threadsafe(websocket.send_json(result), loop)
            except Exception as e:
                print("Worker thread encountered an error:", e)

    # 启动后台工作线程，daemon=True 表示主线程退出时自动结束
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    try:
        # 主循环用于接收客户端消息
        while True:
            data = await websocket.receive_bytes()
            print("Received:", data)
            # 将收到的消息放入队列中，按顺序排队等待后台线程处理
            message_queue.put(data)
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        # 退出前向队列发送退出信号，并等待工作线程结束
        message_queue.put(None)
        worker_thread.join()

if __name__ == '__main__':
    warmup_librosa()
    uvicorn.run(app, host="127.0.0.1", port=5000)
