from fastapi import FastAPI, Body
import io
import time
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
    print("Librosa 预热完成，加载耗时:", time.time() - t0)

def process_data(data: bytes):
    """
    使用音频数据生成 blendshapes 数据。
    """
    generated_facial_data = generate_facial_data_from_bytes(data, blendshape_model, device, config)
    if isinstance(generated_facial_data, np.ndarray):
        generated_facial_data_list = generated_facial_data.tolist()
    else:
        generated_facial_data_list = generated_facial_data
    return {'blendshapes': generated_facial_data_list}

@app.post("/audio_to_blendshapes")
async def audio_to_blendshapes(audio: bytes = Body(...)):
    """
    RESTful 接口：接收请求体中的音频 bytes 数据，
    返回处理后的 blendshapes 数据。
    """
    try:
        result = process_data(audio)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    warmup_librosa()
    uvicorn.run(app, host="127.0.0.1", port=5000)
