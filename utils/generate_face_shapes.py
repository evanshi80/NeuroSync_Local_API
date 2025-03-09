# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# generate_face_shapes.py

import time
import numpy as np

from utils.audio.extraction.extract_features import extract_audio_features
from utils.audio.processing.audio_processing import process_audio_features

def generate_facial_data_from_bytes(audio_bytes, model, device, config):
    t0 = time.time()
    audio_features, y = extract_audio_features(audio_bytes, from_bytes=True)
    t1 = time.time()
    print(f"extract_audio_features - > Took {t1-t0} sec")
    if audio_features is None or y is None:
        return [], np.array([])
  
    final_decoded_outputs = process_audio_features(audio_features, model, device, config)
    t2 = time.time()
    print(f"extract_audio_features - > Took {t2-t1} sec")
    return final_decoded_outputs

