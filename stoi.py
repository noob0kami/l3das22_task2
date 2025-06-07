import librosa
from pystoi.stoi import stoi
import numpy as np
import os 

def calculate_stoi_for_files(clean_audio_path, enhanced_audio_path, sr):
    if not os.path.exists(clean_audio_path):
        print(f"[錯誤] 找不到乾淨音訊檔案: {clean_audio_path}")
        return None
    if not os.path.exists(enhanced_audio_path):
        print(f"[錯誤] 找不到增強音訊檔案: {enhanced_audio_path}")
        return None

    try:
        clean_audio, sr_clean = librosa.load(clean_audio_path, sr=sr, mono=True)
        enhanced_audio, sr_enhanced = librosa.load(enhanced_audio_path, sr=sr, mono=True)

        if sr_clean != sr or sr_enhanced != sr:
            print(f"[警告] 載入音訊時採樣率不符。Clean: {sr_clean} vs Enhanced: {sr_enhanced} vs Target: {sr}")
            pass 

        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio_cropped = clean_audio[:min_len]
        enhanced_audio_cropped = enhanced_audio[:min_len]

        stoi_score = stoi(clean_audio_cropped, enhanced_audio_cropped, sr)

        return stoi_score

    except Exception as e:
        print(f"[錯誤] 計算 STOI 時發生例外: {e}")
        return None

if __name__ == '__main__':
    SR = 16000 

    clean_audio_for_stoi_path = 'L3DAS22_Task1_dev/labels/84-121123-0001.wav'
    enhanced_output_path = 'enhanced_output_xtf_final.wav'

    stoi_value = calculate_stoi_for_files(clean_audio_for_stoi_path, enhanced_output_path, SR)

    if stoi_value is not None:
        print(f"\nSTOI 分數: {stoi_value}")