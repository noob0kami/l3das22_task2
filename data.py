import os
import numpy as np
import librosa
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import gc
import time

def preprocess_and_load_l3das(
    data_dir, label_dir,
    segment_len=1.0, overlap=0.5, sr=16000, n_fft=512, hop_length=128, start_index=0, max_files=500
):
    x = []
    y = []
    segment_id = 0
    file_count = 0
    all_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav") and ("_A" in f or "_B" in f)])
    filenames_to_process = all_filenames[start_index:start_index + max_files]

    for filename in filenames_to_process:
        clean_name = filename.replace("_A.wav", ".wav").replace("_B.wav", ".wav")
        label_path = os.path.join(label_dir, clean_name)
        data_path = os.path.join(data_dir, filename)

        if not os.path.exists(label_path):
            print(f"[警告] 找不到乾淨音檔: {label_path}，跳過 {filename}")
            continue

        mix_audio, _ = librosa.load(data_path, sr=sr, mono=False)
        clean_audio, _ = librosa.load(label_path, sr=sr, mono=True)

        if mix_audio.shape[0] != 4:
            print(f"[錯誤] {filename} 通道不足")
            continue

        seg_samples = int(segment_len * sr)
        hop_samples = int(segment_len * (1 - overlap) * sr)
        total_len = mix_audio.shape[1]
        num_segments_in_file = int(np.ceil((total_len - seg_samples) / hop_samples)) + 1

        for i in range(num_segments_in_file):
            start = i * hop_samples
            end = start + seg_samples

            mix_seg = mix_audio[:, start:end]
            clean_seg = clean_audio[start:end]

            if mix_seg.shape[1] < seg_samples:
                pad_len = seg_samples - mix_seg.shape[1]
                mix_seg = np.pad(mix_seg, ((0, 0), (0, pad_len)))
                clean_seg = np.pad(clean_seg, (0, pad_len))

            # STFT
            mix_stft = [librosa.stft(mix_seg[c], n_fft=n_fft, hop_length=hop_length) for c in range(4)]
            mix_stft = np.stack([np.stack([np.real(s), np.imag(s)], axis=-1) for s in mix_stft], axis=0)

            clean_stft = librosa.stft(clean_seg, n_fft=n_fft, hop_length=hop_length)
            clean_stft = np.stack([np.real(clean_stft), np.imag(clean_stft)], axis=-1)

            # 將資料添加到x和y
            x.append(mix_stft)
            y.append(clean_stft)
            segment_id += 1

        file_count += 1

    # 轉換為 NumPy 陣列
    x = np.array(x)
    y = np.array(y)

    print(f"預處理完成，共處理 {file_count} 個音訊檔案，產生 {segment_id} 個 1 秒片段")

    return x, y

def process_audio_tensors(x, y, target_f=256, target_t=128):
    """
    將音訊張量 x 和 y 處理到目標頻率 (256) 和時間 (128) 維度。
    輸入 x 形狀預期為 (N, 4, 258, 126, 2)。
    輸入 y 形狀預期為 (N, 258, 126, 2)。
    """
    x_processed = x[:, :, :target_f, :, :]
    padding_t = target_t - x.shape[3]
    x_processed = torch.nn.functional.pad(torch.from_numpy(x_processed).float(), (0, 0, 0, padding_t, 0, 0, 0, 0))

    y_processed = y[:, :target_f, :, :]
    y_processed = torch.nn.functional.pad(torch.from_numpy(y_processed).float(), (0, 0, 0, padding_t, 0, 0)) # Adjusted padding for y

    return x_processed, y_processed

def append_to_hdf5(output_h5_path, x_data, y_data, chunk_size=64):
    num_samples = x_data.shape[0]
    num_full_batches = num_samples // chunk_size

    if num_full_batches > 0:
        x_to_save = x_data[:num_full_batches * chunk_size].numpy()
        y_to_save = y_data[:num_full_batches * chunk_size].numpy()

        with h5py.File(output_h5_path, 'a') as hf:
            if 'x' in hf:
                old_x_len = hf['x'].shape[0]
                hf['x'].resize((old_x_len + x_to_save.shape[0],) + x_to_save.shape[1:])
                hf['x'][old_x_len:] = x_to_save
            else:
                chunk_shape_x = (chunk_size,) + x_to_save.shape[1:]
                hf.create_dataset('x', data=x_to_save, maxshape=(None,) + x_to_save.shape[1:], chunks=chunk_shape_x)

            if 'y' in hf:
                old_y_len = hf['y'].shape[0]
                hf['y'].resize((old_y_len + y_to_save.shape[0],) + y_to_save.shape[1:])
                hf['y'][old_y_len:] = y_to_save
            else:
                chunk_shape_y = (chunk_size,) + y_to_save.shape[1:]
                hf.create_dataset('y', data=y_to_save, maxshape=(None,) + y_to_save.shape[1:], chunks=chunk_shape_y)

        print(f"成功將 {num_full_batches * chunk_size} 個樣本追加到 {output_h5_path} (chunk size: {chunk_size})")
    else:
        print("處理後的數據不足一個批次，未追加到 HDF5 檔案。")

# --- 主要執行部分 ---
data_dir="L3DAS22_Task1_dev/data"
label_dir="L3DAS22_Task1_dev/labels"
output_h5_path = 'total.h5'
batch_size = 64
files_per_batch = 500
num_total_files = len([f for f in os.listdir(data_dir) if f.endswith(".wav") and ("_A" in f or "_B" in f)])

for i in range(0, num_total_files, files_per_batch):
    print(f"\n--- 處理檔案 {i} 到 {min(i + files_per_batch, num_total_files)} ---")
    x, y = preprocess_and_load_l3das(data_dir, label_dir, segment_len=1.0, overlap=0.5, sr=16000, max_files=files_per_batch, start_index=i)

    if x.shape[0] > 0:
        x_processed, y_processed = process_audio_tensors(x, y) # Removed unsqueeze here
        append_to_hdf5(output_h5_path, x_processed, y_processed, batch_size)

    # 刪除參數以釋放記憶體
    del x
    del y
    del x_processed
    del y_processed
    gc.collect()
    print("已刪除處理後的參數並進行垃圾回收。")

print(f"\n所有符合條件的音訊檔案已處理完畢，並儲存至 {output_h5_path}。")

# --- 從 total.h5 讀取的 PyTorch Dataset (用於驗證) ---
class L3DAS_Total_HDF5_Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, 'r') as hf:
            if 'y' in hf:
                self.len = len(hf['y'])
            else:
                self.len = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            x = hf['x'][idx]
            y = hf['y'][idx]
            return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

    def __del__(self):
        pass

# --- 創建 Dataset 和 DataLoader 進行簡單測試 ---
if os.path.exists(output_h5_path):
    total_h5_dataset = L3DAS_Total_HDF5_Dataset(output_h5_path)
    if len(total_h5_dataset) > 0:
        total_dataloader = DataLoader(total_h5_dataset, batch_size=64, shuffle=False, num_workers=0)
        if __name__ == '__main__':
            print("\n--- 測試從 total.h5 載入數據 ---")
            start_time = time.time()
            for i, (batch_x, batch_y) in enumerate(total_dataloader):
                print(f"測試批次 {i+1} x 的形狀:", batch_x.shape)
                print(f"測試批次 {i+1} y 的形狀:", batch_y.shape)
                break
            end_time = time.time()
            print(f"載入第一個測試批次所需時間: {end_time - start_time:.4f} 秒")
    else:
        print("total.h5 中沒有數據。")
else:
    print("total.h5 不存在，跳過 Dataset 和 DataLoader 的測試。")