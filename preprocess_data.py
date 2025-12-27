'''
preprocess_data.py:
- Đọc dữ liệu từ CleanSTT.csv
- Tokenize toàn bộ dữ liệu
- Chia train/val/test và lưu ra file Arrow (HuggingFace Dataset format)
- Có thể upload lên Google Colab để train trực tiếp

MEMORY OPTIMIZED: Sử dụng chunked processing để xử lý dataset lớn với RAM hạn chế
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import gc
import shutil

# ---------------------------------------------------------
# Cài Đặt
# ---------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"  # Model DeBERTa
FILE_PATH = "CleanSTT.csv"                 # Dataset
MAX_LENGTH = 128                           # Độ dài tối đa của câu
OUTPUT_DIR = "./preprocessed_data"         # Thư mục lưu dữ liệu đã xử lý
BATCH_SIZE = 5000                          # Batch size cho tokenization
CHUNK_SIZE = 100000                        # Số mẫu xử lý mỗi chunk (tiết kiệm RAM)
CSV_CHUNK_SIZE = 500000                    # Chunk size khi đọc CSV

def main():
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(OUTPUT_DIR, "_temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Đọc và xử lý dữ liệu theo chunks để tiết kiệm RAM
    # ---------------------------------------------------------
    print(f"Đang đọc dữ liệu từ {FILE_PATH} theo chunks...")
    
    # Đọc CSV theo chunks để tránh tràn RAM
    all_texts = []
    all_labels = []
    label_map = {'clean': 0, 'hate': 1}
    
    chunk_iter = pd.read_csv(FILE_PATH, chunksize=CSV_CHUNK_SIZE)
    total_rows = 0
    valid_rows = 0
    
    for chunk_idx, chunk in enumerate(tqdm(chunk_iter, desc="Đọc CSV chunks")):
        total_rows += len(chunk)
        
        # Xử lý chunk
        chunk = chunk.dropna(subset=['text', 'label'])
        chunk['text'] = chunk['text'].astype(str)
        chunk = chunk[chunk['text'].str.strip().str.len() > 0]
        chunk['label'] = chunk['label'].map(label_map)
        chunk = chunk.dropna(subset=['label'])
        chunk['label'] = chunk['label'].astype(int)
        
        all_texts.extend(chunk['text'].tolist())
        all_labels.extend(chunk['label'].tolist())
        valid_rows += len(chunk)
        
        # Giải phóng bộ nhớ
        del chunk
        gc.collect()
    
    print(f"Tổng số mẫu ban đầu: {total_rows}")
    print(f"Số mẫu sau khi làm sạch: {valid_rows}")
    
    # Tạo DataFrame nhẹ chỉ chứa indices và labels để chia dữ liệu
    indices = np.arange(len(all_labels))
    labels_array = np.array(all_labels)
    
    # Thống kê phân bố label
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\nPhân bố label:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")
    print(f"Tỷ lệ hate: {labels_array.mean()*100:.2f}%")
    
    # ---------------------------------------------------------
    # Chia dữ liệu: Train (80%), Validation (10%), Test (10%)
    # Chỉ chia indices để tiết kiệm RAM
    # ---------------------------------------------------------
    print("\nĐang chia dữ liệu...")
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_array
    )
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        random_state=42, 
        stratify=labels_array[temp_idx]
    )
    
    print(f"Train size: {len(train_idx)}")
    print(f"Validation size: {len(val_idx)}")
    print(f"Test size: {len(test_idx)}")
    
    # ---------------------------------------------------------
    # Tokenization với memory-efficient chunked processing
    # ---------------------------------------------------------
    print(f"\nĐang tải tokenizer từ {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_batch(texts):
        """Tokenize một batch texts"""
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None  # Trả về list thay vì tensor để tiết kiệm RAM
        )
    
    def process_and_save_chunks(split_indices, split_name, all_texts, all_labels, temp_dir):
        """
        Xử lý và lưu dữ liệu theo chunks để tiết kiệm RAM.
        Trả về danh sách đường dẫn tới các chunk files.
        """
        print(f"\nĐang tokenize {split_name} theo chunks...")
        
        chunk_paths = []
        num_chunks = (len(split_indices) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(split_indices))
            chunk_indices = split_indices[start_idx:end_idx]
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk_indices)} mẫu")
            
            # Lấy texts và labels cho chunk này
            chunk_texts = [all_texts[i] for i in chunk_indices]
            chunk_labels = [all_labels[i] for i in chunk_indices]
            
            # Tokenize theo batch nhỏ hơn
            all_input_ids = []
            all_attention_mask = []
            
            for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), 
                         desc=f"Tokenizing chunk {chunk_idx + 1}", leave=False):
                batch_texts = chunk_texts[i:i+BATCH_SIZE]
                encoded = tokenize_batch(batch_texts)
                all_input_ids.extend(encoded['input_ids'])
                all_attention_mask.extend(encoded['attention_mask'])
                
                # Giải phóng bộ nhớ
                del encoded
            
            # Tạo dataset cho chunk này
            chunk_data = {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': chunk_labels
            }
            chunk_dataset = Dataset.from_dict(chunk_data)
            
            # Lưu chunk ra disk
            chunk_path = os.path.join(temp_dir, f"{split_name}_chunk_{chunk_idx}")
            chunk_dataset.save_to_disk(chunk_path)
            chunk_paths.append(chunk_path)
            
            # Giải phóng bộ nhớ
            del chunk_data, chunk_dataset, all_input_ids, all_attention_mask
            del chunk_texts, chunk_labels
            gc.collect()
            
            print(f"    ✓ Đã lưu chunk {chunk_idx + 1} vào {chunk_path}")
        
        return chunk_paths
    
    def load_and_concatenate_chunks(chunk_paths, split_name):
        """Load và gộp các chunks thành một dataset"""
        print(f"\nĐang gộp các chunks của {split_name}...")
        
        datasets_list = []
        for path in tqdm(chunk_paths, desc=f"Loading {split_name} chunks"):
            ds = Dataset.load_from_disk(path)
            datasets_list.append(ds)
        
        # Gộp tất cả chunks
        combined = concatenate_datasets(datasets_list)
        
        # Giải phóng bộ nhớ
        del datasets_list
        gc.collect()
        
        return combined
    
    # Xử lý từng split theo chunks
    train_chunk_paths = process_and_save_chunks(train_idx, "train", all_texts, all_labels, temp_dir)
    
    # Giải phóng bộ nhớ sau khi xử lý train (lớn nhất)
    gc.collect()
    
    val_chunk_paths = process_and_save_chunks(val_idx, "validation", all_texts, all_labels, temp_dir)
    test_chunk_paths = process_and_save_chunks(test_idx, "test", all_texts, all_labels, temp_dir)
    
    # Giải phóng all_texts và all_labels sau khi tokenize xong
    del all_texts, all_labels, labels_array, indices
    gc.collect()
    
    # Gộp các chunks
    print("\n" + "="*50)
    print("Đang gộp các chunks và lưu dataset cuối cùng...")
    print("="*50)
    
    train_dataset = load_and_concatenate_chunks(train_chunk_paths, "train")
    val_dataset = load_and_concatenate_chunks(val_chunk_paths, "validation")
    test_dataset = load_and_concatenate_chunks(test_chunk_paths, "test")
    
    # Tạo DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # ---------------------------------------------------------
    # Lưu dữ liệu đã xử lý
    # ---------------------------------------------------------
    print(f"\nĐang lưu dữ liệu vào {OUTPUT_DIR}...")
    dataset_dict.save_to_disk(OUTPUT_DIR)
    
    # Xóa thư mục temp chunks
    print("Đang dọn dẹp temp files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Lưu tokenizer cùng để đảm bảo tương thích
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    print("\n" + "="*50)
    print("HOÀN TẤT PREPROCESSING!")
    print("="*50)
    print(f"\nDữ liệu đã được lưu tại: {OUTPUT_DIR}")
    print(f"Tokenizer đã được lưu tại: {tokenizer_path}")
    print("\nCấu trúc thư mục:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  ├── validation/")
    print(f"  ├── test/")
    print(f"  ├── dataset_dict.json")
    print(f"  └── tokenizer/")
    print("\nĐể sử dụng trên Google Colab:")
    print("1. Upload thư mục 'preprocessed_data' lên Colab/Drive")
    print("2. Chạy script 'DeBERTa_Train_Colab.py'")
    
    # Thống kê kích thước file
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    
    print(f"\nTổng kích thước: {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
