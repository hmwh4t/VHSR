'''
Dataset CleanSTT.csv:
- Lấy gộp từ dữ liệu ViHSD và VOZ-HSD
- Sửa lại để theo cấu trúc Speech-To-Text (Xóa Emoji, Xóa Email, Xóa Link, Buộc Chữ Thường, Hoàn Thành Các Từ Viết Tắt)
- Gồm cột text và label, label chỉ có giá trị ['clean', 'hate']
'''

'''
Model_Train.py:
- Huấn luyện Model DeBERTa dựa trên ./CleanSTT.csv
- Lưu Model này vào thư mục ./output_model
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import torch

# ---------------------------------------------------------
# Cài Đặt
# ---------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"  # Model DeBERTa
FILE_PATH = "CleanSTT.csv"                 # Dataset
MAX_LENGTH = 128                           # Độ dài tối đa của câu
BATCH_SIZE = 16                            # Giảm xuống 8 nếu bị tràn VRAM
EPOCHS = 3                                 # Số vòng lặp huấn luyện
LEARNING_RATE = 2e-5                       # Tốc độ học chuẩn cho DeBERTa

print(f"Đang đọc dữ liệu từ {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

# clean -> 0, hate -> 1
label_map = {'clean': 0, 'hate': 1}
df['label'] = df['label'].map(label_map)

# Train (80%), Validation (10%), Test (10%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Định dạng HuggingFace Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(val_df, preserve_index=False),
    'test': Dataset.from_pandas(test_df, preserve_index=False)
})
print(f"Kích thước tập Train: {len(dataset['train'])}")
print(f"Kích thước tập Val: {len(dataset['validation'])}")

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
# Sử dụng F1-score và Accuracy. F1 rất quan trọng với dữ liệu mất cân bằng.
f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    acc = acc_metric.compute(predictions=predictions, references=labels)
    
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
print("Đang tải Model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results_hsd_deberta",
    eval_strategy="epoch",            # Đánh giá sau mỗi epoch
    save_strategy="epoch",            # Lưu model sau mỗi epoch
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,      # Tự động load model tốt nhất sau khi train xong
    metric_for_best_model="f1_macro", # Chọn model dựa trên F1-score cao nhất
    save_total_limit=2,               # Chỉ giữ lại 2 checkpoint gần nhất
    fp16=torch.cuda.is_available(),   # Bật Mixed Precision nếu có GPU
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Đang huấn luyện...")
trainer.train()

# ---------------------------------------------------------
# Đánh Giá Model
# ---------------------------------------------------------
print("Đang đánh giá...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print("Kết quả trên tập Test:", test_results)

save_path = "./output_model"
print(f"Lưu model tại: {save_path}")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("Hoàn tất.")