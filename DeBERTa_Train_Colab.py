'''
DeBERTa_Train_Colab_Optimized.py:
- Script huấn luyện tối ưu cho Google Colab với A100 (40GB VRAM)
- Đọc dữ liệu đã được tokenize từ preprocessed_data/
- FIX: Thêm Class Weights để xử lý dữ liệu mất cân bằng (94% vs 6%)
- FIX: Tối ưu Batch Size và Validation để tăng tốc độ training
'''

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import torch
import torch.nn as nn
import os
import json
import gc

# ---------------------------------------------------------
# Cài Đặt - Tối ưu cho A100 40GB VRAM
# ---------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"
PREPROCESSED_DATA_DIR = "./preprocessed_data"
OUTPUT_DIR = "./results_hsd_deberta"
FINAL_MODEL_DIR = "./output_model"

# Hyperparameters - OPTIMIZED FOR A100
MAX_LENGTH = 128

# BATCH SIZE OPTIMIZATION
# A100 có thể xử lý batch size lớn hơn nhiều. Tăng lên 64.
BATCH_SIZE = 64
# Giảm accumulation xuống để giữ tổng hiệu ứng batch size ~128 (64 * 2)
GRADIENT_ACCUMULATION_STEPS = 2 

# TRAINING LENGTH
EPOCHS = 1  # 1 Epoch là đủ cho 7 triệu mẫu

# CHANGE 1: REDUCE LEARNING RATE (Critical Fix)
LEARNING_RATE = 1e-5  # Reduced from 2e-5 to handle Class Weights safely

# CHANGE 2: DISABLE TORCH COMPILE (Stability Fix)
# Compile is causing issues with the custom weighted loss loop
USE_TORCH_COMPILE = False 

# CHANGE 3: REDUCE GRADIENT ACCUMULATION (Safety)
# Reducing accumulation reduces the chance of gradient explosion
GRADIENT_ACCUMULATION_STEPS = 1 
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

def main():
    # Kiểm tra GPU
    print("="*50)
    print("KIỂM TRA HỆ THỐNG")
    print("="*50)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: Không tìm thấy GPU! Training sẽ rất chậm.")

    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Load dữ liệu đã tokenize
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("LOADING PREPROCESSED DATA")
    print("="*50)

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục {PREPROCESSED_DATA_DIR}. Hãy upload dữ liệu trước.")

    print(f"Đang tải dữ liệu từ {PREPROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PREPROCESSED_DATA_DIR)

    print(f"Train size: {len(dataset['train']):,}")
    print(f"Validation size: {len(dataset['validation']):,}")
    print(f"Test size: {len(dataset['test']):,}")

    # OPTIMIZATION: Tạo validation set nhỏ cho quá trình training
    # Validation full (884k) quá tốn thời gian mỗi lần eval.
    # Chỉ lấy 10% (khoảng 88k) để đánh giá nhanh trong quá trình train.
    small_eval_dataset = dataset["validation"].shard(num_shards=10, index=0)
    print(f"Small Validation set for training loop: {len(small_eval_dataset):,} samples")

    # Load tokenizer
    tokenizer_path = os.path.join(PREPROCESSED_DATA_DIR, "tokenizer")
    print(f"Đang tải tokenizer từ {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---------------------------------------------------------
    # FIX: MANUAL SAFE CLASS WEIGHTS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("SETTING CLASS WEIGHTS")
    print("="*50)
    
    # Thay vì tính toán tự động (8.26), ta đặt thủ công để ổn định hơn.
    # 1.0 cho Clean, 4.0 cho Hate.
    # Điều này có nghĩa là: "Nếu bạn miss 1 cái hate, phạt gấp 4 lần miss 1 cái clean".
    # Đủ để model chú ý, nhưng không gây exploding gradient.
    
    class_weights = torch.tensor([1.0, 6.0], dtype=torch.float32).to(device)
    
    print(f"Manual Class Weights: {class_weights}")
    print("-> Sử dụng weights an toàn hơn để tránh việc model dự đoán toàn bộ là 'hate'.")

    # ---------------------------------------------------------
    # Chuẩn bị Metrics
    # ---------------------------------------------------------
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        acc = acc_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
        recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")

        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1["f1"],
            "precision_macro": precision["precision"],
            "recall_macro": recall["recall"]
        }

    # ---------------------------------------------------------
    # FIX: Custom Trainer với Weighted Loss
    # ---------------------------------------------------------
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Tính loss với class weights
            # Move logits và labels về cùng device với weights (đã là device.cuda)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # ---------------------------------------------------------
    # Load Model
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("LOADING MODEL")
    print("="*50)

    print(f"Đang tải model {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "clean", 1: "hate"},
        label2id={"clean": 0, "hate": 1}
    )

    # ---------------------------------------------------------
    # FIX: INITIALIZE BIAS (Crucial for Imbalanced Data)
    # ---------------------------------------------------------
    # Thay vì để model khởi tạo bias ngẫu nhiên (gây ra việc dự đoán 0% hate),
    # ta set bias để model ngay từ đầu đã có xu hướng dự đoán 6% hate.
    
    # Tỷ lệ hate trong train set (tính lại chính xác hoặc dùng số liệu cũ)
    # Train: 7,076,485 mẫu -> Hate: 428,138
    prior_hate = 428138 / 7076485
    
    # Công thức: bias = log(prior_hate / (1 - prior_hate))
    import math
    initial_bias = math.log(prior_hate / (1 - prior_hate))
    
    # Set bias cho classifier
    # Logit[Clean] = 0
    # Logit[Hate] = initial_bias (sẽ là số âm, vd -2.75)
    # Điều này khiến softmax output ra approx 6% cho hate ngay lập tức.
    model.classifier.bias.data = torch.tensor([0.0, initial_bias], device=device)
    
    print(f"Initialized classifier bias to: {model.classifier.bias.data}")
    print(f"-> Model bắt đầu với định kiến dữ liệu (6% hate) thay vì ngẫu nhiên.")

    # ---------------------------------------------------------
    # Training Arguments
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("CONFIGURING TRAINING")
    print("="*50)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,

        per_device_train_batch_size=BATCH_SIZE,       # Keep 64
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # Now 1

        learning_rate=LEARNING_RATE,                   # Now 5e-6
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        max_grad_norm=1.0,  # Keep this!
        num_train_epochs=EPOCHS,

        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        logging_dir='./logs',
        logging_steps=100,
        report_to=["tensorboard"],
        
        seed=42,
        data_seed=42,
        torch_compile=USE_TORCH_COMPILE, # Set to False
    )
    
    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = WeightedTrainer(  # <-- Sử dụng WeightedTrainer
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=small_eval_dataset, # <-- Dùng validation set NHỎ cho quá trình train
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, # Tăng patience vì eval_steps giờ ít hơn
                early_stopping_threshold=0.001
            )
        ]
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("BẮT ĐẦU TRAINING")
    print("="*50)

    # Clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_result = trainer.train()

    # Log training metrics
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

    # ---------------------------------------------------------
    # Final Evaluation on FULL Test Set
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL EVALUATION ON FULL TEST SET")
    print("="*50)
    
    # Bây giờ mới chạy evaluation trên toàn bộ Test set (884k mẫu)
    print("Đang đánh giá trên toàn bộ Test set (có thể mất vài phút)...")
    test_results = trainer.evaluate(dataset["test"])

    print("\nKết quả trên tập Test (Final):")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    # Save test results
    results_path = os.path.join(FINAL_MODEL_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"Model đã được lưu tại: {FINAL_MODEL_DIR}")
    print(f"Test results đã được lưu tại: {results_path}")
    print("\n" + "="*50)
    print("HOÀN TẤT!")
    print("="*50)

if __name__ == "__main__":
    main()