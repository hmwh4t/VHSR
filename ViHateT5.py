import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Cấu hình thiết bị (Ưu tiên GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng: {device}")

# 2. Lựa chọn Model
MODEL_NAME = "tarudesu/ViHateT5-base-HSD" 

print("Đang tải model vào máy...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def detect_hate_speech(input):
    """
    Input: Chuỗi văn bản tiếng Việt.
    Output: Nhãn (Là clean/offensive/hate đối với 'hate-speech-detection').

    Input ViHateT5: text = prefix + ': ' + input
    Các lựa chọn prefix: ['hate-speech-detection', 'toxic-speech-detection', 'hate-spans-detection']
    """
    text = 'hate-speech-detection' + ': ' + input

    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=10)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction

# Giả lập dữ liệu từ Speech-to-Text
stt_inputs = [
    # 1. Clean
    "hôm nay trời đẹp tôi muốn đi chơi",
    "tôi không thích cách anh nói chuyện lắm",
    "alo alo alo nhôm nay đi nhậu nhé anh em",
    "vải hơi xấu mà giá rẻ thì cũng tạm được",
    "tôi biến ngựa thành keo vì tôi ghét ngựa",
    
    # 2. Offensive
    "giết nó đi",
    "cái máy tính này chậm vãi muốn đập nát nó ra",
    
    # Hate
    "mấy thằng da đen này bẩn thỉu quá",
    "mày đi đứng kiểu gì đấy mắt để dưới chân à đồ ngu",
    "nói mãi mà không hiểu à cái thằng này đầu đất thật sự",
    "bọn này ngứa mắt vãi đánh chết nó đi",
]

print("\n--- Kết Quả ---")
for text in stt_inputs:
    # Label 'clean': Sạch
    # Label 'offensive': Xúc phạm
    # Label 'hate': Thù ghét
    label = detect_hate_speech(text)

    meaning = "Sạch (Clean)"
    if "offensive" in label.lower():
        meaning = "Xúc phạm (Offensive)"
    elif "hate" in label.lower():
        meaning = "Thù ghét (Hate)"

    print(f"Input: {text}")
    print(f"Output: \'{label}\' -> {meaning}")
    print("-" * 30)