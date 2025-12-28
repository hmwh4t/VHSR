from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import re
import emoji
import sys

# 1. Đăng nhập (Để truy cập ViHSD)
login(token="")

print("Đang tải dataset...")

# ---------------------------------------------------------
# sonlam1102/vihsd
# ---------------------------------------------------------
try:
    ds_vihsd = load_dataset("sonlam1102/vihsd")
    if hasattr(ds_vihsd, 'keys'):
        list_dfs = []
        for split in ds_vihsd.keys():
            df_temp = ds_vihsd[split].to_pandas()
            list_dfs.append(df_temp)
        df_vihsd = pd.concat(list_dfs, ignore_index=True)
    else:
        df_vihsd = ds_vihsd.to_pandas()

    # Rename: free_text -> text
    df_vihsd = df_vihsd.rename(columns={'free_text': 'text'})

    # Map Label: 0 -> clean; 1, 2 -> hate
    def process_label_vihsd(label_id):
        if label_id == 0:
            return 'clean'
        else:
            return 'hate'

    df_vihsd['label'] = df_vihsd['label_id'].apply(process_label_vihsd)
    df_vihsd = df_vihsd[['text', 'label']]
    print(f"Đã xử lý ViHSD: {len(df_vihsd)} dòng.")

except Exception as e:
    print(f"Lỗi khi xử lý ViHSD: {e}")
    df_vihsd = pd.DataFrame(columns=['text', 'label'])

# ---------------------------------------------------------
# tarudesu/VOZ-HSD
# ---------------------------------------------------------
try:
    ds_voz = load_dataset("tarudesu/VOZ-HSD", split="train") 
    df_voz = ds_voz.to_pandas()

    # Rename: texts -> text
    df_voz = df_voz.rename(columns={'texts': 'text'})

    # Map label: 0 -> clean, 1 -> hate
    def process_label_voz(lbl):
        if lbl == 0:
            return 'clean'
        return 'hate'

    df_voz['label'] = df_voz['labels'].apply(process_label_voz)
    df_voz = df_voz[['text', 'label']]
    print(f"Đã xử lý VOZ-HSD: {len(df_voz)} dòng.")

except Exception as e:
    print(f"Lỗi khi xử lý VOZ-HSD: {e}")
    df_voz = pd.DataFrame(columns=['text', 'label'])

# ---------------------------------------------------------
# Gộp và lưu CSV
# ---------------------------------------------------------
print("Đang gộp dữ liệu...")
df_final = pd.concat([df_vihsd, df_voz], ignore_index=True)

print("Đang xóa dữ liệu trùng lặp...")
df_final.drop_duplicates(subset=['text'], inplace=True)

print("Thống kê dữ liệu sau khi gộp:")
print(df_final['label'].value_counts())

output_file = "UnprocessedSTT.csv"
print(f"Đang lưu file {output_file}...")
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print("Hoàn tất!")

# ---------------------------------------------------------
# Preprocessing: Chuẩn hóa sang dạng Speech To Text
# ---------------------------------------------------------
teencode_dict = {
    "k": "không", "ko": "không", "kh": "không", "khg": "không", "hok": "không",
    "dc": "được", "đc": "được", "dk": "được",
    "dm": "địt mẹ", "đm": "địt mẹ", "dkm": "địt con mẹ", "đkm": "địt con mẹ", "vcl": "vãi cả lồn",
    "vl": "vãi lồn", "loz": "lồn", "l": "lồn",
    "ck": "chồng", "vk": "vợ",
    "dt": "điện thoại", "đt": "điện thoại",
    "thik": "thích", "thix": "thích",
    "ms": "mới",
    "trc": "trước",
    "ng": "người", "n": "người",
    "vs": "với",
    "vn": "việt nam",
    "uh": "ừ", "uhm": "ừ",
    "r": "rồi",
    "bh": "bây giờ",
    "h": "giờ",
    "wa": "quá", "qua": "quá",
    "gud": "tốt",
    "t": "tao", "m": "mày",
    "ch": "chưa",
    "bit": "biết", "bik": "biết",
    "gd": "gia đình",
    "kb": "không biết",
    "msg": "tin nhắn",
    "fl": "theo dõi", "follow": "theo dõi",
    "ad": "quản trị viên", "admin": "quản trị viên",
    "ib": "nhắn tin", "inbox": "nhắn tin",
    "cmt": "bình luận",
    "add": "thêm",
    "thanks": "cảm ơn", "tks": "cảm ơn", "tk": "cảm ơn",
    "view": "lượt xem",
    "share": "chia sẻ",
    "like": "thích",
    "dl": "dữ liệu",
    "klq": "không liên quan",
    "gt": "giới thiệu",
    "hp": "hạnh phúc",
    "hn": "hôm nay",
    "hqua": "hôm qua",
    "t7": "thứ bảy", "cn": "chủ nhật",
    "b": "bạn",
    "mk": "mình", "mik": "mình",
    "j": "gì", "z": "gì",
    "v": "vậy",
    "cx": "cũng"
}

# Tạo Regex Pattern sắp xếp thay thế theo độ dài giảm dần
sorted_teencode = sorted(teencode_dict.keys(), key=len, reverse=True)
pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_teencode) + r')\b', re.IGNORECASE)

def replace_teencode(text):
    """
    Hàm thay thế từ viết tắt bằng từ đầy đủ, Regex Boundary đảm bảo chỉ thay thế từ đứng riêng lẻ.
    VD: 'k' -> 'không', 'kiến' vẫn là 'kiến'.
    """
    if not isinstance(text, str):
        return str(text)
    
    def replace(match):
        word = match.group(0).lower()
        return teencode_dict.get(word, word)
    
    return pattern.sub(replace, text)

def clean_text_for_stt(text):
    if not isinstance(text, str):
        return ""

    # 1. Xóa Emoji
    text = emoji.replace_emoji(text, replace='')

    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Xóa URL/Link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 4. Xóa User Tag (@user) hoặc Hashtag (#tag)
    text = re.sub(r'@\w+', '', text) 

    # 5. Xử lý viết tắt
    text = replace_teencode(text)

    # 6. Xóa tất cả những ký tự không phải là chữ cái, số, khoảng trắng hoặc chữ Việt
    # "Vui quá :)) =))" --> "Vui quá"
    text = re.sub(r'[^\w\sđĐàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]', ' ', text)

    # 7. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ---------------------------------------------------------
# Chạy Preprocessing và lưu CSV
# ---------------------------------------------------------
input_file = "UnprocessedSTT.csv"
output_file = "CleanSTT.csv"

print(f"Đang đọc {input_file}...")
try:
    df = pd.read_csv(input_file)
    
    print("Đang làm sạch dữ liệu...")
    df['text_clean'] = df['text'].astype(str).apply(clean_text_for_stt)
    
    # Loại bỏ các dòng vô nghĩa sau khi làm sạch
    original_len = len(df)
    df = df[df['text_clean'].str.len() > 1] # Giữ dòng có ít nhất 2 ký tự
    print(f"Đã loại bỏ {original_len - len(df)} dòng vô nghĩa.")

    final_df = df[['text_clean', 'label']]
    final_df.rename(columns={'text_clean': 'text'}, inplace=True)

    print(f"Đang lưu {output_file}...")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

except FileNotFoundError:
    print("Không tìm thấy file .csv đầu vào.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
