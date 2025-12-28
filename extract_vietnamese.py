"""Extract Vietnamese translations from bz2 JSONL file."""
import bz2
import json
import csv

input_file = "cc3m_mt_train.jsonl.bz2"
output_file = "vietnamese_translations.csv"

with bz2.open(input_file, "rt", encoding="utf-8") as f_in, \
     open(output_file, "w", newline="", encoding="utf-8") as f_out:
    
    writer = csv.writer(f_out)
    writer.writerow(["translation_tokenized"])
    
    count = 0
    for line in f_in:
        obj = json.loads(line)
        if obj.get("trg_lang") == "vi":
            writer.writerow([obj.get("translation_tokenized", "")])
            count += 1

print(f"Extracted {count} Vietnamese translations to {output_file}")
