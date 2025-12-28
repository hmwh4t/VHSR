"""
Clean text data in vietnamese_translations.csv:
- Remove symbols: " . # and other special characters
- Keep only letters, numbers, spaces
- Remove trailing/leading spaces
- Normalize multiple spaces to single space
"""
import pandas as pd
import re

INPUT_FILE = 'vietnamese_translations.csv'
OUTPUT_FILE = 'vietnamese_translations.csv'  # Overwrite the same file

def clean_text(text):
    """Clean a single text string."""
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Remove symbols except letters, numbers, and spaces
    # Keep Vietnamese characters (Unicode letters)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    
    # Remove extra whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    return text

# Read the CSV (no header in vietnamese_translations.csv)
df = pd.read_csv(INPUT_FILE, header=None, names=['text'])
print(f"Loaded {len(df)} rows")

# Show sample before cleaning
print("\nBefore cleaning:")
print(df['text'].head(3).tolist())

# Clean the text column
df['text'] = df['text'].apply(clean_text)

# Show sample after cleaning
print("\nAfter cleaning:")
print(df['text'].head(3).tolist())

# Save the cleaned CSV (without header to match original format)
df.to_csv(OUTPUT_FILE, index=False, header=False)
print(f"\nSaved cleaned data to {OUTPUT_FILE}")
