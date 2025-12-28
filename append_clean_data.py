"""
Append vietnamese_translations.csv rows with 'clean' label 
randomly into CleanSTT.csv
"""
import pandas as pd
import random

# Read the existing CleanSTT.csv
clean_stt = pd.read_csv('CleanSTT.csv')
print(f"Original CleanSTT.csv: {len(clean_stt)} rows")

# Read vietnamese_translations.csv (no header)
viet_trans = pd.read_csv('vietnamese_translations.csv', header=None, names=['text'])
print(f"Vietnamese translations to add: {len(viet_trans)} rows")

# Add 'clean' label to all rows
viet_trans['label'] = 'clean'

# Combine the dataframes
combined = pd.concat([clean_stt, viet_trans], ignore_index=True)

# Shuffle the combined dataframe randomly
combined = combined.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)

# Save back to CleanSTT.csv
combined.to_csv('CleanSTT.csv', index=False)

print(f"New CleanSTT.csv: {len(combined)} rows")
print("Done! Rows have been shuffled randomly.")
