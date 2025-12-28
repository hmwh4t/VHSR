import pandas as pd

df = pd.read_csv('vietnamese_translations.csv', header=None, names=['text'], on_bad_lines='skip')
print(f'Before: {len(df)} rows')
df = df[df['text'].str.contains('con', case=False, na=False)]
print(f'After: {len(df)} rows')
df.to_csv('vietnamese_translations.csv', index=False, header=False)
print('Done!')
