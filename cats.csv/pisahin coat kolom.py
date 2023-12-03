import pandas as pd

# Baca file CSV
data = pd.read_csv('E:\\dataset\\cats.csv\\cats.csv')

# Pilih kolom "coat"
coat_column = data['coat']

import os
save_directory = 'E:\\dataset\\cats.csv'
os.makedirs(save_directory, exist_ok=True)

# Simpan file CSV
coat_column.to_csv(os.path.join(save_directory, 'coat_column.csv'), index=False)

# Simpan kolom "coat" ke file terpisah 
coat_column.to_csv('E:\\dataset\\cats.csv', index=False)

# Tampilkan kolom "coat"
print(coat_column)
