import pandas as pd

# Baca file CSV dataset asli
file_path_original = r'E:\dataset\cats.csv\cats.csv'
data_original = pd.read_csv(file_path_original)

# Baca file CSV hasil oversampling
file_path_oversampled = r'E:\dataset\cats_smote_resampled_coat.csv'
data_oversampled = pd.read_csv(file_path_oversampled)

# Ganti kolom "coat" dengan "resampled_imputed_coat" dalam dataset asli
data_original['coat'] = data_oversampled['resampled_imputed_coat']

# Simpan dataset yang telah diupdate ke dalam file CSV baru
data_original.to_csv(r'E:\dataset\cats_updated.csv', index=False)
