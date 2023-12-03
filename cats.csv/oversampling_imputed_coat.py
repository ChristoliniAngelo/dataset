import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Baca file CSV dengan data yang sudah diimputasi
file_path_imputed = r'E:\dataset\cats.csv\cats_imputed_coat.csv'
data_imputed = pd.read_csv(file_path_imputed)

# Pilih kolom "imputed_coat"
imputed_coat_column = data_imputed['imputed_coat']

# Lakukan encoding pada kolom "imputed_coat"
label_encoder = LabelEncoder()
imputed_coat_encoded = label_encoder.fit_transform(imputed_coat_column.astype(str))

# Tentukan indeks data yang memiliki nilai kosong pada kolom "imputed_coat"
missing_imputed_coat_indices = data_imputed[data_imputed['imputed_coat'].isnull()].index

# Pastikan ada sampel yang memiliki nilai pada kolom "imputed_coat" sebelum melakukan oversampling
if len(missing_imputed_coat_indices) > 0:
    # Reshape kolom "imputed_coat" yang sudah di-encoded menjadi bentuk yang sesuai dengan input RandomOverSampler
    X_imputed_coat = imputed_coat_encoded.reshape(-1, 1)

    # Terapkan RandomOverSampler hanya pada data yang memiliki nilai kosong pada kolom "imputed_coat"
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_imputed_coat[missing_imputed_coat_indices], imputed_coat_encoded[missing_imputed_coat_indices])

    # Ubah kembali ke dalam bentuk string setelah resampling
    resampled_imputed_coat = label_encoder.inverse_transform(y_resampled)

    # Isi data yang kosong pada kolom "imputed_coat" dengan data yang sudah di-resample
    data_imputed.loc[missing_imputed_coat_indices, 'imputed_coat'] = resampled_imputed_coat

    # Simpan data yang sudah diimputasi dan diresample ke dalam file CSV baru
    data_imputed.to_csv(r'E:\dataset\cats_imputed_resampled_coat.csv', index=False)
else:
    print("No missing values found in 'imputed_coat' column.")
