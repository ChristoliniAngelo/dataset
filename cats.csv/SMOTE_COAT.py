import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Baca file CSV dengan data yang sudah diimputasi
file_path_imputed = r'E:\dataset\cats.csv\cats_imputed_coat.csv'
data_imputed = pd.read_csv(file_path_imputed)

# Pilih kolom "imputed_coat"
imputed_coat_column = data_imputed['imputed_coat']

# Lakukan encoding pada kolom "imputed_coat"
label_encoder = LabelEncoder()
imputed_coat_encoded = label_encoder.fit_transform(imputed_coat_column.astype(str))

# Reshape kolom "imputed_coat" agar sesuai dengan input SMOTE
X_imputed_coat = imputed_coat_encoded.reshape(-1, 1)

# Terapkan SMOTE pada data yang sudah diimputasi
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed_coat, imputed_coat_encoded)

# Ubah kembali ke dalam bentuk string setelah resampling
resampled_imputed_coat = label_encoder.inverse_transform(y_resampled)

# Simpan hasil oversampling ke dalam file CSV baru
df_resampled = pd.DataFrame({'resampled_imputed_coat': resampled_imputed_coat})
df_resampled.to_csv(r'E:\dataset\cats_smote_resampled_coat.csv', index=False)


