import pandas as pd
from sklearn.impute import SimpleImputer

# Baca file CSV
file_path = r'E:\dataset\cats.csv\cats.csv'
data = pd.read_csv(file_path)

# Pilih kolom "coat"
coat_column = data['coat']

# Buat objek SimpleImputer untuk mengisi nilai kosong dengan modus
imputer = SimpleImputer(strategy='most_frequent')

# Reshape kolom "coat" agar sesuai dengan input SimpleImputer
X_imputer = coat_column.values.reshape(-1, 1)

# Lakukan imputasi pada nilai kosong dan flatten hasilnya
data['imputed_coat'] = imputer.fit_transform(X_imputer).reshape(-1)

# Simpan data imputasi pada kolom "coat" ke dalam file CSV baru
data[['imputed_coat']].to_csv(r'E:\dataset\cats_imputed_coat.csv', index=False)
