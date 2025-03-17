import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Definir directorio y archivos
directorio = r"C:\Users\Ricardo González\Downloads\dataset"
archivos_csv = {
    "2024PM25.csv": "PM25",
    "2024PM10.csv": "PM10",
    "2024NO2.csv": "NO2",
    "2024O3.csv": "O3",
    "2024SO2.csv": "SO2",
    "2024TMP.csv": "Temperatura",
    "2024RH.csv": "Humedad",
    "2024WSP.csv": "Velocidad del viento",
    "2024WDR.csv": "Direccion del viento",
    "2024PMCO.csv": "Precipitacion"
}

df_final = None

# Cargar y transformar los datos
for archivo, variable in archivos_csv.items():
    ruta = os.path.join(directorio, archivo)
    df = pd.read_csv(ruta, encoding="utf-8")
    df_melted = df.melt(id_vars=["FECHA", "HORA"], var_name="Estacion", value_name=variable)
    if df_final is None:
        df_final = df_melted
    else:
        df_final = df_final.merge(df_melted, on=["FECHA", "HORA", "Estacion"], how="outer")

# Convertir fecha
df_final['FECHA'] = pd.to_datetime(df_final['FECHA'], format='%d/%m/%Y', errors='coerce')

# Limpiar y transformar los datos
columns_to_clean = ["PM25", "PM10", "NO2", "O3", "SO2", "Temperatura", "Humedad", "Velocidad del viento"]
df_final[columns_to_clean] = df_final[columns_to_clean].apply(pd.to_numeric, errors='coerce')
df_final = df_final.dropna(thresh=len(df_final) * 0.5, axis=1)
df_final[columns_to_clean] = df_final[columns_to_clean].fillna(df_final[columns_to_clean].mean())
df_final[columns_to_clean] = df_final[columns_to_clean].ffill().bfill()

# Normalizar los datos
scaler = StandardScaler()
df_final[columns_to_clean] = scaler.fit_transform(df_final[columns_to_clean])

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df_final[columns_to_clean].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación")
plt.show()

# Regresión con Random Forest
X = df_final[['Temperatura', 'Humedad', 'Velocidad del viento']]
y = df_final['PM25']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.title("Análisis de Componentes Principales")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Clustering con K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
plt.scatter(X['Temperatura'], X['Humedad'], c=labels)
plt.title("Clustering con K-Means")
plt.xlabel("Temperatura")
plt.ylabel("Humedad")
plt.show()

# Validación cruzada
scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Scores de Validacion Cruzada:", scores)

# Ajuste de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print("Mejores parámetros:", grid_search.best_params_)
