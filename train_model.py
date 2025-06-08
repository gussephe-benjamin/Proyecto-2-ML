import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_features

# Directorio de imágenes
image_dir = 'data/imagenes'

# Listas para almacenar datos
features_list = []
names = []

# Procesar todas las imágenes
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_dir, filename)
        img = cv2.imread(path)
        if img is not None:
            feats = extract_features(img)
            features_list.append(feats)
            names.append(filename)

# Convertir a numpy array
X = np.array(features_list)

# SVD para reducir a 20 dimensiones
svd = TruncatedSVD(n_components=20, random_state=42)
X_svd = svd.fit_transform(X)
joblib.dump(svd, 'svd_model.pkl')

# GMM para agrupar en 24 clusters
gmm = GaussianMixture(n_components=24, random_state=42)
clusters = gmm.fit_predict(X_svd)
joblib.dump(gmm, 'gmm_model.pkl')

# Guardar los datos procesados
df = pd.DataFrame(X_svd)
df['ImageName'] = names
df['Cluster'] = clusters
df.to_csv('features_svd_with_clusters.csv', index=False)

print("✅ Modelos entrenados y CSV generado con éxito.")
