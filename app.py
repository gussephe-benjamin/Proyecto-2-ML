import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import os
from sklearn.metrics.pairwise import euclidean_distances
from feature_extraction import extract_features
from PIL import Image

# Cargar modelos y datos
svd = joblib.load('svd_model.pkl')
gmm = joblib.load('gmm_model.pkl')
df = pd.read_csv('features_svd_with_clusters.csv')

st.title("🔍 Búsqueda de imágenes similares y por cluster")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Imagen Similar",
    "🎯 Buscar por Cluster",
    "🎞️ Filtrar por Año",
    "🎬 Filtrar por Género"
])

with tab1:
    uploaded_file = st.file_uploader("Sube una imagen:", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen subida", use_column_width=True)

        # Extraer características
        features = extract_features(img)
        features_svd = svd.transform([features])
        cluster_pred = gmm.predict(features_svd)[0]

        # Filtrar por cluster
        df_cluster = df[df['Cluster'] == cluster_pred]
        X_cluster = df_cluster.drop(columns=['ImageName', 'Cluster'])
        distances = euclidean_distances(features_svd, X_cluster)[0]
        df_cluster['Distance'] = distances

        top_images = df_cluster.nsmallest(6, 'Distance')['ImageName'].values

        st.subheader("📸 Imágenes similares:")
        for img_name in top_images:
            img_path = os.path.join("data/imagenes", img_name)
            if os.path.exists(img_path):
                st.image(img_path, width=200, caption=img_name)
            else:
                st.warning(f"⚠️ Imagen no encontrada: {img_name}")

with tab2:
    num_cluster = st.number_input("Ingresa el número de cluster (0-23):", min_value=0, max_value=23, step=1)

    if st.button("Mostrar imágenes del cluster"):
        cluster_imgs = df[df['Cluster'] == num_cluster]['ImageName'].values

        if len(cluster_imgs) == 0:
            st.warning("No hay imágenes en ese cluster.")
        else:
            st.subheader(f"📂 Imágenes del cluster {num_cluster}")
            for name in cluster_imgs[:12]:
                path = os.path.join("data/imagenes", name)
                if os.path.exists(path):
                    st.image(path, width=200, caption=name)
                else:
                    st.warning(f"⚠️ Imagen no encontrada: {name}")

with tab3:
    st.subheader("🎞️ Películas por Año")
    df_movies = pd.read_csv("MovieGenre.csv", encoding="latin1")

    df_movies["GenreList"] = df_movies["Genre"].fillna("").apply(lambda x: x.split("|"))
    df_movies["Year"] = df_movies["Title"].str.extract(r"\((\d{4})\)").astype(float)
    df_movies = df_movies.dropna(subset=["Year"])
    df_movies["Year"] = df_movies["Year"].astype(int)

    years = sorted(df_movies["Year"].unique())
    selected_year = st.selectbox("Selecciona un año:", years)

    filtered_movies = df_movies[df_movies["Year"] == selected_year]

    if filtered_movies.empty:
        st.warning("No hay películas para este año.")
    else:
        st.success(f"{len(filtered_movies)} película(s) encontradas en {selected_year}.")
        for idx, row in filtered_movies.iterrows():
            imdb_id = str(int(row["imdbId"])) if pd.notna(row["imdbId"]) else None
            if imdb_id:
                local_path = os.path.join("data/imagenes", imdb_id + ".jpg")
                if os.path.exists(local_path):
                    st.image(local_path, width=200, caption=row["Title"])
                else:
                    # Intentar descargar el póster desde el link
                    if "Poster" in df_movies.columns and pd.notna(row["Poster"]):
                        try:
                            import requests
                            response = requests.get(row["Poster"], timeout=10)
                            if response.status_code == 200:
                                # Guardar imagen localmente
                                with open(local_path, "wb") as f:
                                    f.write(response.content)
                                st.image(local_path, width=200, caption=row["Title"])
                            else:
                                st.warning(f"No se pudo descargar el póster de {row['Title']}")
                        except Exception as e:
                            st.warning(f"Error al descargar el póster de {row['Title']}: {e}")
                    else:
                        st.warning(f"No hay enlace de póster para {row['Title']}")


with tab4:
    st.subheader("🎬 Películas por Género")

    df_movies = pd.read_csv("MovieGenre.csv", encoding="latin1")
    df_movies["GenreList"] = df_movies["Genre"].fillna("").apply(lambda x: x.split("|"))
    genres = sorted(set(genre for sublist in df_movies["GenreList"] for genre in sublist if genre))

    selected_genre = st.selectbox("Selecciona un género:", genres)

    filtered_genre = df_movies[df_movies["GenreList"].apply(lambda g: selected_genre in g)]

    if filtered_genre.empty:
        st.warning("No hay películas con ese género.")
    else:
        st.success(f"{len(filtered_genre)} película(s) encontradas del género '{selected_genre}'.")
        for idx, row in filtered_genre.iterrows():
            imdb_id = str(int(row["imdbId"])) if pd.notna(row["imdbId"]) else None
            if imdb_id:
                local_path = os.path.join("data/imagenes", imdb_id + ".jpg")
                if os.path.exists(local_path):
                    st.image(local_path, width=200, caption=row["Title"])
                else:
                    print("")