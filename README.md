Explication détaillée du code K-Means sur le jeu de données Iris

Ce projet met en œuvre l'algorithme de clustering K-Means sur le jeu de données Iris. L'objectif est d'identifier les clusters (groupes) dans les données d'iris et de les évaluer par rapport aux étiquettes réelles des espèces d'iris.
1. Importation des bibliothèques nécessaires

Le code commence par importer les bibliothèques nécessaires pour manipuler et visualiser les données, ainsi que pour mettre en œuvre l'algorithme K-Means :

python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score
from sklearn.metrics import silhouette_score

    pandas et numpy : Ces bibliothèques sont utilisées pour manipuler et analyser les données de manière efficace.
    matplotlib.pyplot : Utilisée pour visualiser les données et les résultats du clustering.
    KMeans : Algorithme K-Means de la bibliothèque scikit-learn pour réaliser le clustering.
    StandardScaler : Permet de normaliser les données afin de faciliter l'application de l'algorithme.
    datasets : Permet de charger des jeux de données standards tels que Iris.
    accuracy_score, confusion_matrix, classification_report, adjusted_rand_score : Mesures de performance pour évaluer la qualité du clustering.
    silhouette_score : Mesure la qualité du clustering.

2. Chargement des données

Les données Iris sont chargées à partir de la bibliothèque datasets de scikit-learn :

python

iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target  # Étiquettes réelles des classes

# Aperçu des données
print(data.head())

    datasets.load_iris() : Charge le jeu de données Iris.
    pd.DataFrame : Convertit les données en DataFrame pour les manipuler plus facilement.
    iris.feature_names : Noms des colonnes dans les données.
    iris.target : Étiquettes réelles des classes (0, 1, 2) correspondant aux trois espèces d'iris : Setosa, Versicolor, et Virginica.
    print(data.head()) : Affiche les cinq premières lignes du DataFrame pour voir un aperçu des données.

3. Prétraitement des données

Les données sont préparées pour l'algorithme K-Means :

python

# Sélectionner les caractéristiques pour le clustering
features = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Normaliser les données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

    features : Sélectionne les caractéristiques pertinentes pour le clustering (toutes les caractéristiques dans ce cas).
    StandardScaler : Standardise les caractéristiques pour leur donner la même échelle.
    scaled_features : Les données normalisées sont utilisées pour l'entraînement du modèle.

4. Choix et entraînement de l'algorithme K-Means

L'algorithme K-Means est configuré et entraîné avec les données normalisées :

python

# Choisir le nombre de clusters
num_clusters = 3  # Choix du nombre de clusters (correspondant aux trois espèces d'iris)

# Initialiser et entraîner l'algorithme K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

    num_clusters : Choix du nombre de clusters à identifier (3 dans ce cas, car il y a trois espèces d'iris).
    KMeans : Crée l'algorithme K-Means avec le nombre de clusters spécifié et une graine aléatoire fixe (random_state).
    kmeans.fit() : Entraîne l'algorithme K-Means sur les données normalisées.

5. Ajout des labels de cluster et visualisation

Les résultats de l'algorithme K-Means sont analysés et visualisés :

python

# Ajouter les labels de cluster aux données
data['Cluster'] = kmeans.labels_

# Visualiser les clusters
plt.figure(figsize=(10, 8))
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Clusters dans le jeu de données Iris')
plt.show()

    kmeans.labels_ : Labels des clusters trouvés par l'algorithme K-Means.
    data['Cluster'] : Ajoute les labels de cluster aux données.
    plt.scatter() : Affiche un nuage de points des données, coloré selon les clusters.

6. Visualisation 3D des clusters

Une visualisation 3D des clusters est créée pour mieux comprendre la distribution des données :

python

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['sepal length (cm)'], data['sepal width (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
plt.title('Visualisation 3D des clusters dans le jeu de données Iris')
plt.show()

    mpl_toolkits.mplot3d : Importation de la bibliothèque pour la visualisation 3D.
    ax.scatter() : Affiche un nuage de points en 3D coloré selon les clusters.
    set_xlabel, set_ylabel, set_zlabel : Labels des axes.

7. Analyse des caractéristiques des clusters

Les caractéristiques des clusters sont analysées et comparées pour observer les différences :

python

print(data.groupby('Cluster').mean())

    groupby : Groupe les données par cluster.
    mean : Calcule la moyenne des caractéristiques pour chaque cluster.

8. Évaluation des résultats

Les performances des résultats sont évaluées par rapport aux étiquettes réelles des espèces d'iris :
Exactitude (Accuracy)

L'exactitude mesure la proportion d'exemples correctement classés :

python

# Calcul de l'exactitude (accuracy)
accuracy = accuracy_score(true_labels, data['Cluster'])
print(f"Accuracy: {accuracy:.2f}")

    accuracy_score : Calcule l'exactitude des prédictions par rapport aux étiquettes réelles.
    Le résultat affiché est la proportion d'exemples correctement classés.

Matrice de confusion

La matrice de confusion visualise les erreurs de classification :

python

# Calcul de la matrice de confusion
confusion = confusion_matrix(true_labels, data['Cluster'])
print("Matrice de confusion :")
print(confusion)

    confusion_matrix : Calcule la matrice de confusion qui montre les bonnes et mauvaises prédictions.
    La matrice affiche le nombre de fois où chaque cluster correspond à chaque étiquette réelle.

Indice de Rand Ajusté (ARI)

L'Indice de Rand Ajusté mesure la similarité entre les clusters trouvés et les étiquettes réelles :

python

# Calcul de l'Indice de Rand Ajusté (Adjusted Rand Index)
ari = adjusted_rand_score(true_labels, data['Cluster'])
print(f"Indice de Rand Ajusté: {ari:.2f}")

    adjusted_rand_score : Calcule la similarité entre les clusters trouvés et les étiquettes réelles.
    Plus l'ARI est élevé, plus les résultats du clustering sont en accord avec les étiquettes réelles.

Score de silhouette

Le score de silhouette mesure la qualité des clusters :

python

# Calcul du score de silhouette
silhouette = silhouette_score(scaled_features, data['Cluster'])
print(f"Score de silhouette: {silhouette:.2f}")

    silhouette_score : Calcule la cohésion et la séparation des clusters.
    Un score plus élevé indique des clusters bien définis.
