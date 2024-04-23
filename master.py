# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score, silhouette_score, homogeneity_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Fonction pour calculer la pureté des clusters
def purity_score(true_labels, cluster_labels):
    # Calculer la pureté des clusters
    contingency_matrix = confusion_matrix(true_labels, cluster_labels)
    max_counts = np.max(contingency_matrix, axis=0)
    total_count = np.sum(contingency_matrix)
    purity = np.sum(max_counts) / total_count
    return purity

# Charger le jeu de données Iris depuis scikit-learn
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target  # Vraies étiquettes des classes

# Aperçu des données
print(data.head())

# Analyse exploratoire des données (EDA)
# Visualisation des distributions des caractéristiques
data.hist(bins=20, figsize=(12, 8))
plt.suptitle('Distribution des caractéristiques des données Iris')
plt.show()

# Visualisation des relations entre les caractéristiques
pd.plotting.scatter_matrix(data, c=true_labels, figsize=(12, 12), marker='o', hist_kwds={'bins': 20}, alpha=0.5)
plt.suptitle('Relations entre les caractéristiques')
plt.show()

# Examen des corrélations entre les caractéristiques
correlation_matrix = data.corr()
print("Matrice de corrélation des caractéristiques :")
print(correlation_matrix)

# Normaliser les données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Choisir le nombre de clusters pour K-Means
num_clusters = 3  # Choix du nombre de clusters (correspondant aux trois espèces)

# Initialiser et entraîner l'algorithme K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Ajouter les labels de cluster aux données
data['Cluster'] = kmeans.labels_

# Visualiser les clusters avec K-Means
plt.figure(figsize=(10, 8))
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Clusters avec K-Means dans le jeu de données Iris')
plt.show()

# Visualisation des clusters en 3D avec K-Means
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['sepal length (cm)'], data['sepal width (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
plt.title('Visualisation 3D des clusters avec K-Means dans le jeu de données Iris')
plt.show()

# Analyse des caractéristiques moyennes des clusters
print(data.groupby('Cluster').mean())

# Évaluation des résultats de K-Means
# Calcul de l'exactitude (accuracy)
accuracy = accuracy_score(true_labels, data['Cluster'])
print(f"Accuracy avec K-Means: {accuracy:.2f}")

# Calcul de la pureté des clusters
purity = purity_score(true_labels, data['Cluster'])
print(f"Purity avec K-Means: {purity:.2f}")

# Calcul de l'Indice de Rand Ajusté (Adjusted Rand Index)
ari = adjusted_rand_score(true_labels, data['Cluster'])
print(f"Indice de Rand Ajusté avec K-Means: {ari:.2f}")

# Calcul du score de silhouette
silhouette = silhouette_score(scaled_features, data['Cluster'])
print(f"Score de silhouette avec K-Means: {silhouette:.2f}")

# Classification report
report = classification_report(true_labels, data['Cluster'])
print("Rapport de classification avec K-Means:")
print(report)

# Évaluation par courbe d'inertie avec K-Means
inertia = kmeans.inertia_
print(f"Inertie avec K-Means: {inertia:.2f}")

# Visualisation de la courbe d'inertie pour K-Means
inertia_values = []
for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_features)
    inertia_values.append(kmeans_model.inertia_)
plt.figure()
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Courbe d\'inertie pour K-Means')
plt.show()

# Visualisation de la distribution des clusters dans chaque classe
for cluster in np.unique(data['Cluster']):
    plt.figure(figsize=(8, 6))
    mask = data['Cluster'] == cluster
    plt.hist(true_labels[mask], bins=3, alpha=0.7, label=f'Cluster {cluster}')
    plt.xlabel('Classe réelle')
    plt.ylabel('Nombre')
    plt.title(f'Distribution des classes réelles pour le Cluster {cluster}')
    plt.legend()
    plt.show()

# Comparaison avec un autre algorithme de clustering : AgglomerativeClustering
agglo = AgglomerativeClustering(n_clusters=num_clusters)
agglo_labels = agglo.fit_predict(scaled_features)

# Calcul de la pureté pour AgglomerativeClustering
agglo_purity = purity_score(true_labels, agglo_labels)
print(f"Purity avec AgglomerativeClustering: {agglo_purity:.2f}")

# Calcul de l'Indice de Rand Ajusté (Adjusted Rand Index) pour AgglomerativeClustering
agglo_ari = adjusted_rand_score(true_labels, agglo_labels)
print(f"Indice de Rand Ajusté avec AgglomerativeClustering: {agglo_ari:.2f}")

# Calcul du score de silhouette pour AgglomerativeClustering
agglo_silhouette = silhouette_score(scaled_features, agglo_labels)
print(f"Score de silhouette avec AgglomerativeClustering: {agglo_silhouette:.2f}")

# Comparaison des résultats entre K-Means et AgglomerativeClustering
print("Comparaison des résultats entre K-Means et AgglomerativeClustering:")
print(f"Purity avec K-Means: {purity:.2f}, Purity avec AgglomerativeClustering: {agglo_purity:.2f}")
print(f"Indice de Rand Ajusté avec K-Means: {ari:.2f}, Indice de Rand Ajusté avec AgglomerativeClustering: {agglo_ari:.2f}")
print(f"Score de silhouette avec K-Means: {silhouette:.2f}, Score de silhouette avec AgglomerativeClustering: {agglo_silhouette:.2f}")

# Conclusion sur les performances des algorithmes
print("Les résultats montrent que K-Means et AgglomerativeClustering sont tous les deux capables de trouver des clusters significatifs dans le jeu de données Iris.")
print("Les scores de pureté, d'ARI et de silhouette suggèrent des performances similaires entre les deux algorithmes.")
print("L'algorithme le plus approprié dépendra des besoins spécifiques de l'application, de la nature des données et des objectifs de l'analyse.")
