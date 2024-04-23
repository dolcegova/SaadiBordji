Explication détaillée du code K-Means sur le jeu de données Iris

Ce projet met en œuvre l'algorithme de clustering K-Means sur le jeu de données Iris. L'objectif est d'identifier les clusters (groupes) dans les données d'iris et de les évaluer par rapport aux étiquettes réelles des espèces d'iris.

### Étape 1 : Importer les bibliothèques nécessaires
Le code importe les bibliothèques nécessaires pour effectuer les opérations de clustering, l'analyse et la visualisation des données.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score, silhouette_score, homogeneity_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
```

- `pandas` pour la manipulation des données.
- `numpy` pour les opérations numériques.
- `matplotlib.pyplot` pour la visualisation graphique.
- `sklearn.cluster` pour les algorithmes de clustering K-Means et AgglomerativeClustering.
- `sklearn.preprocessing` pour la normalisation des données.
- `sklearn.datasets` pour charger le jeu de données Iris.
- `sklearn.metrics` pour calculer les métriques de performance.
- `mpl_toolkits.mplot3d` pour la visualisation 3D.

### Étape 2 : Charger le jeu de données Iris
Le jeu de données Iris est chargé à partir de scikit-learn, et les données sont stockées dans un DataFrame `data`. Les véritables étiquettes de classes sont également stockées dans `true_labels`.

```python
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target  # Vraies étiquettes des classes

# Aperçu des données
print(data.head())
```

- `datasets.load_iris()` charge le jeu de données Iris.
- Les données sont stockées dans `data`.
- Les véritables étiquettes de classe sont stockées dans `true_labels`.
- `data.head()` affiche un aperçu des données.

### Étape 3 : Analyse exploratoire des données (EDA)
L'analyse exploratoire des données comprend :
- Visualisation de la distribution des caractéristiques.
- Visualisation des relations entre les caractéristiques.
- Examen des corrélations entre les caractéristiques.

```python
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
```

- `data.hist()` montre les distributions des caractéristiques.
- `pd.plotting.scatter_matrix()` montre les relations entre les caractéristiques.
- `data.corr()` affiche la matrice de corrélation des caractéristiques.

### Étape 4 : Normalisation des données
Les données sont normalisées pour garantir que toutes les caractéristiques sont sur la même échelle.

```python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
```

- `StandardScaler` est utilisé pour normaliser les données.
- `scaled_features` contient les données normalisées.

### Étape 5 : Clustering K-Means
Le code effectue le clustering K-Means avec 3 clusters, correspondant aux trois espèces dans le jeu de données Iris.

```python
# Choisir le nombre de clusters pour K-Means
num_clusters = 3  # Choix du nombre de clusters (correspondant aux trois espèces)

# Initialiser et entraîner l'algorithme K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Ajouter les labels de cluster aux données
data['Cluster'] = kmeans.labels_
```

- `num_clusters = 3` définit le nombre de clusters.
- `KMeans` est utilisé pour initialiser et entraîner l'algorithme K-Means.
- Les étiquettes de clusters sont ajoutées aux données originales dans la colonne `'Cluster'`.

### Étape 6 : Visualisation des clusters
Les clusters sont visualisés en 2D et 3D pour montrer comment les données sont regroupées.

```python
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
```

- `plt.scatter()` visualise les clusters en 2D.
- `fig.add_subplot(111, projection='3d')` crée un graphique 3D pour visualiser les clusters.

### Étape 7 : Analyse des caractéristiques moyennes des clusters
Le code affiche les caractéristiques moyennes des clusters.

```python
print(data.groupby('Cluster').mean())
```

- `data.groupby('Cluster').mean()` calcule les moyennes des caractéristiques pour chaque cluster.

### Étape 8 : Évaluation des résultats de K-Means
Les performances de K-Means sont évaluées à l'aide de différentes métriques.

```python
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
```

- `accuracy_score`, `purity_score`, `adjusted_rand_score`, `silhouette_score` et `classification_report` calculent les métriques de performance.

### Étape 9 : Évaluation par courbe d'inertie avec K-Means
Le code calcule la courbe d'inertie pour évaluer le bon nombre de clusters.

```python
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
```

- `inertia_values` calcule les valeurs d'inertie pour différents nombres de clusters.
- Le graphique montre comment l'inertie évolue avec le nombre de clusters.

### Étape 10 : Visualisation de la distribution des clusters dans chaque classe
Le code visualise la distribution des classes réelles pour chaque cluster.

```python
for cluster in np.unique(data['Cluster']):
    plt.figure(figsize=(8, 6))
    mask = data['Cluster'] == cluster
    plt.hist(true_labels[mask], bins=3, alpha=0.7, label=f'Cluster {cluster}')
    plt.xlabel('Classe réelle')
    plt.ylabel('Nombre')
    plt.title(f'Distribution des classes réelles pour le Cluster {cluster}')
    plt.legend()
    plt.show()
```

- La distribution des classes réelles est visualisée pour chaque cluster.

### Étape 11 : Comparaison avec AgglomerativeClustering
Le code compare les résultats avec K-Means à l'aide de l'algorithme AgglomerativeClustering.

```python
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
```

- `AgglomerativeClustering` est utilisé pour initialiser et entraîner l'algorithme d'agrégation.
- Les étiquettes de clusters sont stockées dans `agglo_labels`.
- Les métriques de performance (pureté, ARI, score de silhouette) sont calculées.

### Étape 12 : Comparaison des résultats entre K-Means et AgglomerativeClustering
Le code compare les résultats des deux algorithmes.

```python
# Comparaison des résultats entre K-Means et AgglomerativeClustering
print(f"Purity de K-Means: {purity:.2f}, Purity d'AgglomerativeClustering: {agglo_purity:.2f}")
print(f"ARI de K-Means: {ari:.2f}, ARI d'AgglomerativeClustering: {agglo_ari:.2f}")
print(f"Score de silhouette de K-Means: {silhouette:.2f}, Score de silhouette d'AgglomerativeClustering: {agglo_silhouette:.2f}")
```

- Les métriques de performance (pureté, ARI, score de silhouette) sont comparées entre K-Means et AgglomerativeClustering.

### Conclusion
Le code conclut que les deux algorithmes sont capables de trouver des clusters significatifs dans le jeu de données Iris, et que le choix entre eux dépend des besoins spécifiques de l'application.

Cette explication étape par étape avec des extraits de code devrait vous aider à comprendre comment le code fonctionne et comment vous pouvez l'utiliser pour effectuer des analyses de clustering sur vos propres jeux de données.
