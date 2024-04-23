# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score, silhouette_score, homogeneity_score, purity_score

# Charger le jeu de données Iris depuis scikit-learn
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target  # Vraies étiquettes des classes

# Aperçu des données
print("Aperçu des données Iris:")
print(data.head())
print("\nDescription statistique des données:")
print(data.describe())

# Analyse exploratoire des données (EDA):
# Visualisation des distributions des caractéristiques individuelles
print("\nVisualisation des distributions des caractéristiques individuelles:")
data.hist(figsize=(12, 8))
plt.show()

# Visualisation des relations entre les caractéristiques
print("\nVisualisation des relations entre les caractéristiques:")
pd.plotting.scatter_matrix(data, figsize=(15, 12), alpha=0.8)
plt.show()

# Visualisation des corrélations entre les caractéristiques
print("\nVisualisation des corrélations entre les caractéristiques:")
correlation_matrix = data.corr()
print(correlation_matrix)
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(data.columns)), data.columns, rotation=90)
plt.yticks(range(len(data.columns)), data.columns)
plt.title('Matrice de corrélation')
plt.show()

# Sélectionner les caractéristiques pour le clustering
features = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Normaliser les données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Tuning des hyperparamètres:
# Essayer différentes valeurs de nombre de clusters et comparer les résultats
num_clusters = 3  # Nombre de clusters à tester
for k in range(2, 6):
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    print(f"\nK-Means avec {k} clusters:")
    print(f"Inertie: {kmeans.inertia_:.2f}")
    silhouette = silhouette_score(scaled_features, kmeans.labels_)
    print(f"Score de silhouette: {silhouette:.2f}")

    # DBSCAN
    dbscan = DBSCAN()
    labels_dbscan = dbscan.fit_predict(scaled_features)
    silhouette_dbscan = silhouette_score(scaled_features, labels_dbscan)
    print(f"\nDBSCAN:")
    print(f"Score de silhouette: {silhouette_dbscan:.2f}")

    # Agglomératif
    agglo = AgglomerativeClustering(n_clusters=k)
    labels_agglo = agglo.fit_predict(scaled_features)
    silhouette_agglo = silhouette_score(scaled_features, labels_agglo)
    print(f"\nAgglomératif avec {k} clusters:")
    print(f"Score de silhouette: {silhouette_agglo:.2f}")

# Initialiser et entraîner l'algorithme K-Means avec le nombre de clusters optimal
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Ajouter les labels de cluster aux données
data['Cluster'] = kmeans.labels_

# Visualisation avancée:
# Visualiser les clusters en 2D
plt.figure(figsize=(10, 8))
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Clusters dans le jeu de données Iris')
plt.show()

# Visualisation 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['sepal length (cm)'], data['sepal width (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
plt.title('Visualisation 3D des clusters dans le jeu de données Iris')
plt.show()

# Interprétation des clusters:
# Analyser les caractéristiques des clusters
print("\nCaractéristiques moyennes des clusters:")
print(data.groupby('Cluster').mean())

# Comparaison avec les vraies classes d'iris
print("\nComparaison avec les vraies classes d'iris:")
for cluster in np.unique(data['Cluster']):
    true_class_counts = data[data['Cluster'] == cluster]['Cluster'].value_counts()
    print(f"Cluster {cluster}:")
    print(true_class_counts)

# Évaluation des résultats
# Calcul de l'exactitude (accuracy)
accuracy = accuracy_score(true_labels, data['Cluster'])
print(f"\nAccuracy: {accuracy:.2f}")

# Calcul de la matrice de confusion
confusion = confusion_matrix(true_labels, data['Cluster'])
print("\nMatrice de confusion :")
print(confusion)

# Calcul de l'Indice de Rand Ajusté (Adjusted Rand Index)
ari = adjusted_rand_score(true_labels, data['Cluster'])
print(f"\nIndice de Rand Ajusté: {ari:.2f}")

# Calcul du score de silhouette
silhouette = silhouette_score(scaled_features, data['Cluster'])
print(f"\nScore de silhouette: {silhouette:.2f}")

# Calcul de la purity et de l'homogeneity
purity = purity_score(true_labels, data['Cluster'])
homogeneity = homogeneity_score(true_labels, data['Cluster'])
print(f"\nPurity: {purity:.2f}")
print(f"Homogeneity: {homogeneity:.2f}")

# Classification report
report = classification_report(true_labels, data['Cluster'])
print("\nRapport de classification :")
print(report)

# Évaluation par courbe d'inertie
inertia = kmeans.inertia_
print(f"\nInertie: {inertia:.2f}")

# Visualisation de la courbe d'inertie
inertia_values = []
for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_features)
    inertia_values.append(kmeans_model.inertia_)
plt.figure()
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Courbe d\'inertie')
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

# Conclusion sur les performances de l'algorithme K-Means
print("\nLes résultats montrent que l'algorithme K-Means peut trouver des clusters significatifs dans le jeu de données Iris.")
print("Toutefois, certaines classes peuvent être confondues avec d'autres, ce qui explique la matrice de confusion et l'Indice de Rand Ajusté.")
print("Le score de silhouette indique que les clusters sont relativement bien séparés et denses.")
print("La purity et l'homogeneity fournissent des mesures supplémentaires de la qualité du clustering.")
