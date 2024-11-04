from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# Funzione che calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito
def elbowMethod(dataSet):
    inertia = []
    maxK = 10
    for i in range(2, maxK):
        kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=42)
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)
    kl = KneeLocator(range(2, maxK), inertia, curve="convex", direction="decreasing")

    # Visualizzo il grafico
    plt.plot(range(2, maxK), inertia, 'bx-')
    plt.scatter(kl.elbow, inertia[kl.elbow - 2], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow method for finding the optimal k')
    plt.legend()
    plt.show()
    return kl.elbow

# Funzione che esegue il KMeans sul dataset
def clusterDataAndVisualize(dataSet):
    # Riduzione della dimensionalità mediante PCA
    pca = PCA(n_components=0.8)
    _dataSet = pca.fit_transform(dataSet)

    k = elbowMethod(_dataSet)
    km = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
    clusters = km.fit_predict(_dataSet)

    labels = km.labels_
    cluster_centers = km.cluster_centers_
    print(f"Number of Clusters: {k}")

    import cuml
    silhouette_avg = cuml.metrics.cluster.silhouette_score(_dataSet, labels)
    print("Silhouette score avg: " + silhouette_avg)

    # Plot dei cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(_dataSet[:, 0], _dataSet[:, 1], c=clusters, cmap='viridis', s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.75)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

    dataSet['clusterIndex'] = labels

    # Plot della distribuzione dei cluster
    dataSet['clusterIndex'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Clusters Distribution')
    plt.show()