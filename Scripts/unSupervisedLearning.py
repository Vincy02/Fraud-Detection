from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Funzione che calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito
def regolaGomito(dataSet):
    inertia = []
    maxK = 10
    for i in range(2, maxK):
        # eseguo il kmeans per ogni k, con 10 inizializzazioni diverse e con inizializzazione k-means++
        kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=42)
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)
    # mediante la libreria kneed trovo il k ottimale
    kl = KneeLocator(range(2, maxK), inertia, curve="convex", direction="decreasing")

    # Visualizza il grafico con la nota per il miglior k
    plt.plot(range(2, maxK), inertia, 'bx-')
    plt.scatter(kl.elbow, inertia[kl.elbow - 2], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.show()
    return kl.elbow

# Funzione che esegue il KMeans sul dataset e restituisce le etichette, centroidi e Silhouette Score
def calcolaCluster(dataSet):
    k = regolaGomito(dataSet)
    km = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
    km = km.fit(dataSet)
    etichette = km.labels_
    centroidi = km.cluster_centers_
    print(f"Numero di Cluster: {k}")

    # Calcolo del Silhouette Score per validare il clustering
    silhouette_avg = silhouette_score(dataSet, etichette)
    print(f"Silhouette Score: {silhouette_avg}")

    return etichette, centroidi, silhouette_avg

# Funzione per visualizzare i cluster (se i dati sono 2D)
def visualizzaCluster(dataSet, etichette, centroidi):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=etichette, s=50, cmap='viridis')
    plt.scatter(centroidi[:, 0], centroidi[:, 1], c='red', s=200, alpha=0.75, label='Centroidi')
    plt.title('Cluster dei dati e centroidi')
    plt.legend()
    plt.show()
