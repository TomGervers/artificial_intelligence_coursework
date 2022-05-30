# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    # Load Data
    digits = load_digits()
    data = digits.data
    pca = PCA(2)

    # Transform the data to PCA
    data_pca = pca.fit_transform(data)

    # Initialize the KMeans class, random_state set to 0 so it is reproducible
    kmeans = KMeans(n_clusters=10, random_state=0)

    # Predict the labels of the clusters
    label = kmeans.fit_predict(data_pca)

    # Get unique labels
    u_labels = np.unique(label)

    # Find the centroids
    centroids = kmeans.cluster_centers_

    # Plot the clusters on a matplotlib scatter graph:
    for i in u_labels:
        plt.scatter(data_pca[label == i, 0], data_pca[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', marker='x')
    plt.legend()
    plt.title("Scatter graph of K-Means clusters for handwritten digits")
    plt.show()

    # Predict clusters for non-pca data
    clusters = kmeans.fit_predict(digits.data)

    # Create the labels for each cluster 0 to 9
    labels = np.zeros_like(clusters)
    for i in range(10):
        lab = (clusters == i)
        labels[lab] = mode(digits.target[lab])[0]

    # Print out the accuracy score of the test
    print("Accuracy score of: %f" % accuracy_score(digits.target, labels))

    # Create a confusion matrix of the data
    digits_cm = confusion_matrix(digits.target, labels)

    # Use matplotlib and seaborn to make a seaborn heatmap of the data showing
    # Which were commonly incorrect
    plt.figure(figsize=(10, 7))
    sns.heatmap(digits_cm.T, annot=True, fmt='g', cmap='rocket', cbar=True)
    plt.xlabel('Actual digit')
    plt.ylabel('Predicted digit')
    plt.title("Heatmap of answers")
    plt.show()


if __name__ == '__main__':
    main()
