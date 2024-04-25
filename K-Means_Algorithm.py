# 20201202
# Norhan Hassan

import tkinter
import random
from tkinter import filedialog
import numpy
import pandas
from matplotlib import pyplot as plt


class K_Means_Clustering:
    def __init__(self, root):
        self.outliers = None
        self.root = root
        self.root.title("K Means Clustering")
        self.root.configure(bg="#F7F0FF")
        self.data = None

        # GUI elements
        tkinter.Label(root, text="Value of K :").pack()
        self.value_of_k_entry = tkinter.Entry(root)
        self.value_of_k_entry.pack()

        tkinter.Label(root, text="Percentage of Data (%):").pack()
        self.percentage_entry = tkinter.Entry(root)
        self.percentage_entry.pack()

        self.preprocess_button = tkinter.Button(root, text="Choose File", command=self.calling_function)
        self.preprocess_button.pack()

        # Display clusters
        self.cluster_label = tkinter.Label(root, text="Clusters:")
        self.cluster_label.pack()
        self.cluster_text = tkinter.Text(root, height=20, width=80)
        self.cluster_text.pack()

        # Display outliers
        self.outlier_label = tkinter.Label(root, text="Outliers:")
        self.outlier_label.pack()
        self.outlier_text = tkinter.Text(root, height=17, width=80)
        self.outlier_text.pack()

    def calling_function(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        value_of_k_entry = int(self.value_of_k_entry.get())
        percentage_entry = float(self.percentage_entry.get())

        df = pandas.read_csv(file_path)
        num_records = int(len(df) * (percentage_entry / 100))
        df = df.head(num_records)

        # Preprocess data
        self.data = self.preprocess_data(df)
        print(self.data)
        clusters, centroids = self.k_means_clustering(value_of_k_entry)

        outliers = self.detect_outliers(clusters, centroids)

        self.display_outliers(outliers, clusters)
        # self.display_clusters(clusters)
        self.data = self.remove_outliers(clusters, centroids)

        # after removing outliers
        clusters, centroids = self.k_means_clustering(value_of_k_entry)
        self.display_clusters(clusters)

        # Visualize clusters
        # self.visualize_clusters(clusters, centroids)

    def preprocess_data(self, df):

        # Check for missing values
        missing_values = df.isnull().sum()
        print('Missing values:\n', missing_values)
        print('----------------------')

        # Remove unused columns
        columns_to_remove = ['Movie Name', 'Release Year', 'Duration', 'Metascore', 'Votes', 'Genre', 'Director',
                             'Cast', 'Gross']
        data = df.drop(columns=columns_to_remove)

        return data

    def euclidean_distance(self, point1, point2):
        return numpy.sqrt(numpy.sum((point1 - point2) ** 2))

    def initialize_centroids(self, k):
        # Randomly initialize centroids
        centroids_indices = random.sample(range(len(self.data)), k)
        centroids = [self.data.iloc[i].values.tolist() for i in centroids_indices]
        print(centroids)
        return centroids

    def assign_clusters(self, centroids):
        clusters = []
        for point in self.data.values:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
            cluster = numpy.argmin(distances)
            clusters.append(cluster)
        return clusters

    def update_centroids(self, clusters, k):
        centroids = []
        for i in range(k):
            cluster_points = [self.data.iloc[j].values for j in range(len(self.data)) if clusters[j] == i]
            if len(cluster_points) > 0:
                centroid = numpy.mean(cluster_points, axis=0)
                print(centroids)
            else:

                centroid = self.data.sample().values[0]
            centroids.append(centroid)
        return centroids

    def k_means_clustering(self, k, max_iterations=100):
        centroids = self.initialize_centroids(k)
        for _ in range(max_iterations):
            clusters = self.assign_clusters(centroids)
            new_centroids = self.update_centroids(clusters, k)

            if numpy.allclose(numpy.array(centroids, dtype=numpy.float64),
                              numpy.array(new_centroids, dtype=numpy.float64)):
                break

            centroids = new_centroids

        return clusters, centroids

    def detect_outliers(self, clusters, centroids):
        distances = [self.euclidean_distance(self.data.iloc[i].values, centroids[cluster]) for i, cluster in
                     enumerate(clusters)]
        q1 = numpy.percentile(distances, 25)
        q3 = numpy.percentile(distances, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        outliers = [i for i, distance in enumerate(distances) if distance > upper_bound or distance < lower_bound]
        return outliers

    def remove_outliers(self, clusters, centroids):
        outliers = self.detect_outliers(clusters, centroids)
        self.outliers = self.data.iloc[outliers]
        self.data = self.data.drop(outliers).reset_index(drop=True)
        return self.data

    def display_clusters(self, clusters):

        clusters = numpy.array(clusters, dtype=int)

        for cluster_id in range(max(clusters) + 1):
            cluster_data = self.data[clusters == cluster_id]
            print(f"Cluster {cluster_id + 1}:")
            print(cluster_data)
            print(f"Count: {len(cluster_data)}")

        self.cluster_text.delete(1.0, tkinter.END)

        cluster_info = ""
        for cluster_id in range(max(clusters) + 1):
            cluster_data = self.data[clusters == cluster_id]
            cluster_info += f"Cluster {cluster_id + 1}:\n{cluster_data}\n"
            cluster_info += f"Count: {len(cluster_data)}\n\n"

        self.cluster_text.insert(tkinter.END, cluster_info)

    def display_outliers(self, outliers, clusters):

        clusters = numpy.array(clusters, dtype=int)

        outlier_counts = [0] * (max(clusters) + 1)
        upper_bounds = [float("-inf")] * (max(clusters) + 1)
        lower_bounds = [float("inf")] * (max(clusters) + 1)
        cluster_outliers = [[] for _ in range(max(clusters) + 1)]

        for i in outliers:
            cluster_id = clusters[i]
            outlier_counts[cluster_id] += 1
            cluster_outliers[cluster_id].append(i)
            data_point = self.data.iloc[i]
            for j, value in enumerate(data_point):
                if value > upper_bounds[cluster_id]:
                    upper_bounds[cluster_id] = value
                if value < lower_bounds[cluster_id]:
                    lower_bounds[cluster_id] = value

        outlier_info = "Outlier records:\n"
        if outliers:
            for cluster_id in range(len(cluster_outliers)):
                outlier_info += f"Cluster {cluster_id + 1} outliers ({outlier_counts[cluster_id]}):\n"
                if cluster_outliers[cluster_id]:
                    cluster_records = self.data.iloc[cluster_outliers[cluster_id]]
                    outlier_info += str(cluster_records) + "\n"
                outlier_info += f"Upper Bound: {upper_bounds[cluster_id]}\n"
                outlier_info += f"Lower Bound: {lower_bounds[cluster_id]}\n\n"

        self.outlier_text.delete(1.0, tkinter.END)
        #  in GUI
        if outliers:
            self.outlier_text.insert(tkinter.END, outlier_info)
        else:
            self.outlier_text.insert(tkinter.END, "No outliers found.")

    """
    def detect_outliers(self, clusters, centroids, percentile=95):
        distances = [self.euclidean_distance(self.data.iloc[i].values, centroids[cluster]) for i, cluster in enumerate(clusters)]
        threshold = numpy.percentile(distances, percentile)
        outliers = [i for i, distance in enumerate(distances) if distance > threshold]
        return outliers
    """
    """
    def visualize_clusters(self, clusters, centroids):
        # Convert clusters to integer array
        clusters = numpy.array(clusters, dtype=int)

        # Plot the data points with cluster assignments
        plt.figure(figsize=(8, 6))
        for cluster_id in range(max(clusters) + 1):
            cluster_data = self.data[clusters == cluster_id]
            plt.scatter(cluster_data.iloc[:, 0], [cluster_id] * len(cluster_data), label=f'Cluster {cluster_id + 1}')

        # Plot the centroids
        centroids = numpy.array(centroids)
        plt.scatter(centroids[:, 0], [i for i in range(len(centroids))], color='black', marker='x', label='Centroids')

        plt.title('K-Means Clustering')
        plt.xlabel('IMDB Rating')
        plt.ylabel('Cluster')
        plt.legend()
        plt.grid(True)
        plt.show()
    """


if __name__ == "__main__":
    root = tkinter.Tk()
    app = K_Means_Clustering(root)
    root.resizable(True, True)
    root.mainloop()
