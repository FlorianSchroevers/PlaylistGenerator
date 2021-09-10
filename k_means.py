import numpy as np
import matplotlib.pyplot as plt

def dist(p1, p2):
    return np.linalg.norm(p1-p2)

def k_means(points, n_clusters, n_iterations=100):
    mean_indices = np.random.choice(points.shape[0], n_clusters, replace=False)
    means = points[mean_indices]

    print(mean_indices, means)

    for i in range(n_iterations):

        clusters = {j: [] for j in mean_indices}

        # associate each point with closest mean
        for pointindex, point in enumerate(points):

            best_cluster = 0
            closest_dist = float('inf')
            for meanindex, mean in zip(mean_indices, means):
                d = dist(mean, point)
                if d < closest_dist:
                    closest_dist = d
                    best_cluster = meanindex
            
            clusters[best_cluster].append(pointindex)
        
        # calculate new means
        mean_indices = []
        for cluster_index, points_indices in clusters.items():
            cluster_points = points[points_indices]
            cluster_mean = np.mean(cluster_points)

            closest_point = None
            closest_dist = float('inf')
            for point_index in points_indices:
                d = dist(cluster_mean, points[point_index])
                if d < closest_dist:
                    closest_dist = d
                    closest_point = point_index
            
            mean_indices.append(closest_point)
        
        means = points[mean_indices]
    
    return clusters


def iterative_clustering(points):
    pass



if __name__ == "__main__":
    points = np.random.rand(1000, 2)
    clusters = k_means(points, 20, 10)

    for j in clusters:
        plt.scatter(points[clusters[j]][:, 0], points[clusters[j]][:, 1])
    plt.show()