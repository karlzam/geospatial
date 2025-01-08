from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import re

fps = pd.read_excel(r'C:\Users\kzammit\Documents\plumes\dfs\clusters-temp.xlsx')

# eps: epsilon, max distance
# min_samples: minimum cluster size

coords = fps[['latitude', 'longitude']].to_numpy()
kms_per_radian = 6371.0088
epsilon = 2/kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

print('Number of clusters: {}'.format(num_clusters))

for idx, row in enumerate(clusters):

    if len(row)>0:

        lats, lons = zip(*row)

        # Create a new figure
        plt.figure(figsize=(6, 6))

        # Plot the data
        #plt.plot(lons, lats, marker='o', label=f'Row {idx}')
        plt.scatter(lons, lats)

        # Add labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Plot for Row {idx}')
        plt.legend()

        plt.savefig(r'C:\Users\kzammit\Documents\plumes\plots\dbscan-plots' + '\\' + str(idx) + '.png')

## Copying in my other code because I can't read my excel files lol


