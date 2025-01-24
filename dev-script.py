import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from networkx import Graph, connected_components
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
import alphashape
import re

file = pd.read_excel(r'C:\Users\kzammit\Documents\plumes\dfs\all-false-positives-2023-09-24.xlsx')

perim_points = [[float(lon), float(lat)] for lon, lat in re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", file['fire-perimeter'][0])]
poly = Polygon(perim_points)
x, y = poly.exterior.xy

fig, ax = plt.subplots(figsize=(10,8))
plt.plot(x, y, color='blue')
plt.fill(x, y, color='lightblue', alpha=0.5)

plt.savefig('test.png')

