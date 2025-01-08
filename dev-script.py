import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

db_pts = pd.read_excel(r'C:\Users\kzammit\Documents\plumes\dfs\clusters-dbscan.xlsx')
db_pts = db_pts.rename(columns={0:"Clusters"})

kz_pts = pd.read_excel(r'C:\Users\kzammit\Documents\plumes\dfs\all-false-positives.xlsx')




print('test')