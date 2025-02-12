"""

connect-fp-classifications.py

This script connects fp to manual classifications completed after running get-fp-hotspots

Author: Karlee Zammal the Party Mammal
Contact: karlee.zammit@nrcan-rncan.gc.ca
Date: 2025-01-25

"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
import ast
import seaborn as sns

df_folder = r'C:\Users\kzammit\Documents\plumes\dfs'

fp_file = pd.read_csv(df_folder + '\\' + 'fp-2023-09-23.csv')
fp_file = fp_file.loc[:, : 'sat']

class_file = pd.read_excel(df_folder + '\\' + 'all-false-positives-manual-2023-09-23.xlsx')

# assign the ba class to the false positive dataframe
class_file['orig_index'] = class_file['orig_index'].apply(ast.literal_eval)

# Loop over each row in class_file to assign 'In_BA' values to the corresponding indices in fp_file
for idx, row in class_file.iterrows():
    indices = row['orig_index']  # List of indices
    in_ba_value = row['In BA?']  # Value from 'In_BA' column
    img_value = row['VIIRS Smoke?']

    # Assign the value from 'In_BA' to the corresponding indices in fp_file
    fp_file.loc[indices, 'BA'] = in_ba_value
    fp_file.loc[indices, 'Img'] = img_value


TP = fp_file[fp_file['BA']=='Y']
FP = fp_file[fp_file['BA']=='N']
FP_smoke = FP[FP['Img'].isin(['ST1', 'ST2'])]
FP_not_smoke = FP[~FP['Img'].isin(['ST1', 'ST2'])]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, sharey=True)
sns.histplot(ax=ax1, data=TP, x='frp', hue='daynight', stat='density')
ax1.set_title('TP')
sns.histplot(ax=ax2, data=FP, x='frp', hue='daynight',  stat='density')
ax2.set_title('FP-all')
sns.histplot(ax=ax3, data=FP_smoke, x='frp', hue='daynight',  stat='density')
ax3.set_title('FP-smoke')
sns.histplot(ax=ax4, data=FP_not_smoke, x='frp', hue='daynight',  stat='density')
ax4.set_title('FP-not-smoke')
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
plt.tight_layout()

plt.savefig('test.png')

