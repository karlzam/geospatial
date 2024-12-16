import matplotlib.pyplot as plt
import rioxarray


dob_tiff = rioxarray.open_rasterio(r'C:\Users\kzammit\Documents\CFSDS\CFSDS_example_Nov2023\firearrival_decimal_krig.tif', masked=True)

fig, ax1 = plt.subplots(1, 1, figsize=(20, 10), sharex=True, sharey=True)
dob_tiff.plot(cmap='viridis', ax=ax1)

plt.savefig(r'C:\Users\kzammit\Documents\CFSDS\CFSDS_example_Nov2023\tiff-plot.png')