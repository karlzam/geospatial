# Question two: can we classify VIIRS hotspots as T or F positive using a BNN?
# " we present a method for training a probabilistic BNN to perform classification on the vectorized fire data into
# the classes: true or false positive"

# Step One: Pull all VIIRS hotspots for Canada for a specific day
# Step Two: Load NBAC, NFDB, and persistent hot spots
# Step Three: Buffer these by 375*sqrt(2)
# Step Four: Flag as outside or inside a known boundary and set this is as "0" and "1" for TP/FP
# Step Five: Apply to hotspots from a different day and classify as TP/FP


