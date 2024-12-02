#
# Author: Karlee Zammit

###################################################################################################
# Description
###################################################################################################

"""
Following this tutorial and checking that tensorflow/package versions are compatible:
https://pieriantraining.com/tensorflow-lstm-example-a-beginners-guide/
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import csv
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
#print(tf.config.list_physical_devices('GPU'))

# Downloads to C:\Users\kzammit\AppData\Roaming\nltk_data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Downloaded alice text from https://github.com/ElizaLo/Machine-Learning/blob/master/Text%20Generator/alice_in_wonderland.txt
# Load text file
with open(r'C:\Users\kzammit\Documents\LSTM-example\alice.txt', 'r') as f:
    text = f.read()

# Convert to list of sentences using "sent_tokenize()" function
sentences = sent_tokenize(text)

# Remove stop words and convert all to lowercase
stop_words = set(stopwords.words('english'))

preprocessed_sentences = []

for sentence in sentences:
    words = sentence.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    preprocessed_sentences.append(filtered_words)

print('test')

# From here: https://stackoverflow.com/questions/2084069/create-a-csv-file-with-values-from-a-python-list
with open(r'C:\Users\kzammit\Documents\LSTM-example\dataset.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(preprocessed_sentences)

# Ugh this person just loads in a random text file called dataset.csv and doesn't explain how he created it
#data = pd.read_csv(r'C:\Users\kzammit\Documents\LSTM-example\dataset.csv')



print('test')

