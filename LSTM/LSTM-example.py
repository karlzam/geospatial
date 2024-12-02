# Trying this tutorial now: https://keras.io/examples/generative/lstm_character_level_text_generation/
# Try this next: https://keras.io/examples/nlp/lstm_seq2seq/

# This tutorial uses a LSTM model to generate text character-by-character

import tensorflow as tf
import numpy as np
import random
import io

# Download nietzche.txt from amazon
path = tf.keras.utils.get_file(
    "nietzsche.txt",
    origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
)

# Read the text replacing "\n" with spaces
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")

print("Corpus length:", len(text))

# Get list of all possible characters from the corpus
chars = sorted(list(set(text)))
print("Total chars:", len(chars))

# get character indices, and equivalent indices for where that character is
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# This makes sentences of max length 40 characters skipping 3 characters at the end of each (?)
maxlen = 40
step = 3
sentences = []
next_chars = []
# from 0 to the maximum number of characters - the length of sentence,
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

# Set all to "False", x: (200285, 40, 56)
# x[0].shape = (40,56)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype="bool")

# y[0].shape = (56,)
y = np.zeros((len(sentences), len(chars)), dtype="bool")

# For each sentence
for i, sentence in enumerate(sentences):
    # for all characters (using t as index, or "time" in sequence)
    for t, char in enumerate(sentence):
        # set the current character index to "True"
        x[i, t, char_indices[char]] = 1
    # Set the next character equal to true in the target variable
    y[i, char_indices[next_chars[i]]] = 1

# Build model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(maxlen, len(chars))),
        # This means LSTM with 128 UNITS, I believe a "unit" means the number of previous inputs to consider?
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(chars), activation="softmax"),
    ]
)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Temperature modifies the probability distribution output by a model, to control the diversity of
# generated outputs
# T = 1: no effect
# T < 1: probabilities skewed towards most likely outcome, reducing randomness
# T > 1 : probabilities become more uniform, increasing randomness and encouraging exploration of less likely predictions
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    # Sample the predictions distribution by taking the log and dividing by the temp
    # chatgpt says that this is done as at "linearizes the multiplicative relationships in probabilities", if
    # <1 sharpens the distribution making the higher probabilities more pronounced, if >1 it smoothens making the
    # predictions more uniform
    preds = np.log(preds) / temperature
    # convert the adjusted logits back to positive values (probability-like scores)
    exp_preds = np.exp(preds)
    # normalize the values so they sum to one, forming a valid probability distribution
    preds = exp_preds / np.sum(exp_preds)
    # simulates one draw from the probability distribution, and results in a one-hot encoded vector where one index is 1
    # (the selected class) and all others are 0
    probas = np.random.multinomial(1, preds, 1)
    # return the most likely character (?)
    return np.argmax(probas)

epochs = 40
batch_size = 128

for epoch in range(epochs):
    # Fit the model to the data for this epoch
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    # Get a random index to start sampling from
    start_index = random.randint(0, len(text) - maxlen - 1)

    # Sample the predictions at different temperatures
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""

        # Get a random sentence from the text, starting at the start index defined above
        sentence = text[start_index : start_index + maxlen]
        print('...Generating with seed: "' + sentence + '"')

        # Get the next 400 most probable characters
        for i in range(400):
            # set up a np array full of zeros the correct size
            x_pred = np.zeros((1, maxlen, len(chars)))
            # set the character index to true for each character within the sentence creating a time element
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            # use the model to get the prediction distribution
            preds = model.predict(x_pred, verbose=0)[0]
            # sample the distribution with the stated diversity
            next_index = sample(preds, diversity)
            # get the letter corresponding to that index
            next_char = indices_char[next_index]
            # updates the sentence to remove the first character and append the newly generated character
            # this ensures the next input will use the latest context
            sentence = sentence[1:] + next_char
            # add that character to the generated
            generated += next_char

        print("...Generated: ", generated)
        print("-")

