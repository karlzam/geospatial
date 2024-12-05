import pandas as pd
import numpy as np
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

tf.random.set_seed(42)

# installed the above plus conda install openpyxl

### Define the model

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf_keras.Sequential( # KZ: Updated from keras
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf_keras.Sequential( # KZ: Updated from keras
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_model_inputs():
    inputs = {}
    for feature_name in feature_names:
        inputs[feature_name] = tf_keras.Input(name=feature_name, shape=(1,), dtype=tf.float32)
    return inputs


# CHANGED FROM REGRESSION TO CLASSIFICATION TASK
def create_probablistic_bnn_model(train_size):

    # For each column, it creates a keras tensor like this:
    # <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'acq_time')>
    inputs = create_model_inputs()

    # Create one keras tensor for all inputs
    features = tf_keras.layers.concatenate(list(inputs.values())) # KZ: updated from keras
    features = tf_keras.layers.BatchNormalization()(features) # KZ: Updated from layers to tf_keras.layers
    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    hidden_units = [8,8]

    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="relu", # KZ changed from sigmoid
        )(features)

    # Create a probabilistic output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = tf_keras.layers.Dense(units=2)(features) # KZ: Updated from layers.Dense to tf_keras.layers
    #outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    # Sigmoid is appropriate for binary classification, but would be replaced with a softmax for multi-class
    outputs = tf_keras.layers.Dense(1, activation="sigmoid")(features) # Changed to sigmoid activation for binary probs

    model = tf_keras.Model(inputs=inputs, outputs=outputs) # KZ: Updated from keras.Model to tf_keras.Model
    return model


def run_experiment(model, loss, metrics, X_train, X_test, y_train, y_test):
    # Compile the model
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    X_train_format = {str(feature_names[i]): X_train[:, i] for i in range(X_train.shape[1])}
    X_test_format = {str(feature_names[i]): X_test[:, i] for i in range(X_test.shape[1])}

    # Train the model
    history = model.fit(
        x=X_train_format,
        y=y_train,
        batch_size=32,
        epochs=60,
        #validation_data=(X_test_format, y_test)
        validation_split = 0.2
    )

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    return model, train_loss, val_loss


if __name__ == "__main__":

    proj_dir = r'C:\Users\kzammit\Documents\DL-chapter'

    # 2023 data for training
    train_data_file = proj_dir + '\\' + 'train.xlsx'

    # 2022 data for unbiased validation
    val_data_file = proj_dir + '\\' + 'val.xlsx'

    ### Set up data
    train_data = pd.read_excel(train_data_file)

    # Drop the target variable
    X = train_data.drop(columns=['Class'])

    # Set the target variable
    y = train_data['Class']

    # Create the training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_names = X_train.columns

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Normalize the data
    # We fit the scaler to the X training data, and apply this fitted normalization function to the rest of the data
    # The target variable does not need to be normalized
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #train_size = X_train_scaled.shape[0]
    train_size = X_train.shape[0]

    model = create_probablistic_bnn_model(train_size)

    #model, train_loss, val_loss = run_experiment(model, 'binary_crossentropy', ['accuracy'],
    #                                             X_train_scaled, X_test_scaled, y_train, y_test)

    model, train_loss, val_loss = run_experiment(model, 'binary_crossentropy', ['accuracy'],
                                                 X_train, X_test, y_train, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(proj_dir + '\\' + 'loss-curve.png')

    data_val = pd.read_excel(val_data_file)

    # Sample some 0's and some 1's
    #data_val_1 = data_val[0:10]
    #data_val_2 = data_val.tail(10)
    #data_val_f = pd.concat([data_val_1, data_val_2])

    X_val = data_val.drop(columns=['Class'])
    y_val = data_val['Class']

    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    #examples = {str(feature_names[i]): X_val[:, i] for i in range(X_val.shape[1])}
    examples = {str(feature_names[i]): X_test[:, i] for i in range(X_test.shape[1])}
    predictions = model(examples)
    predicted_classes = (predictions.numpy() >= 0.5).astype(int)
    #print(predicted_classes)
    #print(y_val)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cm = confusion_matrix(y_test, predicted_classes)
    sns.heatmap(ax=ax, data=cm, annot=True, fmt='g')
    plt.title('BNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predicted_classes)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(proj_dir + '\\' + 'conf-mat.png')

    print('test')



