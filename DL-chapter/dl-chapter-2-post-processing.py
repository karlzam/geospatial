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


def prior(kernel_size, bias_size, dtype=None):
    """
    Define the prior distribution (not tunable)
    :param kernel_size: defined by model
    :param bias_size: defined by model
    :param dtype: defined by model
    :return: prior model
    """

    n = kernel_size + bias_size
    prior_model = tf_keras.Sequential(
      [
          tfp.layers.DistributionLambda(
              lambda t: tfp.distributions.MultivariateNormalDiag(
                  loc=tf.zeros(n), scale_diag=tf.ones(n)
              )
          )
      ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    """
    Define the posterior distribution (tunable)
    :param kernel_size: defined by model
    :param bias_size:
    :param dtype:
    :return: posterior model
    """

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


def create_model_inputs(feature_names):
  """
  Create inputs of tensorflow float type 32 for model input
  :return: inputs
  """
  inputs = {}
  for feature_name in feature_names:
      inputs[feature_name] = tf_keras.Input(name=feature_name, shape=(1,), dtype=tf.float32)
  return inputs


# @title
def create_bnn_model(train_size, prior_func, feature_names, activation_fun = "relu", unit_dim = 8):
    """
    Create bayesian neural network model
    :param feature_names:
    :param train_size: number of samples in training dataset
    :param prior_func: name of prior function (posterior function never changes in this example)
    :param activation_fun: activation function for hidden layers
    :param unit_dim: number of units in hidden layers
    :return: model
    """

    inputs = create_model_inputs(feature_names)

    # Create one keras tensor for all inputs
    features = tf_keras.layers.concatenate(list(inputs.values()))
    features = tf_keras.layers.BatchNormalization()(features)
    hidden_units = [unit_dim,unit_dim]

    for units in hidden_units:
      features = tfp.layers.DenseVariational(
          units=units,
          make_prior_fn=prior_func,
          make_posterior_fn=posterior,
          kl_weight=1 / train_size,
          activation=activation_fun,
      )(features)

    # this may be the summary layer units =2 fr the 2 classes
    distribution_params = tf_keras.layers.Dense(units=2)(features)

    outputs = tf_keras.layers.Dense(1, activation="sigmoid")(features)

    model = tf_keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_tr_val_data(data_file):
  """
  Split data into training and validation datasets
  :param data_file: file path to csv file, ex. 'train-raw.csv' if you've uploaded locally
  :return: X train, X val, y train, y val, feature names (used in model creation)
  """

  # Define raw data file and read
  train_data = pd.read_csv(data_file)

  # Drop the target variable
  X = train_data.drop(columns=['Class'])

  # Set the target variable
  y = train_data['Class']

  # Create the training and testing datasets
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  feature_names = X_train.columns

  X_train = X_train.to_numpy()
  X_val = X_val.to_numpy()
  y_train = y_train.to_numpy()
  y_val = y_val.to_numpy()

  return X_train, X_val, y_train, y_val, feature_names


def run_experiment(model, loss, metrics, epochs, feature_names, X_train, y_train):
  """
  Run the experiment
  :param model: defined model to train
  :param loss: loss function to use during training
  :param metrics: metrics to output during training
  :param epochs: number of epochs
  :param X_train: training data
  :param X_val: validation data
  :param y_train: training data labels
  :param y_val: validation data labels
  :return: model
  """

  # Compile the model
  model.compile(optimizer='adam', loss=loss, metrics=metrics)

  X_train_format = {str(feature_names[i]): X_train[:, i] for i in range(X_train.shape[1])}

  # Train the model
  history = model.fit(
      x=X_train_format,
      y=y_train,
      batch_size=32,
      epochs=epochs,
      validation_split = 0.2
  )

  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  return model, train_loss, val_loss


def compute_predictions(model, examples, iterations=100):
  """
  Computes predictions from a trained model based on
  :param model: trained model
  :param iterations: number of predictions to compute
  :return: None
  """

  predicted = []
  for _ in range(iterations):
      predicted.append(model(examples).numpy())
  predicted = np.concatenate(predicted, axis=1)

  prediction_mean = np.mean(predicted, axis=1).tolist()
  prediction_min = np.min(predicted, axis=1).tolist()
  prediction_max = np.max(predicted, axis=1).tolist()
  prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

  for idx in range(10):
      print(
          f"Predictions mean: {round(prediction_mean[idx], 2)}, "
          f"min: {round(prediction_min[idx], 2)}, "
          f"max: {round(prediction_max[idx], 2)}, "
          f"range: {round(prediction_range[idx], 2)} - "
          f"Actual: {y_val[idx]}"
      )


if __name__ == "__main__":

    # Create X and y train and test data arrays
    # feature_names becomes a global variable that is used in model creation
    X_train_raw, X_val_raw, y_train_raw, y_val_raw, feature_names_raw = create_tr_val_data(r'C:\Users\kzammit\Documents\DL-chapter\train-raw.csv')

    train_size = X_train_raw.shape[0]

    model_raw = create_bnn_model(train_size, prior, feature_names_raw, activation_fun='relu', unit_dim=8)

    # input order: model, loss, metrics, epochs, feature_names, X_train, y_train
    model_raw, train_loss_raw, val_loss_raw = run_experiment(model_raw, 'binary_crossentropy',
                                                             [['accuracy', tf_keras.metrics.Precision(),
                                                               tf_keras.metrics.Recall()]], 80,
                                                             feature_names_raw, X_train_raw, y_train_raw)

    # Create X and y train and test data arrays
    X_train, X_val, y_train, y_val, feature_names = create_tr_val_data(r'C:\Users\kzammit\Documents\DL-chapter\train-preprocessed.csv')

    train_size = X_train.shape[0]

    model_pre = create_bnn_model(train_size, prior, feature_names, activation_fun='relu', unit_dim=8)

    # input order: model, loss, metrics, epochs, X_train, X_val, y_train, y_val
    model_pre, train_loss_pre, val_loss_pre = run_experiment(model_pre, 'binary_crossentropy',
                                                             [['accuracy', tf_keras.metrics.Precision(),
                                                               tf_keras.metrics.Recall()]], 80,
                                                             feature_names, X_train, y_train)

    def compute_metrics(labels, scores, threshold=0.5):
        # Compute the positive scores above threshold, 1 if it is above threshold, 0 if it is not
        predictions = np.where(scores >= threshold, 1, 0)

        # TP: Does the annotated label match the prediction above threshold? Bc "scores" is defined as the positive threshold, this represents TP
        TP = tf.math.count_nonzero(predictions * labels).numpy()

        # TN: Negative score is "predictions - 1" bc predictions was for the positive result, labels-1 so that the negatives are multiplied by 1
        TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()

        # And so on
        FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
        FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

        return predictions, TP, TN, FP, FN

    def plot_results(feature_names, model, X_val, y_val, train_loss, val_loss, title):

        predictions = {str(feature_names[i]): X_val[:, i] for i in range(X_val.shape[1])}
        predictions = model(predictions)
        predicted_classes = (predictions.numpy() >= 0.5).astype(int)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.plot(train_loss, label='Training Loss', marker='o')
        ax1.plot(val_loss, label='Validation Loss', marker='o')
        ax1.set_title('Training and Validation Loss Over Epochs')
        ax1.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax1.grid()
        ax1.legend()

        cm = confusion_matrix(y_val, predicted_classes)
        sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='inferno')
        ax2.set_title('BNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_val, predicted_classes)))
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        fig.suptitle(title)
        plt.savefig(r'C:\Users\kzammit\Documents\DL-chapter\test.png')

    #plot_results(model_raw, X_val_raw, y_val_raw, train_loss_raw, val_loss_raw)
    #plot_results(feature_names, model_pre, X_val, y_val, train_loss_pre, val_loss_pre, title='Preprocessed')

    def compute_results(feature_names, model, X_val, y_val, title):
        predictions = {str(feature_names[i]): X_val[:, i] for i in range(X_val.shape[1])}
        predictions = model(predictions)
        predicted_classes = (predictions.numpy() >= 0.5).astype(int)

        # create precision/recall curve plots
        thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

        classifications_csv = "classifications.csv"
        stats_csv = "stats.csv"

        df_groups = pd.DataFrame()
        #df_stats = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'FPP'])
        df_stats = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1'])
        for thresh in thresholds:
            predicted, TP, TN, FP, FN = compute_metrics(y_val, predictions, thresh)
            df_group = pd.DataFrame()

            df_group['label'] = y_val[:]
            df_group['predicted'] = predicted[:]
            df_group['score'] = predictions[:]
            df_group['threshold'] = thresh

            # Calculate performance metrics
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            FPP = FP / (TN + FP)

            #stats = [thresh, precision, recall, f1, FPP]
            stats = [thresh, precision, recall, f1]
            df_groups = pd.concat([df_groups, df_group])
            df_stats.loc[len(df_stats)] = stats

        df_groups.to_csv(r'C:\Users\kzammit\Documents\DL-chapter' + '\\' + classifications_csv, index=False)
        df_stats.to_csv(r'C:\Users\kzammit\Documents\DL-chapter' + '\\' + stats_csv, index=False)

        sns.lineplot(x='threshold', y='value', hue='variable', data=pd.melt(df_stats, ['threshold']))
        plt.title(title)
        plt.savefig(r'C:\Users\kzammit\Documents\DL-chapter\recall-curve.png')

        return df_groups, df_stats

    classes, stats = compute_results(feature_names, model_pre, X_val, y_val, title='Preprocessed')

    print('test')



    #fig, (ax) = plt.subplots(1, 1, figsize=(5, 3))
    #cm = confusion_matrix(y_val, predicted_classes)
    #sns.heatmap(cm / np.sum(cm), annot=True,
    #            fmt='.2%', cmap='Blues')
    # sns.heatmap(ax=ax, data=cm, annot=True, fmt='g')
    #plt.title('BNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_val, predicted_classes)))
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.show()

    #compute_predictions(model, examples)

    print('test')
