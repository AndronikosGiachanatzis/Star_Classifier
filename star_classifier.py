import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# define constants
VARIANCE_THRESHOLD = 0  # the variance threshold below which variables will be deleted
N_CLASSES = 6   # the number of classes
LEARNING_RATE = 0.003   # the learning rate of the optimizer
N_EPOCHS = 300  # number of training epochs


def getCMDArgs():
    parser = argparse.ArgumentParser(description="Trains a neural network to classify different star types",
                                     usage="python star_classifier [OPTION]...")
    parser.add_argument("-v", help="Verbosity")

    args = parser.parse_args()
    return args.file


def loadDataset(name):
    '''
    Load a csv dataset from memory
    :return (pandas.DataFrame): The loaded dataset
    '''

    # load the datast
    print("Loading dataset...")
    dataset = pd.read_csv(name, decimal=",")
    print("[+] Dataset loaded successfully")

    return dataset


def heatMap(dataset):
    '''
    Generates a correlation heatmap for the dataset's variables
    :param dataset: The variables of which the correlations will be found
    '''
    corr_matrix = dataset.corr()
    print(corr_matrix)
    # create the plot and save it
    plt.figure(figsize=(13, 13))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn")

    # save plot
    plt.savefig('corrs.png')
    # plt.show()


def removeLowVariances(x):
    '''
    Removes all variables from the dataset that have variance below a given threshold
    :param x (pandas.DataFrame): The variables whose variance will be analyzed
    :return (pandas.DataFrame): The variables whose variance surpasses the threshold
    '''
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(x)
    new_x = x[x.columns[selector.get_support(indices=True)]]
    return new_x


def scaleX(x_train, x_test=None):
    '''
    Scales the features to fit in the range [0,1]. If passed with a test set then, the test set is scaled using the
    training set.
    :param x_train (pandas.DataFrame): The training x (features)
    :param x_test (pandas.DataFrame): (optional) The test x (features)
    :return (pandas.DataFrame(s)): The scaled dataset(s)
    '''
    # define the scaler
    scaler = MinMaxScaler().fit(x_train)
    # get the column names from the dataset
    x_cols = x_train.columns

    # scale the training set
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_cols)

    # scale the dataset if passed as an argument
    if x_test is not None:
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_cols)
        return x_train_scaled, x_test_scaled

    return x_train_scaled


def standardizeX(x_train, x_test=None):

    standardizer = StandardScaler().fit(x_train)

    x_cols = x_train.columns

    x_train_std = pd.DataFrame(standardizer.transform(x_train), columns=x_cols)

    if x_test is not None:
        x_test_std = pd.DataFrame(standardizer.transform(x_test), columns=x_cols)
        return x_train_std, x_test_std

    return x_train_std


def defineModel(n_features):
    model = keras.models.Sequential()
    # model.add(keras.Input(shape=(n_features, ), name="Input"))
    model.add(keras.layers.InputLayer(input_shape=[n_features]))
    model.add(keras.layers.Dense(10, activation="relu", name="Hidden_1"))
    model.add(keras.layers.Dense(10, activation="relu", name="Hidden_2"))
    model.add(keras.layers.Dense(N_CLASSES, activation="softmax", name="Output"))

    return model


def plotHistory(history):
    plt.close("all")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def main():
    # load dataset
    dataset = loadDataset("dataset/star_dataset.csv")
    print(dataset)
    print()

    # ----- DATA EXPLORATION -----
    print("GENERAL INFORMATION")
    print(dataset.info())
    print()

    print("STATISTICAL DESCRIPTION")
    print(dataset.describe())
    print()

    # check for missing values
    if dataset.isnull().values.any():
        print("The dataset has some missing values")
    else:
        print("The dataset does not have missing values")

    # check data balance
    print(dataset["Star_type"].value_counts())  # data seems to be balanced as there are exactly 40 observations of each class
    print()

    # calculate the correlations between the numerical data
    print("PEARSON CORRELATION COEFFICIENTS")
    heatMap(dataset)

    # ----- DATA PREPROCESSING -----

    # lowercase Star_color column
    dataset["Star_color"] = dataset["Star_color"].str.lower()
    # convert int colum to float
    dataset["Temperature_(K)"] = dataset["Temperature_(K)"].astype(float)


    # One-Hot Encode the categorical features
    color_dummy = pd.get_dummies(dataset["Star_color"], dtype=float)
    spectral_dummy = pd.get_dummies(dataset["Spectral_Class"], dtype=float)

    dataset = dataset.drop(['Star_color', "Spectral_Class"], axis='columns')

    dataset = pd.concat([dataset, color_dummy], axis='columns')
    dataset = pd.concat([dataset, spectral_dummy], axis='columns')

    # split into features (x) and label (y)
    x = dataset.drop("Star_type", axis='columns')

    y = dataset["Star_type"].copy()

    # split train and test set (STRATIFIED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_index, test_index in split.split(dataset, dataset["Star_type"]):
        # train sets
        x_train = x.loc[tr_index]
        y_train = y.loc[tr_index]

        # test sets
        x_test = x.loc[test_index]
        y_test = y.loc[test_index]

    # rest indices
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # scale the features in the range [0,1]
    x_train_scaled, x_test_scaled = scaleX(x_train, x_test)
    x_train_scaled, x_test_scaled = standardizeX(x_train_scaled, x_test_scaled)

    # ----- TRAINING -----

    # define model
    model = defineModel(len(x_train_scaled.columns))

    # compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"]
                  )


    # train the model

    # use early stopping
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

    history = model.fit(x_train_scaled, y_train, epochs=N_EPOCHS, validation_data=(x_test_scaled, y_test),
                        callbacks=[early_stopping_cb], batch_size=12)

    # plot the training statistics
    plotHistory(history)

    print(model.evaluate(x_train_scaled, y_train))
    print(model.evaluate(x_test_scaled, y_test))





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ki_ex:
        print(ki_ex)
        exit(1)