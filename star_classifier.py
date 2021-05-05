import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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
    # create the plot and save it
    plt.figure(figsize=(13,13))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn")

    # save plot
    plt.savefig('corrs.png')
    # plt.show()


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
    print(dataset["Star_type"].value_counts())    # data seems to be balanced as there are exactly 40 observations of each class
    print()

    # calculate the correlations between the numerical data
    print("PEARSON CORRELATION COEFFICIENTS")
    heatMap(dataset)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ki_ex:
        print(ki_ex)
        exit(1)