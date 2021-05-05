import pandas as pd



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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ki_ex:
        print(ki_ex)
        exit(1)