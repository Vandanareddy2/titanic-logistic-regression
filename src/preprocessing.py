import pandas as pd

def preprocess_train_data(train_df):
    train_df = train_df.copy()

    train_df = train_df.drop("Cabin", axis=1)

    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

    train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})

    train_df = pd.get_dummies(train_df, columns=["Embarked"])

    train_df = train_df.drop(["Name", "Ticket", "PassengerId"], axis=1)

    return train_df