import pandas as pd
from train import train_model

# Load training data
train_df = pd.read_csv("data/train.csv")

# Apply same preprocessing used during training
train_df = train_df.drop("Cabin", axis=1)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df = train_df.drop(["Name", "Ticket", "PassengerId"], axis=1)

# Train model
model, _, _ = train_model(train_df)

# Extract feature columns used during training
X = train_df.drop("Survived", axis=1)

# Load test data
test_df = pd.read_csv("data/test.csv")
passenger_ids = test_df["PassengerId"]

# Apply same preprocessing as training data
test_df = test_df.drop("Cabin", axis=1)
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df = pd.get_dummies(test_df, columns=["Embarked"])

test_df = test_df.drop(["Name", "Ticket", "PassengerId"], axis=1)

# Ensure test data matches training feature structure
test_df = test_df.reindex(columns=X.columns, fill_value=0)

# Generate predictions
predictions = model.predict(test_df)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

print("done")