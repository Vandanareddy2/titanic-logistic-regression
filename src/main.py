from data_loader import load_data
from preprocessing import preprocess_train_data
from train import train_model
from evaluate import evaluate_model

def main():
    # 1. Load data
    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    # 2. Preprocess training data
    train_df = preprocess_train_data(train_df)

    # 3. Train model
    model, X_test, y_test = train_model(train_df)

    # 4. Evaluate model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)

    # 5. Print results
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)


if __name__ == "__main__":
    main()