import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run(fold):
    # Load the data
    df = pd.read_csv("./input/train_folds.csv")

    # Define categorical and numerical columns
    categorical = ['Sex', 'Embarked']
    numerical = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    # Get the training and validation data using folds
    X_train = df[df.kfold != fold].reset_index(drop=True)
    X_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the target column from the training and validation data
    y_train = X_train.Survived
    y_valid = X_valid.Survived
    X_train = X_train.drop(columns=['Survived'], axis=1)
    X_valid = X_valid.drop(columns=['Survived'], axis=1)

    # Impute missing values in numerical columns
    imputer = SimpleImputer(strategy='mean')
    X_train[numerical] = imputer.fit_transform(X_train[numerical])
    X_valid[numerical] = imputer.transform(X_valid[numerical])

    # Impute missing values in categorical columns
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical] = imputer.fit_transform(X_train[categorical])
    X_valid[categorical] = imputer.transform(X_valid[categorical])

    # Scale numerical columns
    scaler = StandardScaler()
    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    X_valid[numerical] = scaler.transform(X_valid[numerical])

    # one-hot encode the categorical columns
    ohe = OneHotEncoder()
    full_data = pd.concat(
        [X_train[categorical], X_valid[categorical]],
        axis=0
    )
    ohe.fit(full_data)

    # transform the traning data
    X_train_cat = ohe.transform(X_train[categorical]).toarray()
    X_valid_cat = ohe.transform(X_valid[categorical]).toarray()

    # Create the final dataset
    X_train = np.hstack((X_train[numerical], X_train_cat))
    X_valid = np.hstack((X_valid[numerical], X_valid_cat))

    # Initialize the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict the probabilities
    valid_preds = model.predict(X_valid)

    # Calculate the accuracy
    accuracy = metrics.accuracy_score(y_valid, valid_preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold=fold_)
    