import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run():
    # Load the data
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")

    # Define categorical and numerical columns
    categorical = ['Sex', 'Embarked']
    numerical = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    # Drop unnecessary columns
    train.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    test_ids = test['PassengerId']
    test.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    # Seperate the target column from features
    y_train = train.Survived
    X_train = train.drop(columns=['Survived'], axis=1)

    # Impute missing values in numerical columns
    imputer = SimpleImputer(strategy='mean')
    X_train[numerical] = imputer.fit_transform(X_train[numerical])
    test[numerical] = imputer.transform(test[numerical])

    # Impute missing values in categorical columns
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical] = imputer.fit_transform(X_train[categorical])
    test[categorical] = imputer.transform(test[categorical])

    # Scale numerical columns
    scaler = StandardScaler()
    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    test[numerical] = scaler.transform(test[numerical])

    # one-hot encode the categorical columns
    ohe = OneHotEncoder()
    full_data = pd.concat(
        [X_train[categorical], test[categorical]],
        axis=0
    )
    ohe.fit(full_data)

    # transform the traning data
    X_train_cat = ohe.transform(X_train[categorical]).toarray()
    test_cat = ohe.transform(test[categorical]).toarray()

    # Create the final dataset
    X_train = np.hstack((X_train[numerical], X_train_cat))
    test = np.hstack((test[numerical], test_cat))

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test data
    preds = model.predict(test)

    # Prepare the submission file
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})

    # Save the submission file
    submission.to_csv(f"./output/submission.csv", index=False)
    print("Submission saved successfully!")

if __name__ == "__main__":
    run()
