# Titanic - Machine Learning from Disaster

![Titanic](https://i.imgur.com/4qhUoh9.jpeg)

## Overview

This project is a solution to the popular Kaggle competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). The goal is to predict the survival of passengers aboard the Titanic based on various features such as age, gender, and passenger class.

The dataset is provided by Kaggle and contains information about the passengers on the Titanic, including whether they survived or not. This project uses Python and popular data science libraries to explore the data, build a predictive model, and submit the results.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/surajwate/Titanic-Machine-Learning-from-Disaster.git
    cd Titanic-Machine-Learning-from-Disaster
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Download the dataset from Kaggle:**
   - Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/c/titanic/data).
   - Place them in the `input/` folder.
  
**Note**: Ensure that you run the Python scripts from the root folder of the repository (`Titanic-Machine-Learning-from-Disaster`). The scripts rely on relative paths to load data from the `input/` folder and save outputs.

## Data

- **train.csv:** Contains the training data with 891 rows and 12 columns.
- **test.csv:** Contains the test data with 418 rows and 11 columns (missing the `Survived` column).

## Exploratory Data Analysis

Exploratory data analysis was performed in the Jupyter notebook `titanic_EDA.ipynb`. The notebook covers:

- Overview of the dataset
- Missing data analysis
- Univariate and bivariate analysis
- Feature engineering and creation of new features

## Modeling

The model used in this project is Logistic Regression. The modeling process involves:

1. **Creating Stratified K-Folds:**
   - The script `create_folds.py` is used to create stratified k-folds for cross-validation.
  
2. **Model Training and Validation:**
   - The script `logreg.py` performs training using cross-validation and evaluates the model performance on each fold.

3. **Final Model Training and Prediction:**
   - The final model was trained on the entire dataset, and predictions for the test set were made.

4. **Additional Notebooks:**
   - **`logistic_model.ipynb:`** This notebook details the steps taken to build the logistic regression model, including data cleaning, normalization, and encoding.
   - **`pipeline.ipynb:`** This notebook explores automating parts of the data processing and model building pipeline, testing different approaches to streamline the workflow.

## Results

The model achieved the following accuracy scores during cross-validation:

```bash
Fold=0, Accuracy=0.7709
Fold=1, Accuracy=0.7865
Fold=2, Accuracy=0.8371
Fold=3, Accuracy=0.7865
Fold=4, Accuracy=0.8202
```

The final submission to Kaggle resulted in a score of `0.76794`.

## Improvements

Potential improvements include:

- Advanced feature engineering (e.g., extracting titles from names, creating interaction terms).
- Trying different models (e.g., Random Forest, Gradient Boosting).
- Hyperparameter tuning using Grid Search or Randomized Search.
- Ensembling multiple models to improve prediction accuracy.

## How to Use

1. **Create Folds:**

    ```bash
    python src/create_folds.py
    ```

2. **Train Model and Evaluate:**

    ```bash
    python src/logreg.py
    ```

3. **Train Final Model and Make Predictions:**
   - Modify `logreg.py` or use the approach outlined in `logistic_model.ipynb` to train on the full dataset and generate predictions.
   - Submit the `submission.csv` file generated to Kaggle.

## Contributing

Contributions are welcome! Please create a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
