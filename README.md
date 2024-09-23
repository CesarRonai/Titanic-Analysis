# Titanic-Analysis
# Titanic Survival Prediction Project

## Overview

This project involves analyzing the Titanic dataset and building a machine learning model to predict the survival of passengers. The dataset contains various features such as passenger class, age, gender, number of siblings/spouses aboard, number of parents/children aboard, and the fare they paid. By exploring these features, we aim to understand what factors contributed to a higher chance of survival during the disaster.

## Objectives

- **Exploratory Data Analysis (EDA):** Analyze the data to discover patterns and insights, especially related to the survival of passengers.
- **Data Preprocessing:** Clean the data, handle missing values, and transform categorical data for use in the machine learning models.
- **Feature Engineering:** Identify important features that influence survival and transform them to optimize the model's performance.
- **Modeling:** Train a classification model to predict survival and evaluate its performance using various metrics.

## Dataset

The dataset used in this project is the classic Titanic dataset, which contains the following columns:
- **PassengerId:** Unique ID for each passenger.
- **Survived:** Target variable, where 0 indicates the passenger did not survive, and 1 indicates they did.
- **Pclass:** Passenger's ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Name, Sex, Age:** Basic demographic information.
- **SibSp:** Number of siblings or spouses aboard.
- **Parch:** Number of parents or children aboard.
- **Ticket, Fare:** Ticket number and the fare paid for the trip.
- **Cabin, Embarked:** Cabin number and port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Tools and Libraries Used

The following Python libraries were used in this project:
- **Pandas:** For data loading, cleaning, and manipulation.
- **Seaborn:** For creating informative visualizations and exploring relationships between variables.
- **Matplotlib:** For plotting graphs.
- **Scikit-learn:** For machine learning model building, evaluation, and preprocessing (logistic regression, data splitting, scaling, etc.).
- **Joblib:** For saving and loading the trained machine learning models.

## Project Workflow

1. **Exploratory Data Analysis (EDA):**
   - Visualize survival rates across different passenger groups (e.g., by gender, class, age).
   - Investigate correlations between features and survival using heatmaps and pair plots.
   - Explore the distribution of key variables like age, fare, and class.

2. **Data Preprocessing:**
   - Handle missing values in the `Age`, `Cabin`, and `Embarked` columns.
   - Convert categorical features (`Sex`, `Embarked`) into numerical values using one-hot encoding.
   - Standardize numerical features like `Age` and `Fare` for better model performance.

3. **Model Building and Evaluation:**
   - Split the data into training and testing sets using `train_test_split`.
   - Train a logistic regression model and evaluate its performance.
   - Calculate accuracy, confusion matrix, precision, recall, F1-score, and AUC-ROC curve to assess the model's effectiveness.

4. **Saving the Model:**
   - Save the trained model using `joblib` for future use.

## How to Run This Project

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   [git clone https://github.com/YOUR_USERNAME/Titanic-Analysis.git](https://github.com/CesarRonai/Titanic-Analysis)
   ```

2. **Install the necessary dependencies:**
   Make sure you have Python installed and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook or Python scripts:**
   You can open and execute the notebook (`ProjecTitanic.ipynb`) or run Python scripts for model training.

## Model Performance

The logistic regression model performed with an accuracy of around 80% based on the test data. Here are some key evaluation metrics:

- **Accuracy:** The overall correctness of the model's predictions.
- **Confusion Matrix:** To see the true positives, true negatives, false positives, and false negatives.
- **AUC-ROC Curve:** Measures the trade-off between true positive rate and false positive rate.

## Future Work

- Experiment with more advanced models such as Random Forest, Gradient Boosting, and XGBoost to see if they improve accuracy.
- Perform hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
- Explore additional feature engineering methods to improve model performance (e.g., binning ages or creating interaction terms).
- Deploy the model using Flask or FastAPI to create an API for real-time predictions.

## License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.

## Acknowledgments

Special thanks to the creators of the Titanic dataset and the open-source tools used in this project.
