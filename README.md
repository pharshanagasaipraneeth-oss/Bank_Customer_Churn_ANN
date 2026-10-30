# Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## Project Overview

The **Bank Customer Churn Prediction** project uses Artificial Neural Networks (ANN) to predict whether a bank customer is likely to leave the bank or continue using its services.

The project focuses on data preprocessing, exploratory data analysis, building multiple ANN architectures, evaluating their performance, and identifying the best-performing model based on the F1-score.

## Objectives

- Analyze the bank customer churn dataset.
- Perform data preprocessing and feature transformation.
- Build and train four different ANN architectures.
- Evaluate the models using multiple classification metrics.
- Compare the performance of all four architectures.
- Save the trained ANN model and preprocessing pipeline for future predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Google Colab / Jupyter Notebook

## Dataset Description

**Dataset:** Bank Customer Churn Prediction.csv

The dataset contains customer-related information used to predict customer churn.

The target variable is:

- `churn` – Indicates whether a customer has left the bank.

The following columns are removed from the input features:

- `customer_id`
- `credit_score`
- `churn` (used as the target variable)

## Project Workflow

### 1. Data Loading and Exploratory Data Analysis (EDA)

- Loaded the dataset using Pandas.
- Checked the first few records and dataset dimensions.
- Examined column names and data types.
- Checked for missing values and duplicate records.
- Generated descriptive statistical summaries.
- Analyzed the distribution of the target variable (`churn`).

### 2. Data Preprocessing

- Separated the independent features (`X`) and target variable (`y`).
- Split the dataset into training and testing sets using an 80:20 ratio.
- Identified categorical and numerical columns.
- Applied **OneHotEncoder** to categorical features.
- Applied **StandardScaler** to numerical features.
- Used **ColumnTransformer** to combine both preprocessing steps.

### 3. ANN Model Architectures

Four different ANN architectures were implemented and compared.

**Architecture 1: Simple ANN Baseline Model**

- Input Layer
- Dense Layer – 16 neurons, ReLU activation
- Output Layer – 1 neuron, Sigmoid activation

**Architecture 2: Deep Neural Network**

- Input Layer
- Hidden Layer 1 – 64 neurons, ReLU activation
- Hidden Layer 2 – 32 neurons, ReLU activation
- Output Layer – 1 neuron, Sigmoid activation

**Architecture 3: ANN with Batch Normalization and Dropout**

- Input Layer
- Dense Layer – 64 neurons, ReLU activation
- Batch Normalization
- Dropout – 0.3
- Dense Layer – 32 neurons, ReLU activation
- Batch Normalization
- Dropout – 0.2
- Output Layer – 1 neuron, Sigmoid activation

**Architecture 4: Regularized ANN**

- Input Layer
- Dense Layer – 128 neurons, ReLU activation
- Batch Normalization
- Dropout – 0.4
- Dense Layer – 64 neurons, ReLU activation
- Batch Normalization
- Dropout – 0.3
- Dense Layer – 32 neurons, ReLU activation
- Output Layer – 1 neuron, Sigmoid activation

### 4. Model Compilation and Training

All four ANN models were compiled using:

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 10
- Batch Size: 32
- Validation Split: 0.2

### 5. Model Evaluation and Comparison

All four models were evaluated using the test dataset.

The following evaluation metrics were used:

- Test Loss
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

A comparison table and bar chart were generated to analyze the performance of the four ANN architectures.

The best-performing architecture was identified based on the highest F1-score.

### 6. Model Saving

The project saves the trained ANN model in Keras format:

`best_bank_churn_model.keras`

The fitted preprocessing pipeline is saved using Joblib:

`bank_churn_preprocessor.pkl`

These files can be reused to preprocess new customer data and generate churn predictions.

## Project Files

| File | Description |
|---|---|
| `Bank_Customer_Churn_Prediction_project(ANN).ipynb` | Jupyter Notebook containing EDA, preprocessing, ANN architectures, training, and evaluation |
| `best_bank_churn_model.keras` | Saved trained ANN model |
| `bank_churn_preprocessor.pkl` | Saved preprocessing pipeline |

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone <your-github-repository-url>
   ```

2. Navigate to the project folder:

   ```bash
   cd Bank_Customer_Churn_ANN
   ```

3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow keras joblib
   ```

4. Open the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the project notebook and update the dataset path according to your system.

6. Run the notebook cells sequentially to perform EDA, preprocessing, model training, and evaluation.

## Future Enhancements

- Deploy the trained ANN model using Streamlit.
- Create an interactive interface for entering customer details.
- Display customer churn predictions and churn probabilities.
- Improve model performance through hyperparameter tuning.
- Explore additional feature engineering techniques.

## Conclusion

This project demonstrates how Artificial Neural Networks can be applied to a bank customer churn prediction problem. By implementing and comparing four different ANN architectures, the project provides an opportunity to understand neural network design, data preprocessing, model training, and classification evaluation.

The model comparison uses the F1-score to identify the best-performing architecture.

---

**Developed as a Machine Learning / Deep Learning Project using Python and Artificial Neural Networks.**
