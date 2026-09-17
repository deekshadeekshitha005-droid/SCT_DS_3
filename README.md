SCT_DS_3 - Bank Marketing Decision Tree Classifier
📌 Project Overview
This project focuses on building a Decision Tree Classification Model using the Bank Marketing dataset.
The objective is to predict whether a customer will subscribe to a term deposit based on demographic and behavioral information.
🎯 Objective
To develop a machine learning model that predicts customer subscription outcomes and evaluates the model's performance using classification metrics.
📊 Dataset
Dataset: Bank Marketing Dataset
Source: UCI Machine Learning Repository
The dataset contains customer information such as:
Age
Job
Marital Status
Education
Account Balance
Housing Loan
Personal Loan
Contact Information
Previous Campaign Details
Subscription Status
Target Variable: y
yes – Customer subscribed
no – Customer did not subscribe
🛠️ Technologies Used
Python
Pandas – Data processing
Scikit-learn – Machine learning
Matplotlib – Data visualization
VS Code – Development environment
⚙️ Methodology
Loaded the Bank Marketing dataset.
Converted categorical variables into numerical values.
Separated input features and target variable.
Split the dataset into training and testing sets.
Built a Decision Tree Classifier.
Trained the model using training data.
Predicted customer subscription outcomes.
Evaluated the model using accuracy and a confusion matrix.
Visualized the decision tree structure.
🤖 Machine Learning Model
Algorithm: Decision Tree Classifier
The model uses customer information to predict whether a customer is likely to subscribe to a term deposit.
The maximum tree depth was limited to help control model complexity.
📈 Model Evaluation
The project includes:
Accuracy Score – Measures the percentage of correct predictions.
Confusion Matrix – Displays correct and incorrect predictions for each class.
Decision Tree Visualization – Shows the structure of the trained classification model.
🖼️ Visualizations
Confusion Matrix
The confusion matrix presents the model's classification results for customers who subscribed and did not subscribe.
Decision Tree
The decision tree visualization displays the decision-making structure used by the model.
📁 Project Structure
SCT_DS_3/
│
├── bank-full.csv
├── task3.py
├── confusion_matrix.png
├── decision_tree.png
└── README.md

▶️ How to Run
Install Required Libraries
pip install pandas scikit-learn matplotlib

Run the Python Program
python task3.py

The program trains the Decision Tree Classifier, evaluates its performance, and generates visualization files.
🏁 Conclusion
This project demonstrates fundamental machine learning skills, including:
Data preprocessing
Categorical data encoding
Train-test splitting
Decision Tree classification
Model evaluation
Data visualization
The project provides practical experience in applying machine learning to customer subscription prediction.
