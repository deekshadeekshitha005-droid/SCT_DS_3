import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("bank-full.csv", sep=";")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ---------------- DATA PREPROCESSING ----------------

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("y_yes", axis=1)
y = df["y_yes"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- DECISION TREE MODEL ----------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# ---------------- MODEL EVALUATION ----------------

accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Classifier")
print("Accuracy:", round(accuracy * 100, 2), "%")

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------- DECISION TREE VISUALIZATION ----------------

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.close()

print("\nTask 3 completed successfully!")