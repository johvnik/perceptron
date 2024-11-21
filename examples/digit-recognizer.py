import pandas as pd
from perceptron import Tensor
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv("~/Documents/Data/digit-recognizer/train.csv")

# Separate the target variable (y) and input features (X)
y = df["label"]  # Target variable as a Series
X = df.drop(columns=["label"])  # Input features as a DataFrame

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
