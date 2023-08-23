import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv("D:\Programming\AI\loan\Loan1.csv")

# Extract features and target
X = data[['Annual Salary']]
y = data['Bank Balance']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
joblib.dump(model, "trained_model.pkl")
