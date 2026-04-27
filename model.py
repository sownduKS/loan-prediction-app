import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Create a simple dataset (Simulating real loan data)
data = {
    'Income': [5000, 2000, 10000, 3000, 8000, 1500],
    'LoanAmount': [200, 500, 100, 300, 150, 600],
    'Credit_History': [1, 0, 1, 0, 1, 0], # 1 = Good, 0 = Bad
    'Approved': [1, 0, 1, 0, 1, 0]        # 1 = Yes, 0 = No
}

df = pd.DataFrame(data)

# 2. Separate Features (Input) and Target (Output)
X = df.drop('Approved', axis=1) # The inputs: Income, Loan, Credit
y = df['Approved']               # The output: Approved or not

# 3. Choose the Algorithm and Train
model = RandomForestClassifier()
model.fit(X, y)

# 4. Save the "Brain" to a file
joblib.dump(model, 'loan_model.pkl')

print("Success! Your AI model is trained and saved as 'loan_model.pkl'")