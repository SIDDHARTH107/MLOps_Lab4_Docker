"""
train_model.py — Trains a Loan Default Prediction Model

WHAT THIS DOES:
1. Creates a fake (synthetic) credit/loan dataset with 1000 rows
2. Trains a Random Forest classifier to predict loan default
3. Saves the trained model + scaler to files so the API can use them later

Think of it like:
- Generating practice exam questions (fake data)
- Studying for the exam (training the model)
- Writing down the answers in a cheat sheet (saving model.pkl)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ============================================================
# STEP 1: Generate Synthetic (Fake) Credit Dataset
# ============================================================
# Why fake data? It's simpler — no need to download anything.
# In the real world, this would come from a database or CSV file.

print("📊 Step 1: Generating synthetic credit dataset...")

np.random.seed(42)  # Makes the random data the same every time you run it
n_samples = 1000    # 1000 fake customers

# Create features (the information we know about each customer)
data = {
    # Annual income between $20,000 and $150,000
    "income": np.random.randint(20000, 150000, n_samples),

    # Loan amount between $1,000 and $50,000
    "loan_amount": np.random.randint(1000, 50000, n_samples),

    # Credit score between 300 (terrible) and 850 (excellent)
    "credit_score": np.random.randint(300, 850, n_samples),

    # Months at current job (0 to 360 = 30 years)
    "months_employed": np.random.randint(0, 360, n_samples),

    # Number of existing credit lines (0 to 20)
    "num_credit_lines": np.random.randint(0, 20, n_samples),

    # Interest rate on the loan (5% to 25%)
    "interest_rate": np.round(np.random.uniform(5.0, 25.0, n_samples), 2),

    # Loan term in months (12, 24, 36, 48, or 60)
    "loan_term": np.random.choice([12, 24, 36, 48, 60], n_samples),

    # Debt-to-income ratio (0.05 to 0.80) — how much of income goes to debt
    "dti_ratio": np.round(np.random.uniform(0.05, 0.80, n_samples), 2),
}

df = pd.DataFrame(data)

# ============================================================
# STEP 2: Create the Target Variable (What We're Predicting)
# ============================================================
# default = 1 means the customer FAILED to repay the loan
# default = 0 means the customer repaid successfully
#
# We create a realistic rule:
# Higher chance of default if: low income, high loan, low credit score,
# high interest rate, high debt-to-income ratio

default_probability = (
    (df["loan_amount"] / df["income"])           # High loan relative to income = risky
    + (1 - df["credit_score"] / 850)             # Low credit score = risky
    + (df["interest_rate"] / 25)                 # High interest rate = risky
    + (df["dti_ratio"])                          # High debt-to-income = risky
    - (df["months_employed"] / 360)              # More employment = safer
) / 5  # Normalize to a reasonable range

# Add some randomness (real life isn't perfectly predictable)
default_probability += np.random.normal(0, 0.1, n_samples)

# Convert probability to 0 or 1 (default or not)
df["default"] = (default_probability > 0.45).astype(int)

print(f"   ✅ Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(f"   📈 Default rate: {df['default'].mean():.1%}")
print(f"   Columns: {list(df.columns)}")

# Save the dataset (optional, nice for reference)
os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_data.csv", index=False)
print("   💾 Dataset saved to data/credit_data.csv")

# ============================================================
# STEP 3: Prepare Data for Training
# ============================================================
print("\n🔧 Step 2: Preparing data for training...")

# X = features (input), y = target (what we predict)
X = df.drop("default", axis=1)  # Everything except 'default'
y = df["default"]               # Just the 'default' column

# Split into training (80%) and testing (20%)
# Training = what the model learns from
# Testing = what we use to check if the model actually learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features (makes all numbers on a similar range)
# Without scaling: income is 20000-150000 but interest_rate is 5-25
# The model might think income is more important just because the numbers are bigger
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Training set: {X_train_scaled.shape[0]} rows")
print(f"   ✅ Testing set: {X_test_scaled.shape[0]} rows")

# ============================================================
# STEP 4: Train the Model
# ============================================================
print("\n🤖 Step 3: Training Random Forest model...")

# Random Forest = a bunch of decision trees that vote together
# Think of it like asking 100 experts and going with the majority opinion
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # Each tree can be up to 10 levels deep
    random_state=42     # Reproducible results
)

model.fit(X_train_scaled, y_train)
print("   ✅ Model trained successfully!")

# ============================================================
# STEP 5: Evaluate the Model
# ============================================================
print("\n📋 Step 4: Evaluating model performance...")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"   🎯 Accuracy: {accuracy:.2%}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Default', 'Default'])}")

# ============================================================
# STEP 6: Save the Model and Scaler
# ============================================================
print("💾 Step 5: Saving model and scaler...")

os.makedirs("model", exist_ok=True)

# Save the trained model (the brain)
joblib.dump(model, "model/model.pkl")
print("   ✅ Model saved to model/model.pkl")

# Save the scaler (the preprocessing step)
# The API needs this to scale new input data the same way
joblib.dump(scaler, "model/scaler.pkl")
print("   ✅ Scaler saved to model/scaler.pkl")

# Save feature names (so the API knows what columns to expect)
feature_names = list(X.columns)
joblib.dump(feature_names, "model/feature_names.pkl")
print(f"   ✅ Feature names saved: {feature_names}")

print("\n🎉 Done! Model is ready to be served by the Flask API.")
