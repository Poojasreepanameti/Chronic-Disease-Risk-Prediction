import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Dataset: CDC BRFSS 2022 (Kaggle: kamilpytlak/personal-key-indicators-of-heart-disease)
DATA_FILE = "heart_2022_with_nans.csv"

# The 10 diseases to predict
DISEASES = [
    "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
    "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis",
    "HadDiabetes",
]

# Input features used for prediction (edit this list to add or remove features)
FEATURES = [
    "AgeCategory", "Sex", "BMI", "SmokerStatus", "GeneralHealth",
    "PhysicalActivities", "SleepHours", "AlcoholDrinkers",
    "PhysicalHealthDays", "MentalHealthDays", "DifficultyWalking",
]

# Load data, keep only the needed columns, remove rows with missing values
df = pd.read_csv(DATA_FILE)[FEATURES + DISEASES].dropna()

# Targets: 1 = "Yes", 0 = anything else
Y = pd.DataFrame({d: (df[d] == "Yes").astype(int) for d in DISEASES})

# Features: only the selected columns
X = df[FEATURES].copy()

# LabelEncoder for text columns
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])
        encoders[col] = enc

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One model per disease, SMOTE on training data only
models = {}
for d in DISEASES:
    y_train = Y_train[d]
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_res, y_res)
    models[d] = model

    acc = accuracy_score(Y_test[d], model.predict(X_test))
    print(f"{d}: Yes before SMOTE = {y_train.sum()}/{len(y_train)}, "
          f"after SMOTE = {y_res.sum()}/{len(y_res)}, test accuracy = {acc:.3f}")

# Save everything the app needs
saved = {
    "models": models,
    "scaler": scaler,
    "encoders": encoders,
    "columns": list(X.columns),
    "defaults": X.mean().to_dict(),
    "diseases": DISEASES,
}
with open("model.pkl", "wb") as f:
    pickle.dump(saved, f)

print("Saved model.pkl")