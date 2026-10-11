import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from imblearn.over_sampling import SMOTE

DATA_FILE = "heart_2022_with_nans.csv"

DISEASES = [
    "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
    "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis",
    "HadDiabetes",
]

FEATURES = [
    "AgeCategory", "Sex", "BMI", "SmokerStatus", "GeneralHealth",
    "PhysicalActivities", "SleepHours", "AlcoholDrinkers",
    "PhysicalHealthDays", "MentalHealthDays", "DifficultyWalking",
]

EPOCHS = 10
BATCH_SIZE = 256

# Load data
df = pd.read_csv(DATA_FILE)[FEATURES + DISEASES].dropna()
Y = pd.DataFrame({d: (df[d] == "Yes").astype(int) for d in DISEASES})
X = df[FEATURES].copy()

# LabelEncoder
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col])

# Split and StandardScaler
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

saved_models = {}
for d in DISEASES:
    # SMOTE on training data only
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, Y_train[d])
    X_t = torch.tensor(X_res, dtype=torch.float32)
    y_t = torch.tensor(y_res.values, dtype=torch.float32).view(-1, 1)

    # Small neural network: 11 inputs -> 16 hidden -> 1 output
    model = nn.Sequential(
        nn.Linear(len(FEATURES), 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        order = torch.randperm(len(X_t))  # shuffle every epoch
        for i in range(0, len(X_t), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t))
    pred = (probs > 0.5).int().numpy().ravel()
    acc = accuracy_score(Y_test[d], pred)
    rec = recall_score(Y_test[d], pred)
    print(f"{d}: accuracy = {acc:.3f}, recall (Yes class) = {rec:.3f}")

    saved_models[d] = model.state_dict()

torch.save(saved_models, "model_pytorch.pt")
print("Saved model_pytorch.pt")