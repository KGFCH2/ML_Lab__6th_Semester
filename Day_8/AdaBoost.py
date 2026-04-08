import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic_toy.csv")

# Handle missing values
df = df.dropna()

# Convert categorical columns to numeric
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop('Survived', axis=1)   # Target column = Survived
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost model
model = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))
