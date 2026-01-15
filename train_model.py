import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

#load data
df = pd.read_csv("data/cleaned_survey.csv")

#encode the target column
df["treatment"]=df["treatment"].map({"Yes":1,"No":0})

#drop unnecessary columns
columns_to_drop = ["state","comments","obs_consequence","Country","Timestamp","tech_company"]
df = df.drop(columns=columns_to_drop, errors="ignore")

#features and target
X = df.drop("treatment", axis=1) #X is the input matrix with every columns except the ones in the list above and "treatment"
y = df["treatment"] #y is the binary label with 0=No, 1=Yes

#identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

#preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

#model pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

#train-test split
#split into train and test sets with 20% testing and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit model
model_pipeline.fit(X_train, y_train)

# Save feature names for validation during prediction
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

#save entire pipeline
joblib.dump(model_pipeline, "models/model_pipeline.pkl")
print("Model pipeline saved as 'model_pipeline.pkl'")

#keep y_test as numeric for metric calculation
y_pred = model_pipeline.predict(X_test)

#calculate metrics (using numeric labels)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
labels = [0, 1]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
class_report = classification_report(y_test, y_pred, target_names=["No", "Yes"])

#display results
print("🎯 Model Evaluation Metrics\n")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\n🔄 Confusion Matrix:")
print(conf_matrix)

print("\n📊 Classification Report:")
print(class_report)

print("\n✅ Evaluation complete!")