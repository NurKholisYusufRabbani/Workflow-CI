import pandas as pd
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if os.path.exists("model_output"):
    shutil.rmtree("model_output")

file_path = 'StudentPerformanceFactors_preprocessing.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

mlflow.sklearn.save_model(model, "model_output")
print("SUKSES: Model berhasil di-retraining dan disimpan ke folder 'model_output'")