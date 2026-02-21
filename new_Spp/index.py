from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

# Load dataset
df = pd.read_csv('salary_data.csv')

# Handle missing salaries
df = df.dropna(subset=['salary'])  # remove rows where salary is NaN

# Rename columns for consistency
df.rename(columns={
    'yearsExperiance': 'YearsExperience',
    'post': 'Post',
    'salary': 'Salary'
}, inplace=True)

# Features and target
X = df[['YearsExperience', 'Education', 'Post']]
y = df['Salary']

# Preprocess categorical features
categorical_features = ['Education', 'Post']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)


# Train model
X_processed = preprocessor.fit_transform(X)
model = LinearRegression()
model.fit(X_processed, y)


# Step 3: Evaluate model accuracy
from sklearn.metrics import r2_score

y_pred = model.predict(X_processed)
r2 = r2_score(y, y_pred)
print(f"Model R² Score (Accuracy): {r2:.2f}")

@app.route('/')
def index():
    return render_template('index.html')  # your HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    experience = int(data['experience'])
    education = data['education']
    job_type = data['jobType']

    # Prepare input for the model
    input_df = pd.DataFrame([[experience, education, job_type]], columns=['YearsExperience','Education','Post'])
    input_processed = preprocessor.transform(input_df)

    # Predict salary
    predicted_salary = round(model.predict(input_processed)[0])

    # Confidence (simplified)
    confidence = min(95, 70 + experience * 5)

    # Salary range ±15%
    lower = round(predicted_salary * 0.85)
    upper = round(predicted_salary * 1.15)

    return jsonify({
        'predictedSalary': predicted_salary,
        'confidence': confidence,
        'lowerBound': lower,
        'upperBound': upper
    })


if __name__ == '__main__':
    app.run(debug=True, port=8000)

