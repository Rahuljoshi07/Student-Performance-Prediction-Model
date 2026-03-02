# Student Performance Prediction Model

A machine learning project that predicts student academic performance using regression models.

## Overview

This project uses machine learning techniques to predict student final grades based on various factors such as study hours, attendance, previous grades, and more.

## Features

- **Data Preprocessing**: Using Pandas and NumPy for data cleaning, encoding, and scaling
- **Machine Learning Models**: Linear Regression and Random Forest Regressor
- **Model Evaluation**: RMSE (Root Mean Squared Error) and R² Score metrics
- **Data Visualization**: Matplotlib charts for results analysis

## Dataset Features

| Feature | Description |
|---------|-------------|
| study_hours | Daily study hours (1-10) |
| attendance_rate | Class attendance percentage (50-100%) |
| previous_grade | Previous semester grade (40-100) |
| sleep_hours | Daily sleep hours (4-10) |
| extracurricular_activities | Number of activities (0-4) |
| parent_education | Education level (High School/Bachelor/Master/PhD) |
| internet_access | Internet availability (Yes/No) |
| tutoring | Private tutoring (Yes/No) |

## Results

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | 5.07 | 0.81 |
| Random Forest | 4.50 | 0.85 |

**Best Model**: Random Forest Regressor with 85% accuracy

### Key Findings
- **Study Hours** is the most important predictor (64% importance)
- **Previous Grade** is the second most important factor (21% importance)

## Installation

```bash
# Clone the repository
git clone https://github.com/Rahuljoshi07/Student-Performance-Prediction-Model.git

# Navigate to project directory
cd Student-Performance-Prediction-Model

# Install dependencies
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```bash
python student_performance_model.py
```

## Output

The script generates:
- Console output with model metrics and feature importance
- `student_performance_results.png` - Visualization with 4 charts
- `model_comparison.png` - Model comparison bar chart

## Visualizations

1. **Actual vs Predicted** - Scatter plots for both models
2. **Residual Distribution** - Histogram comparing model errors
3. **Feature Importance** - Bar chart showing feature contributions
4. **Model Comparison** - Performance metrics comparison

## Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Author

**Rahul Joshi**

## License

This project is open source and available under the [MIT License](LICENSE).
