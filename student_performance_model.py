"""
Student Performance Prediction Model
=====================================
- Preprocessed dataset using Pandas and NumPy
- Built regression model using Scikit-learn
- Evaluated performance using RMSE and R² metrics
- Visualized results using Matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. CREATE SYNTHETIC STUDENT DATASET
# =============================================================================
print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION MODEL")
print("=" * 60)

n_students = 500

# Generate synthetic student data
data = {
    'study_hours': np.random.uniform(1, 10, n_students),
    'attendance_rate': np.random.uniform(50, 100, n_students),
    'previous_grade': np.random.uniform(40, 100, n_students),
    'sleep_hours': np.random.uniform(4, 10, n_students),
    'extracurricular_activities': np.random.randint(0, 5, n_students),
    'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_students),
    'internet_access': np.random.choice(['Yes', 'No'], n_students),
    'tutoring': np.random.choice(['Yes', 'No'], n_students),
}

# Create target variable (final_grade) with realistic relationships
final_grade = (
    5 * data['study_hours'] +
    0.3 * data['attendance_rate'] +
    0.4 * data['previous_grade'] +
    2 * data['sleep_hours'] +
    1.5 * data['extracurricular_activities'] +
    np.random.normal(0, 5, n_students)  # Add some noise
)

# Normalize to 0-100 range
final_grade = np.clip(final_grade, 0, 100)
data['final_grade'] = final_grade

# Create DataFrame
df = pd.DataFrame(data)

print("\n📊 Dataset Overview:")
print("-" * 40)
print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns) - 1}")
print(f"\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. DATA PREPROCESSING USING PANDAS AND NUMPY
# =============================================================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\n📋 Missing Values:")
print(df.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['parent_education', 'internet_access', 'tutoring']

df_processed = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"\n🔄 Encoded '{col}': {dict(zip(le.classes_, range(len(le.classes_))))}")

# Statistical summary
print("\n📈 Statistical Summary:")
print(df_processed.describe().round(2))

# Separate features and target
X = df_processed.drop('final_grade', axis=1)
y = df_processed['final_grade']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Features scaled using StandardScaler")

# =============================================================================
# 3. BUILD REGRESSION MODELS USING SCIKIT-LEARN
# =============================================================================
print("\n" + "=" * 60)
print("MODEL BUILDING")
print("=" * 60)

# Model 1: Linear Regression
print("\n🔹 Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Model 2: Random Forest Regressor
print("🔹 Training Random Forest Regressor Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

print("\n✅ Models trained successfully!")

# =============================================================================
# 4. EVALUATE PERFORMANCE USING RMSE AND R² METRICS
# =============================================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {model_name}:")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    return rmse, r2

# Evaluate both models
lr_rmse, lr_r2 = evaluate_model(y_test, y_pred_lr, "Linear Regression")
rf_rmse, rf_r2 = evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# Compare models
print("\n" + "-" * 40)
print("📈 Model Comparison:")
print("-" * 40)
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'R² Score': [lr_r2, rf_r2]
})
print(comparison_df.to_string(index=False))

# Feature importance (Random Forest)
print("\n📊 Feature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# =============================================================================
# 5. VISUALIZE RESULTS USING MATPLOTLIB
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Student Performance Prediction Model - Results', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (Linear Regression)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_lr, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Final Grade', fontsize=12)
ax1.set_ylabel('Predicted Final Grade', fontsize=12)
ax1.set_title(f'Linear Regression\nRMSE: {lr_rmse:.2f} | R²: {lr_r2:.2f}', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Random Forest)
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_rf, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Final Grade', fontsize=12)
ax2.set_ylabel('Predicted Final Grade', fontsize=12)
ax2.set_title(f'Random Forest Regressor\nRMSE: {rf_rmse:.2f} | R²: {rf_r2:.2f}', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residual Distribution
ax3 = axes[1, 0]
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf
ax3.hist(residuals_lr, bins=30, alpha=0.6, label='Linear Regression', color='blue')
ax3.hist(residuals_rf, bins=30, alpha=0.6, label='Random Forest', color='green')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Residual Distribution Comparison', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_importance)))
bars = ax4.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
ax4.set_xlabel('Importance', fontsize=12)
ax4.set_ylabel('Features', fontsize=12)
ax4.set_title('Feature Importance (Random Forest)', fontsize=12)
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for bar, val in zip(bars, feature_importance['Importance']):
    ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('student_performance_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved as 'student_performance_results.png'")

# Additional visualization: Model Comparison Bar Chart
fig2, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, [lr_rmse, rf_rmse], width, label='RMSE', color='coral')
bars2 = ax.bar(x + width/2, [lr_r2 * 10, rf_r2 * 10], width, label='R² (×10)', color='steelblue')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Linear Regression', 'Random Forest'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{bar.get_height():.2f}', ha='center', fontsize=10)
for bar, val in zip(bars2, [lr_r2, rf_r2]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Model comparison saved as 'model_comparison.png'")

plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
✅ Preprocessing completed:
   - Generated synthetic student dataset (500 samples)
   - Encoded categorical variables
   - Scaled features using StandardScaler
   - Split data into training (80%) and testing (20%) sets

✅ Models Built:
   - Linear Regression
   - Random Forest Regressor

✅ Evaluation Metrics:
   - RMSE (Root Mean Squared Error)
   - R² Score (Coefficient of Determination)

✅ Visualizations Created:
   - Actual vs Predicted scatter plots
   - Residual distribution histogram
   - Feature importance bar chart
   - Model comparison chart
""")

best_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"🏆 Best Performing Model: {best_model}")
print("=" * 60)
