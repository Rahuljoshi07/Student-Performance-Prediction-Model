<div align="center">

# 🎓 Student Performance Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)

**A Machine Learning project that predicts student academic performance using regression models**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Author](#-author)

---

</div>

## 📖 Overview

This project leverages machine learning techniques to predict student final grades based on various academic and personal factors. Using **Linear Regression** and **Random Forest** algorithms, the model analyzes patterns in student data to forecast academic performance with **85% accuracy**.

---

## ✨ Features

| Feature | Description |
|:--------|:------------|
| 🔄 **Data Preprocessing** | Cleaning, encoding & scaling with Pandas and NumPy |
| 🤖 **ML Models** | Linear Regression & Random Forest Regressor |
| 📊 **Evaluation Metrics** | RMSE and R² Score analysis |
| 📈 **Visualizations** | Interactive charts with Matplotlib |

---

## 📁 Dataset Features

The model uses **8 input features** to predict student performance:

| # | Feature | Type | Range | Description |
|:-:|:--------|:----:|:-----:|:------------|
| 1 | `study_hours` | Numeric | 1-10 | Daily study hours |
| 2 | `attendance_rate` | Numeric | 50-100% | Class attendance percentage |
| 3 | `previous_grade` | Numeric | 40-100 | Previous semester grade |
| 4 | `sleep_hours` | Numeric | 4-10 | Daily sleep hours |
| 5 | `extracurricular_activities` | Numeric | 0-4 | Number of activities |
| 6 | `parent_education` | Categorical | - | High School/Bachelor/Master/PhD |
| 7 | `internet_access` | Binary | - | Yes/No |
| 8 | `tutoring` | Binary | - | Yes/No |

---

## 📊 Results

### Model Performance Comparison

<div align="center">

| Model | RMSE ↓ | R² Score ↑ | Performance |
|:------|:------:|:----------:|:-----------:|
| Linear Regression | 5.07 | 0.81 | ⭐⭐⭐⭐ |
| **Random Forest** | **4.50** | **0.85** | ⭐⭐⭐⭐⭐ |

</div>

> 🏆 **Winner**: Random Forest Regressor with **85% accuracy** and lowest error rate

### 🔍 Key Findings

```
📚 Study Hours ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64%  (Most Important)
📝 Previous Grade ━━━━━━━━━━━━━ 21%
📅 Attendance Rate ━━━━━ 7%
😴 Sleep Hours ━━━━ 5%
🎯 Other Factors ━━ 3%
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Rahuljoshi07/Student-Performance-Prediction-Model.git

# Navigate to project directory
cd Student-Performance-Prediction-Model

# Install dependencies
pip install numpy pandas scikit-learn matplotlib
```

---

## 💻 Usage

Run the model with a single command:

```bash
python student_performance_model.py
```

### 📤 Output Files

| File | Description |
|:-----|:------------|
| `student_performance_results.png` | Main visualization with 4 analytical charts |
| `model_comparison.png` | Side-by-side model performance comparison |

---

## 📈 Visualizations

The project generates comprehensive visualizations:

<table>
<tr>
<td align="center"><b>1️⃣ Actual vs Predicted</b><br><sub>Scatter plots for model accuracy</sub></td>
<td align="center"><b>2️⃣ Residual Distribution</b><br><sub>Error analysis histogram</sub></td>
</tr>
<tr>
<td align="center"><b>3️⃣ Feature Importance</b><br><sub>Key predictors ranking</sub></td>
<td align="center"><b>4️⃣ Model Comparison</b><br><sub>Performance metrics chart</sub></td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:----------:|:--------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Programming Language |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data Manipulation |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Machine Learning |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square&logo=python&logoColor=white) | Data Visualization |

</div>

---

## 📂 Project Structure

```
Student-Performance-Prediction-Model/
│
├── 📄 student_performance_model.py    # Main ML pipeline
├── 📊 student_performance_results.png # Visualization results
├── 📈 model_comparison.png            # Model comparison chart
└── 📖 README.md                       # Project documentation
```

---

## 👨‍💻 Author

<div align="center">

**Rahul Joshi**

[![GitHub](https://img.shields.io/badge/GitHub-Rahuljoshi07-181717?style=for-the-badge&logo=github)](https://github.com/Rahuljoshi07)

</div>

---

## 📜 License

This project is **free and open source**  use it however you like! No restrictions, no attribution required. 🎉

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by Rahul Joshi

</div>
