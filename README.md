# Transformer Fault Detection Using Machine Learning Models

A machine-learning-based system for detecting and classifying transformer faults using **Dissolved Gas Analysis (DGA)** data. The project analyzes dissolved gas concentrations and applies multiple machine learning models to identify transformer operating conditions and fault types.

## 📌 Project Overview

Power transformers are critical components of electrical power systems, and early fault detection is important for preventing unexpected failures, reducing maintenance costs, and improving system reliability.

This project uses DGA measurements as input features and machine learning techniques to classify transformer conditions into four categories:

- **Normal**
- **Partial Discharge (PD)**
- **Thermal Fault**
- **Arcing**

The repository contains Python scripts and a Jupyter Notebook for data analysis, model training, model comparison, evaluation, and feature-importance analysis.

## 🎯 Objectives

- Analyze transformer DGA data using Python.
- Explore gas concentration patterns associated with different fault types.
- Preprocess and prepare DGA data for machine learning.
- Train and compare multiple machine learning models.
- Evaluate models using standard classification metrics.
- Analyze the importance of individual dissolved gases for fault diagnosis.
- Provide example predictions for different transformer conditions.

## 🧪 DGA Features

The models use seven dissolved-gas concentration features:

| Feature | Gas |
|---|---|
| H2 | Hydrogen |
| CH4 | Methane |
| C2H2 | Acetylene |
| C2H4 | Ethylene |
| C2H6 | Ethane |
| CO | Carbon Monoxide |
| CO2 | Carbon Dioxide |

The target variable is `fault_type`.

## 🤖 Machine Learning Models

The project includes implementations and comparisons of several approaches:

1. **Linear Regression with Ridge Regularization**  
   Features are standardized before training, and the continuous predictions are rounded and constrained to the available fault classes.

2. **Decision Tree Classifier**  
   A tree-based classification model used to learn relationships between dissolved-gas concentrations and fault classes.

3. **Random Forest Classifier**  
   An ensemble of decision trees used for robust classification and feature-importance analysis.

4. **Optimized Random Forest**  
   Random Forest hyperparameters are tuned using `GridSearchCV` to search for a suitable model configuration.

5. **HRF-MEPSO**  
   The main experimental pipeline also contains an HRF-MEPSO implementation for optimized transformer fault classification.

> Note: The HRF-MEPSO pipeline depends on supporting modules referenced by `transformer_fault_detection.py`. Make sure the corresponding `models` and `utils` modules are available before running that pipeline.

## 🔄 Workflow

```text
DGA Dataset
     ↓
Data Loading
     ↓
Exploratory Data Analysis
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Hyperparameter Optimization
     ↓
Fault Prediction
     ↓
Model Evaluation
     ↓
Model Comparison & Feature Importance
```

## 📊 Evaluation Metrics

The project evaluates model performance using:

- **Accuracy** – proportion of correctly classified samples.
- **Precision** – proportion of predicted samples that are correctly classified for the corresponding classes.
- **Recall** – ability to identify samples belonging to the corresponding classes.
- **F1-Score** – harmonic mean of precision and recall.
- **Confusion Matrix** – shows actual versus predicted class assignments.

## 📁 Repository Structure

```text
Transformerfaultdetection/
│
├── fast_model_comparison.py
├── results_summary.py
├── simple_model_training.py
├── train_models.py
├── transformer_fault_detection.py
├── transformer_fault_analysis.ipynb
└── README.md
```

### Main Files

**`transformer_fault_analysis.ipynb`**  
Jupyter Notebook containing the project analysis and experimental workflow.

**`transformer_fault_detection.py`**  
Main object-oriented pipeline for loading DGA data, preprocessing, training models, comparing results, and performing feature-importance analysis.

**`simple_model_training.py`**  
A compact training and comparison script covering Ridge-based regression, Decision Tree, Random Forest, and optimized Random Forest approaches.

**`train_models.py`**  
Additional model-training workflow for the project.

**`fast_model_comparison.py`**  
Script intended for faster comparison of the implemented machine-learning approaches.

**`results_summary.py`**  
Utility script for summarizing model results and project outputs.

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** – data handling and analysis
- **NumPy** – numerical computation
- **Scikit-learn** – machine learning, preprocessing, model selection, and evaluation
- **Matplotlib** – data visualization
- **Seaborn** – statistical visualization
- **Jupyter Notebook** – interactive experimentation and analysis

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/kalsoom99326/Transformerfaultdetection.git
cd Transformerfaultdetection
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

If you use the notebook, start Jupyter with:

```bash
jupyter notebook
```

Then open:

```text
transformer_fault_analysis.ipynb
```

## ▶️ Running the Project

For the compact model-training workflow:

```bash
python simple_model_training.py
```

For the complete pipeline:

```bash
python transformer_fault_detection.py
```

The complete pipeline expects the required DGA dataset and supporting preprocessing, evaluation, visualization, and HRF-MEPSO modules referenced in the source code.

## 🔬 Example DGA Input Format

A DGA dataset should contain gas-concentration columns similar to:

```text
H2, CH4, C2H2, C2H4, C2H6, CO, CO2, fault_type
```

Example fault labels used by the project are:

```text
Normal
Partial Discharge
Thermal Fault
Arcing
```

## 📈 Analysis and Visualization

The project includes analysis capabilities such as:

- Fault-type distribution plots
- Gas concentration analysis
- Gas correlation heatmaps
- Gas concentration box plots by fault type
- Model performance comparison
- Confusion matrices
- Feature-importance analysis
- HRF-MEPSO optimization-history plots

## ⚠️ Important Notes

- Model performance depends on the quality, size, and distribution of the DGA dataset.
- The example scripts use specific dataset paths, so update the paths if your dataset is stored elsewhere.
- The repository currently contains the Python and notebook files listed above; the complete HRF-MEPSO pipeline references additional `models` and `utils` modules that should be included before running that particular script.
- Results should be interpreted as experimental machine-learning results and should be validated with appropriate engineering/domain expertise before any real-world deployment.

## 🚀 Future Improvements

Possible future extensions include:

- Adding a larger and more diverse DGA dataset.
- Adding additional DGA-derived diagnostic ratios and engineered features.
- Testing additional classification algorithms.
- Improving model validation using repeated cross-validation.
- Adding a simple web or desktop interface for fault prediction.
- Saving trained models for future predictions.
- Adding automated data validation and preprocessing pipelines.
- Integrating real-time or periodically updated DGA measurements.

## 👩‍💻 Author

**Kalsoom**  
Computer Science / Machine Learning Project

## 📄 License

No license has been specified for this repository yet. If you plan to make the project open source, consider adding an appropriate license file.
