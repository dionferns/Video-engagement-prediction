# Predicting Student Engagement in Video Lectures

This project develops a machine learning pipeline to predict **median engagement** scores for online video lectures. The model was trained on a rich dataset containing transcript features, video metadata, and derived linguistic and behavioral features. The workflow includes rigorous preprocessing, detailed exploratory data analysis (EDA), advanced encoding and imputation strategies, and both classical and kernel-based regression modeling.

---

## Project Structure

The project consists of three main parts:

### 1. Exploratory Data Analysis (EDA) and Preprocessing
Thorough analysis was conducted to understand the data distribution, identify outliers, manage missing values, and transform categorical features.

#### Key Preprocessing Techniques Used:
- **Outlier Removal**: Interquartile Range (IQR) method across all numerical features; ~25% of data removed.
- **Imputation**:
  - `has_parts`: Mode imputation (majority class = False)
  - `subject_domain`: Encoded missing values as a separate category
  - `silent_period_rate`: KNN imputation using 10 neighbors, justified by dependency on the `freshness` feature (validated via Chi-square test)
- **Zero Value Handling**: For features where 0 is likely invalid (e.g., `conjugate_rate`, `silent_period_rate`), values were treated as missing and replaced with NaN.
- **Encoding**:
  - `most_covered_topic`: Frequency encoding due to high cardinality
  - `lecture_type`: Frequency encoding
  - `subject_domain`: One-hot encoding
  - `has_parts`: One-hot encoding
- **Feature Scaling**: Standardization was applied across all features to ensure uniform input for regression.

---

### 2. Model Training and Evaluation

#### Models Implemented:
- **Ridge Regression** (using both Scikit-learn and manual NumPy implementation)
- **Kernel Ridge Regression** using a custom **Gaussian (RBF) kernel**

#### Evaluation Metrics:
- **Root Mean Squared Error (RMSE)**
- **Explained Variance Score**
- **Pearson Correlation Coefficient**

> Example Ridge Regression Results:  
> - RMSE: 0.0396  
> - Explained Variance: 0.49  
> - R² Score: 0.33  

#### Overfitting Check:
- Training RMSE: 0.1656  
- Testing RMSE: 0.1724  
- Clear signs of overfitting observed — addressed through regularization tuning.

---

### 3. Ridge Regression from Scratch (Matrix & SGD Formulations)

This section bridges theory and implementation by deriving and implementing ridge regression using both:
- **Closed-form matrix solution**  
- **Stochastic Gradient Descent (SGD)** with:
  - Batch training
  - Training/validation loss tracking
  - Visualizations of training dynamics over 2000 epochs
  - Plots showing the impact of learning rate and regularization

---

## Files

- `preprocess_lecture_dataset()`: Main preprocessing pipeline  
- `train_ridge_model()`: Scikit-learn-based ridge regression  
- `train_kernel_ridge_model()`: Kernel regression with custom Gaussian kernel  
- `fit_ridge_reg()`: Manual ridge regression implementation (matrix-based)  
- `train_stoch_ridge_reg()`: SGD-based ridge regression  
- `evaluate_ridge_model()`: RMSE and variance evaluation on train/test sets  
- `visualise_loss_values()`: Plots of loss curves over training epochs

---

## Dataset Overview

- **File**: `lectures_dataset.csv`
- **Size**: ~11,548 rows × 21 features
- **Target**: `median_engagement` (range 0–1)

Includes:
- Transcript linguistic features (e.g., `preposition_rate`, `document_entropy`)
- Metadata (e.g., `duration`, `freshness`)
- Video attributes (e.g., `has_parts`, `lecture_type`)

---

## Technologies

- Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook
- Custom ML implementations using matrix algebra and SGD

---

## Key Takeaways

- Built a robust ML pipeline from raw CSV to prediction using both Scikit-learn and custom methods
- Practiced real-world preprocessing challenges: missing values, cardinality, and data leakage
- Connected theoretical ML concepts to practical implementations in Python

---

## Future Improvements

- Incorporate deep learning models (e.g., MLPs) for non-linear patterns
- Explore feature selection or dimensionality reduction (PCA)
- Cross-validate regularization parameters and tune kernels more exhaustively
