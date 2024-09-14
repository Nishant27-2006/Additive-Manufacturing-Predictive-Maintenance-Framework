# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np

# Simulating the dataset for testing (replace this with actual data in practice)
np.random.seed(42)
n_samples = 1000
n_features = 20

# Generate random feature data and binary labels for defect status
X_simulated = np.random.randn(n_samples, n_features)
y_simulated = np.random.randint(0, 2, size=n_samples)

# Create a DataFrame for simulation
df = pd.DataFrame(X_simulated, columns=[f'Feature_{i}' for i in range(1, n_features+1)])
df['DefectStatus'] = y_simulated

# 1. Data Preprocessing

# Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.01)
df_high_variance = var_thresh.fit_transform(df.drop('DefectStatus', axis=1))

# Perform feature reduction with PCA to speed up training (optional step)
pca = PCA(n_components=10)  # Keep the top 10 components
X_pca = pca.fit_transform(df_high_variance)

# Prepare data
X = X_pca  # Using PCA reduced features
y = df['DefectStatus']  # Target variable

# Standardize features for models sensitive to scale (like Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Model 1: Random Forest Classifier with RandomizedSearchCV

# Hyperparameter space for Random Forest
param_dist_rf = {
    'n_estimators': [50, 100],  # Fewer estimators for faster training
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist_rf,
                                      n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
random_search_rf.fit(X_train, y_train)

# Predictions using the best Random Forest model
rf_best = random_search_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)

# Accuracy and classification report for Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_classification_report = classification_report(y_test, y_pred_rf)

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-defective', 'Defective'],
            yticklabels=['Non-defective', 'Defective'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance Plot for Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_best.feature_importances_, y=[f'PC{i+1}' for i in range(X_pca.shape[1])])
plt.title('Random Forest Feature Importance (Top PCA Components)')
plt.show()

# 3. Model 2: Logistic Regression with RandomizedSearchCV

# Hyperparameter space for Logistic Regression
param_dist_lr = {
    'C': np.logspace(-2, 2, 5),
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

lr_model = LogisticRegression(max_iter=1000)
random_search_lr = RandomizedSearchCV(estimator=lr_model, param_distributions=param_dist_lr,
                                      n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
random_search_lr.fit(X_train, y_train)

# Predictions using the best Logistic Regression model
lr_best = random_search_lr.best_estimator_
y_pred_lr = lr_best.predict(X_test)

# Accuracy and classification report for Logistic Regression
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_classification_report = classification_report(y_test, y_pred_lr)

# Confusion matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-defective', 'Defective'],
            yticklabels=['Non-defective', 'Defective'])
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 4. Model Comparison (Accuracy)
model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [rf_accuracy, lr_accuracy]
})

# Plot model comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy', data=model_comparison)
plt.title('Model Comparison: Accuracy')
plt.ylabel('Accuracy')
plt.show()
