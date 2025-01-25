import pandas as pd

# Load the dataset
file_path = r'C:\Users\HP\CHd prediction llm\MGH_PredictionDataSet.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head(), data.info(), data.describe()
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define preprocessing functions
def handle_missing_values_advanced(X):
    """Handle missing values using KNN Imputer."""
    imputer = KNNImputer(n_neighbors=5)
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

def handle_outliers_advanced(X, y, contamination=0.05):
    """Detect and remove outliers using Isolation Forest."""
    iso = IsolationForest(contamination=contamination, random_state=42)
    outlier_mask = iso.fit_predict(X) == 1
    return X[outlier_mask], y[outlier_mask]

def encode_categorical_variables(X):
    """Encode categorical variables using OneHotEncoder."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if categorical_cols.any():
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), 
                               columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(categorical_cols, axis=1).reset_index(drop=True)
        X = pd.concat([X, encoded], axis=1)
    return X

def normalize_data(X_train, X_val, X_test, numerical_cols):
    """Normalize numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_val, X_test

# Define SMOTE function
def apply_smote(X, y, random_state=42):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Load dataset
hospital_data = pd.read_csv('MGH_PredictionDataSet.csv')

# Define target and features
X = hospital_data.drop('TenYearCHD', axis=1)
y = hospital_data['TenYearCHD']

# Handle missing values
X = handle_missing_values_advanced(X)

# Handle outliers
X, y = handle_outliers_advanced(X, y)

# Encode categorical variables
X = encode_categorical_variables(X)

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

# Identify numerical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply SMOTE to training data
print(f"Before SMOTE: {Counter(y_train)}")
X_train, y_train = apply_smote(X_train, y_train)
print(f"After SMOTE: {Counter(y_train)}")

# Normalize numerical features
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test, numerical_cols)

# Save preprocessed data for further use
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_val.to_csv('X_val_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)



# TRAINING STEP - IMPROVED ***
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import pandas as pd

# Load data
X_train = pd.read_csv("X_train_preprocessed.csv")
X_val = pd.read_csv("X_val_preprocessed.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_val = pd.read_csv("y_val.csv").squeeze()

# Address class imbalance using SMOTE-Tomek Links
print("Before SMOTE-Tomek:", Counter(y_train))
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
print("After SMOTE-Tomek:", Counter(y_train_balanced))

# Train an XGBoost classifier with optimized parameters
def train_xgboost(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        max_depth=6,
        n_estimators=300,
        learning_rate=0.1,
        scale_pos_weight=len(y_train) / sum(y_train),  # Account for imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    
    # Validation predictions
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    f1 = f1_score(y_val, y_val_pred)
    print(f"Validation ROC AUC Score: {roc_auc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    return model

# Train the model
xgb_model = train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)

# Save the model
import joblib
joblib.dump(xgb_model, "xgboost_chd_model.pkl")

print("XGBoost model training completed and saved successfully.")
