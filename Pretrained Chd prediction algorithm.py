import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score

# -----------------------------
# Data Cleaning and Preprocessing
# -----------------------------
def handle_missing_values_advanced(X):
    """
    Handle missing values by applying KNNImputer solely on numerical columns.
    For categorical columns, fill missing values with the most frequent value.
    """
    # Split data into numeric and categorical dataframes
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns
    if not numeric_cols.empty:
        imputer = KNNImputer(n_neighbors=5)
        X_numeric = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), 
                                 columns=numeric_cols, index=X.index)
    else:
        X_numeric = pd.DataFrame(index=X.index)
    
    # Impute categorical columns using most frequent value imputation
    X_categorical = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    # Concatenate and return the imputed dataframe
    return pd.concat([X_numeric, X_categorical], axis=1)

def handle_outliers_advanced(X, y, contamination=0.05):
    """
    Detect and remove outliers using IsolationForest on numerical features.
    """
    numeric_cols = X.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        iso = IsolationForest(contamination=contamination, random_state=42)
        outlier_mask = iso.fit_predict(X[numeric_cols]) == 1
        return X[outlier_mask].copy(), y.loc[outlier_mask].copy()
    else:
        return X, y

def encode_categorical_variables(X):
    """
    Encode categorical variables using OneHotEncoder.
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]),
                               columns=encoder.get_feature_names_out(categorical_cols),
                               index=X.index)
        X = X.drop(categorical_cols, axis=1)
        X = pd.concat([X, encoded], axis=1)
    return X

def normalize_data(X_train, X_val, X_test, numerical_cols):
    """
    Normalize numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_val, X_test

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance classes.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# -----------------------------
# Feature Selection
# -----------------------------
def select_features_kbest(X_train, y_train, X_val, y_val, k=10):
    """
    Select top k features using mutual information and evaluate performance.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    selected_features = X_train.columns[selector.get_support()]
    print("Selected features (KBest):", selected_features.tolist())
    return X_train_selected, X_val_selected, selected_features

def select_features_rfe(X_train, y_train, X_val, y_val, n_features=10):
    """
    Select features using Recursive Feature Elimination (RFE) with a RandomForestClassifier.
    """
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    selected_features = X_train.columns[selector.get_support()]
    print("Selected features (RFE):", selected_features.tolist())
    return X_train_selected, X_val_selected, selected_features

def evaluate_feature_selection(X_train, y_train, X_val, y_val, method_name):
    """
    Train and evaluate a simple model to compare feature selection methods.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy with features from {method_name}: {accuracy:.2f}")
    return accuracy

# -----------------------------
# Main Preprocessing Pipeline
# -----------------------------
def main():
    # Load dataset
    hospital_data = pd.read_csv('MGH_PredictionDataSet.csv')
    
    # Define target and features
    X = hospital_data.drop('TenYearCHD', axis=1)
    y = hospital_data['TenYearCHD']
    
    # Handle missing values (impute numeric and categorical separately)
    X = handle_missing_values_advanced(X)
    
    # Handle outliers (apply only to numeric features)
    X, y = handle_outliers_advanced(X, y)
    
    # Encode categorical variables
    X = encode_categorical_variables(X)
    
    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]
    
    # Identify numerical columns (after encoding)
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Split the dataset into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Apply SMOTE to training data to balance class distribution
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
    
    # -----------------------------
    # Feature Selection Section
    # -----------------------------
    k = 10
    X_train_kbest, X_val_kbest, selected_kbest = select_features_kbest(X_train, y_train, X_val, y_val, k=k)
    X_train_rfe, X_val_rfe, selected_rfe = select_features_rfe(X_train, y_train, X_val, y_val, n_features=k)
    
    # Evaluate the selected features on a simple RandomForestClassifier model
    accuracy_kbest = evaluate_feature_selection(X_train_kbest, y_train, X_val_kbest, y_val, "KBest")
    accuracy_rfe = evaluate_feature_selection(X_train_rfe, y_train, X_val_rfe, y_val, "RFE")
    
    # Save feature selection results
    pd.DataFrame(X_train_kbest, columns=selected_kbest).to_csv('X_train_kbest.csv', index=False)
    pd.DataFrame(X_val_kbest, columns=selected_kbest).to_csv('X_val_kbest.csv', index=False)
    pd.DataFrame(X_train_rfe, columns=selected_rfe).to_csv('X_train_rfe.csv', index=False)
    pd.DataFrame(X_val_rfe, columns=selected_rfe).to_csv('X_val_rfe.csv', index=False)

if __name__ == "__main__":
    main()
