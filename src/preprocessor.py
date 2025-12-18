import os
import joblib 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import print_header, ensure_artifacts_dir

# Preprocessing function
def preprocess_data(df, target_col,test_size , random_state , scaler_path):
    """ Preprocess the dataset by handling missing values, scaling features, and splitting into train and test sets.
    
    args:
        df (pd.DataFrame): The input dataset.
        target_col (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        scaler_path (str): Path to save the fitted scaler."""
        
    # Handle missing values
    print_header("PREPROCESSING DATA")
    
    # Map target values if necessary
    df[target_col] = df[target_col].map({-1: 0, 1:1})
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y =df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, 
                                    test_size=test_size,
                                    random_state=random_state)
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ensure_artifacts_dir(os.path.dirname(scaler_path))
    # Save the fitted scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")
    
    # Return preprocessed data
    return X_train_scaled, X_test_scaled, y_train, y_test

    
    