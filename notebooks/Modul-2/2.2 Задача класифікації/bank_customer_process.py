from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and validation parts with stratification.

    :param df: Input dataframe
    :param target_col: Name of target column
    :param test_size: Fraction of validation data
    :param random_state: Random seed
    :return: Train and validation dataframes
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )


def get_input_columns(
    df: pd.DataFrame,
    target_col: str,
    cols_to_drop: List[str]
) -> List[str]:
    """
    Get feature columns excluding target and unnecessary columns.

    :param df: Input dataframe
    :param target_col: Target column name
    :param cols_to_drop: Columns to exclude
    :return: List of feature column names
    """
    return [col for col in df.columns if col != target_col and col not in cols_to_drop]


def split_features_target(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    :param df: Input dataframe
    :param input_cols: Feature columns
    :param target_col: Target column
    :return: Features (X) and target (y)
    """
    return df[input_cols], df[target_col]


def get_column_types(
    df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.

    :param df: Input dataframe
    :return: Lists of numeric and categorical column names
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def create_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numeric and categorical data.

    :param numeric_cols: List of numeric columns
    :param categorical_cols: List of categorical columns
    :return: ColumnTransformer object
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


def create_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Create full pipeline with preprocessing and model.

    :param preprocessor: Preprocessing transformer
    :return: sklearn Pipeline
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train pipeline model.

    :param pipeline: Model pipeline
    :param X_train: Training features
    :param y_train: Training targets
    :return: Trained pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def build_and_train_model(
    df: pd.DataFrame,
    target_col: str = 'Exited',
    cols_to_drop: List[str] = None
) -> Dict[str, object]:
    """
    Main orchestration function to prepare data and train model.
    Use this function instead of process_data.

    :param df: Input dataframe
    :param target_col: Target column name
    :param cols_to_drop: Columns to exclude from features
    :return: Dictionary with model and datasets
    """
    if cols_to_drop is None:
        cols_to_drop = ['id', 'CustomerId', 'Surname']

    train_df, val_df = split_data(df, target_col)

    input_cols = get_input_columns(df, target_col, cols_to_drop)

    X_train, y_train = split_features_target(train_df, input_cols, target_col)
    X_val, y_val = split_features_target(val_df, input_cols, target_col)

    numeric_cols, categorical_cols = get_column_types(X_train)

    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    pipeline = create_model_pipeline(preprocessor)

    trained_model = train_model(pipeline, X_train, y_train)

    return {
        'model': trained_model,
        'train_inputs': X_train,
        'train_targets': y_train,
        'val_inputs': X_val,
        'val_targets': y_val,
    }


def predict_and_plot(
    model: Pipeline,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = ''
) -> np.ndarray:
    """
    Make predictions, compute ROC-AUC, and plot confusion matrix.

    :param model: Trained model pipeline
    :param inputs: Feature data
    :param targets: True labels
    :param name: Dataset name (for display)
    :return: Predicted labels
    """
    preds = model.predict(inputs)
    probs = model.predict_proba(inputs)[:, 1]

    roc_auc = roc_auc_score(targets, probs)
    print(f"ROC-AUC on {name}: {roc_auc * 100:.2f}%")

    cm = confusion_matrix(targets, preds, normalize='true')

    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

    return preds