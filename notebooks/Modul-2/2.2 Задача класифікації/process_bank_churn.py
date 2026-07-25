from typing import List, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    Поділяє датасет на train та validation зі стратифікацією.

    :param df: вихідний DataFrame 
    :param target_col: назва цільової змінної 
    :param test_size: частка validation вибірки
    :param random_state: seed
    :return: train_df, val_df
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    return train_df, val_df


def get_input_columns(
    df: pd.DataFrame,
    target_col: str,
    cols_to_drop: List[str]
) -> List[str]:
    """
    Формує перелік вхідних ознак. 

    :param df: вихідний DataFrame 
    :param target_col: цільова колонка 
    :param cols_to_drop: колонки для видалення 
    :return: список колонок-ознаків
    """
    return [
        col for col in df.columns
        if col != target_col and col not in cols_to_drop
    ]


def split_features_target(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Поділяє DataFrame на ознаки і target. 

    :param df: DataFrame 
    :param input_cols: перелік ознак 
    :param target_col: target колонка
    :return: X, y
    """
    return df[input_cols], df[target_col]


def get_column_types(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Визначає числові та категоріальні колонки.

    :param X: DataFrame ознак
    :return: numeric_cols, categorical_cols
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Створює ColumnTransformer для препроцессингу.

    :param numeric_cols: числові колонки
    :param categorical_cols: категоріальні колонки
    :return: ColumnTransformer
    """
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame
) -> ColumnTransformer:
    """
    Навчає препроцесор.

    :param preprocessor: ColumnTransformer
    :param X_train: тренувальні дані
    :return: навчений препроцесор
    """
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Застосовує препроцесинг.

    :param preprocessor: навчений препроцесор
    :param X: вхідні дані
    :return: numpy масив ознак
    """
    return preprocessor.transform(X)


def extract_scaler_encoder(
    preprocessor: ColumnTransformer
) -> Tuple[StandardScaler, OneHotEncoder]:
    """
    Извлекает scaler и encoder из ColumnTransformer.

    :param preprocessor: навчений препроцесор
    :return: scaler, encoder
    """
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    return scaler, encoder


def preprocess_data(
    raw_df: pd.DataFrame
) -> Tuple[
    np.ndarray, pd.Series,
    np.ndarray, pd.Series,
    List[str],
    StandardScaler,
    OneHotEncoder
]:
    """
    Повний цикл препроцессингу даних.

    :param raw_df: исходный DataFrame
    :return:
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder
    """

    target_col = 'Exited'
    cols_to_drop = ['id', 'CustomerId', 'Surname']

    # split
    train_df, val_df = split_data(raw_df, target_col)

    # ознаки
    input_cols = get_input_columns(raw_df, target_col, cols_to_drop)

    # розподіл
    X_train_df, y_train = split_features_target(train_df, input_cols, target_col)
    X_val_df, y_val = split_features_target(val_df, input_cols, target_col)

    # типи колонок
    numeric_cols, categorical_cols = get_column_types(X_train_df)

    # препроцессор
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # навчання
    preprocessor = fit_preprocessor(preprocessor, X_train_df)

    # трансформація
    X_train = transform_data(preprocessor, X_train_df)
    X_val = transform_data(preprocessor, X_val_df)

    # вилучення scaler та encoder
    scaler, encoder = extract_scaler_encoder(preprocessor)

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder	
    }

    return result


def predict_and_plot(
    model,
    X: np.ndarray,
    y: pd.Series,
    name: str = ''
) -> np.ndarray:
    """
    Делает предсказания и строит confusion matrix + ROC AUC.

    :param model: обученная модель
    :param X: признаки
    :param y: target
    :param name: название датасета
    :return: предсказания
    """
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_probs)
        print(f"ROC AUC ({name}): {roc_auc:.4f}")

    cm = confusion_matrix(y, preds, normalize='true')

    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.show()

    return preds