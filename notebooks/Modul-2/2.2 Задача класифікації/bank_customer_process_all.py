# Попередня обробка даних
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# train_df = pd.read_csv("/content/train.csv")

def process_data(train_df):

    train_tr_df, train_val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['Exited'])
    
    # назви колонок, які сформують вхідні незалежні дані
    # підготовка даних для входу в модель
    target_col = 'Exited'
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    input_cols = [col for col in train_df.columns if col != target_col and col not in cols_to_drop]
    
    train_inputs, train_targets = train_tr_df[input_cols], train_tr_df[target_col]
    val_inputs, val_targets = train_val_df[input_cols], train_val_df[target_col]
    
    # назви колонок, які є числовими і категоріальними
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()
    
    # Створюємо трансформери для числових і категоріальних колонок
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Комбінуємо трансформери для різних типів колонок в один препроцесор
    preprocessor = ColumnTransformer(transformers=[
        # ('назва', який_трансформер, для_яких_колонок)
          ('num', numeric_transformer, numeric_cols),
          ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Стоврюємо пайплайн, який спочатку запускає препроцесинг, потім тренуєм модель
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),          # готуємо дані
        ('classifier', LogisticRegression())     # передаємо готові дані в модель
    ])
    
    # Тренуємо пайплайн
    model_pipeline.fit(train_inputs, train_targets)
    
    result = {
        'model': model_pipeline,
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'val_inputs': val_inputs,
        'val_targets': val_targets,
    }

    return result

# Функція, щоб передбачати і рахувати метрики
def predict_and_plot(model_pipeline, inputs, targets, name=''):
    preds = model_pipeline.predict(inputs)
    y_probs = model_pipeline.predict_proba(inputs)[:, 1]
    roc_auc = roc_auc_score(targets, y_probs)
    print(f"Area under ROC score on {name} dataset: {roc_auc*100:.2f}%")
    confusion_matrix_ = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(confusion_matrix_, annot=True, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()
    return preds
