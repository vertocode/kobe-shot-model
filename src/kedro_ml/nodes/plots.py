import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from pycaret.classification import predict_model

target_column = 'shot_made_flag'

def generate_model_report(best_model, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a classification report for the best model on the test dataset.

    Parameters:
    ----------
    best_model : sklearn.base.BaseEstimator
        The selected best model.
    dataset : pd.DataFrame
        The dataset.
    session_id : int
        Random seed for pycaret setup.

    Returns:
    -------
    pd.DataFrame
        A one-row DataFrame with various classification metrics.
    """
    predictions = predict_model(best_model, data=dataset)
    y_true = dataset[target_column]
    y_pred = predictions["prediction_label"]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

    return pd.DataFrame([metrics])
