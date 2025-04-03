import pandas as pd
from pycaret.classification import predict_model
from sklearn.metrics import log_loss, f1_score
import mlflow

target_column = 'shot_made_flag'

def best_model_node(test_set: pd.DataFrame, lr_model, dt_model):
    """
    Compares two classification models (logistic regression and decision tree) on a test set
    using log loss and F1-score, and returns the best model based on log loss.
    Also generates metric comparison images for each model.

    Parameters:
    ----------
    test_set : pd.DataFrame
      The test dataset including features and target column (`shot_made_flag`).
    lr_model : sklearn.base.BaseEstimator
      A trained logistic regression model.
    dt_model : sklearn.base.BaseEstimator
      A trained decision tree model.
    session_id : int
      Random seed used for initializing the PyCaret setup environment.

    Returns:
    -------
    best_model : sklearn.base.BaseEstimator
      The model with the lowest log loss on the test dataset.
    lr_model_metrics_img : bytes
      PNG image with metrics for logistic regression.
    dt_model_metrics_img : bytes
      PNG image with metrics for decision tree.
    """
    lr_pred = predict_model(lr_model, data=test_set)
    dt_pred = predict_model(dt_model, data=test_set)

    y_true = test_set[target_column]
    lr_log_loss = log_loss(y_true, lr_pred["prediction_label"])
    dt_log_loss = log_loss(y_true, dt_pred["prediction_label"])
    lr_f1 = f1_score(y_true, lr_pred["prediction_label"])
    dt_f1 = f1_score(y_true, dt_pred["prediction_label"])

    if lr_log_loss < dt_log_loss:
        print(f"Selected Logistic Regression: log_loss={lr_log_loss:.4f}, f1={lr_f1:.4f}")
        best_model = lr_model
        best_log_loss = lr_log_loss
        best_f1 = lr_f1
    else:
        print(f"Selected Decision Tree: log_loss={dt_log_loss:.4f}, f1={dt_f1:.4f}")
        best_model = dt_model
        best_log_loss = dt_log_loss
        best_f1 = dt_f1

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_metric("log_loss_lr", lr_log_loss)
        mlflow.log_metric("f1_score_lr", lr_f1)
        mlflow.log_metric("log_loss_dt", dt_log_loss)
        mlflow.log_metric("f1_score_dt", dt_f1)
        mlflow.log_metric("log_loss_best", best_log_loss)
        mlflow.log_metric("f1_score_best", best_f1)

    lr_model_metrics= generate_metrics("Logistic Regression Metrics", lr_log_loss, lr_f1)
    dt_model_metrics = generate_metrics("Decision Tree Metrics", dt_log_loss, dt_f1)

    return best_model, lr_model_metrics, dt_model_metrics


# Create metrics
def generate_metrics(title, log_loss_val, f1_val):
    metrics_text = f"{title}\n\nLog Loss: {log_loss_val:.4f}\nF1 Score: {f1_val:.4f}"
    return metrics_text