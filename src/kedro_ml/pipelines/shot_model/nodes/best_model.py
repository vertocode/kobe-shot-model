import pandas as pd
from pycaret.classification import setup, predict_model
from sklearn.metrics import log_loss, f1_score

target_column = 'shot_made_flag'

def best_model_node(test_set: pd.DataFrame, lr_model, dt_model, session_id: int):
    """
    Compares two classification models (logistic regression and decision tree) on a test set
    using log loss and F1-score, and returns the best model based on log loss.

    This function uses PyCaret's `predict_model` to generate predictions and evaluates each model's
    performance with `log_loss` (as the main selection criterion) and `f1_score` (for logging).

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
    """
    # Setup is required before calling predict_model
    setup(
        data=test_set,
        target=target_column,
        session_id=session_id,
        html=False,
        verbose=False,
    )

    # Predict with each model
    lr_pred = predict_model(lr_model, data=test_set)
    dt_pred = predict_model(dt_model, data=test_set)

    # Calculate log loss to each model
    y_true = test_set[target_column]
    lr_log_loss = log_loss(y_true, lr_pred["prediction_label"])
    dt_log_loss = log_loss(y_true, dt_pred["prediction_label"])

    # Calculate f1_score to each model
    lr_f1 = f1_score(y_true, lr_pred["prediction_label"])
    dt_f1 = f1_score(y_true, dt_pred["prediction_label"])

    # Choose the best model based on the log_loss
    if lr_log_loss < dt_log_loss:
        best_model = lr_model
        print(f"Selected Logistic Regression: log_loss={lr_log_loss:.4f}, f1={lr_f1:.4f}")
    else:
        best_model = dt_model
        print(f"Selected Decision Tree: log_loss={dt_log_loss:.4f}, f1={dt_f1:.4f}")

    return best_model
