import pandas as pd
from pycaret.classification import ClassificationExperiment

target_column = 'shot_made_flag'

def model_logistic_regression(train_set: pd.DataFrame, session_id: int):
    """
    Train a logistic regression model using the provided training dataset and tune it for better performance.

    Args:
    train_set (pd.DataFrame): The dataset used for training the model.
    session_id (int): A session ID to ensure reproducibility of the experiment.

    Returns:
    The tuned logistic regression model after optimization.

    The function sets up the PyCaret classification experiment, creates a logistic regression model,
    and tunes it by iterating over different hyperparameters to maximize the AUC (Area Under the Curve).
    """
    exp = ClassificationExperiment()
    exp.setup(data=train_set, target=target_column, session_id=session_id)

    model = exp.create_model('lr')

    tuned_model = exp.tune_model(model, n_iter=10, optimize='AUC')

    return [tuned_model]
    
    
def model_decision_tree(train_set: pd.DataFrame, session_id: int):
    """
    Train a decision tree model using the provided training dataset and tune it for better performance.

    Args:
    train_set (pd.DataFrame): The dataset used for training the model.
    session_id (int): A session ID to ensure reproducibility of the experiment.

    Returns:
    The tuned decision tree model after optimization.

    The function sets up the PyCaret classification experiment, creates a decision tree model,
    and tunes it by iterating over different hyperparameters to maximize the AUC (Area Under the Curve).
    """
    exp = ClassificationExperiment()
    exp.setup(data=train_set, target=target_column, session_id=session_id)

    model = exp.create_model('dt')

    tuned_model = exp.tune_model(model, n_iter=10, optimize='AUC')

    return [tuned_model]