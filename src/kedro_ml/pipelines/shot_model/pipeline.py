from kedro.pipeline import node, Pipeline, pipeline
from .nodes.data_acquisition import data_acquisition_node
from .nodes.data_preparation import data_preparation_node
from .nodes.data_feature import data_feature_preparation
from .nodes.data_splitting import data_splitting
from .nodes.models import model_logistic_regression, model_decision_tree
from .nodes.best_model import best_model_node

def create_pipeline(**kargs) -> Pipeline:
    return pipeline([
        node(
            data_acquisition_node,
            inputs=[],
            outputs=['dev_raw_train', 'prod_raw_train'],
            tags=['data_acquisition']
        ),
        node(
            data_preparation_node,
            inputs=['dev_raw_train', 'prod_raw_train'],
            outputs=['prepared_dev_raw_train', 'prepared_prod_raw_train'],
            tags=['data_preparation']
        ),
        node(
            data_feature_preparation,
            inputs=['prepared_dev_raw_train', 'prepared_prod_raw_train'],
            outputs=['selected_features_dev_raw_train', 'selected_features_prod_raw_train'],
            tags=['data_feature_preparation']
        ),
        node(
            data_splitting,
            inputs=['selected_features_dev_raw_train', 'selected_features_prod_raw_train'],
            outputs=[
                'dev_train_data',
                'dev_test_data',
                'prod_train_data',
                'prod_test_data'
            ],
            tags=['data_splitting']
        ),
        node(
            model_logistic_regression,
            inputs=['dev_train_data', 'params:session_id'],
            outputs=['lr_model'],
            tags=['training', 'model']
        ),
        node(
            model_decision_tree,
            inputs=['dev_train_data', 'params:session_id'],
            outputs=['dt_model'],
            tags=['training', 'model']
        ),
        node(
            best_model_node,
            inputs=[
                'dev_test_data',
                'lr_model',
                'dt_model',
                'params:session_id'
            ],
            outputs='best_model',
            tags=['best_model', 'model']
        )
    ])
