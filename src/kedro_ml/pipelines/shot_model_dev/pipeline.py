from kedro.pipeline import node, Pipeline, pipeline
from kedro_ml.nodes.data_acquisition import data_acquisition_node
from kedro_ml.nodes.data_preparation import data_preparation_node
from kedro_ml.nodes.data_feature import data_feature_preparation
from kedro_ml.nodes.data_splitting import data_splitting
from kedro_ml.nodes.models import model_logistic_regression, model_decision_tree
from kedro_ml.nodes.best_model import best_model_node
from kedro_ml.nodes.plots import generate_model_report

def create_pipeline(**kargs) -> Pipeline:
    return pipeline([
        node(
            data_acquisition_node,
            inputs=['dev'],
            outputs='dev_raw_train',
            tags=['data_acquisition']
        ),
        node(
            data_preparation_node,
            inputs=['dev_raw_train'],
            outputs='prepared_dev_raw_train',
            tags=['data_preparation']
        ),
        node(
            data_feature_preparation,
            inputs=['prepared_dev_raw_train'],
            outputs='selected_features_dev_raw_train',
            tags=['data_feature_preparation']
        ),
        node(
            data_splitting,
            inputs=['selected_features_dev_raw_train'],
            outputs=['dev_train_data', 'dev_test_data'],
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
                'dt_model'
            ],
            outputs=['best_model', 'lr_model_metrics_img', 'dt_model_metrics_img'],
            tags=['best_model', 'model']
        ),
        node(
            generate_model_report,
            inputs=['best_model', 'dev_test_data'],
            outputs='best_model_report',
            tags=['reporting']
        )
    ])
