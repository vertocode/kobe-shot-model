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
            inputs=['params:prod'],
            outputs='prod_raw_train',
            tags=['data_acquisition']
        ),
        node(
            data_preparation_node,
            inputs=['prod_raw_train'],
            outputs='prepared_prod_raw_train',
            tags=['data_preparation']
        ),
        node(
            data_feature_preparation,
            inputs=['prepared_prod_raw_train'],
            outputs='selected_features_prod_raw_train',
            tags=['data_feature_preparation']
        ),
        node(
            data_splitting,
            inputs=['selected_features_prod_raw_train'],
            outputs=['prod_train_data','prod_test_data'],
            tags=['data_splitting']
        ),
        node(
            model_logistic_regression,
            inputs=['prod_train_data', 'params:session_id'],
            outputs=['prod_lr_model'],
            tags=['training', 'model']
        ),
        node(
            model_decision_tree,
            inputs=['prod_train_data', 'params:session_id'],
            outputs=['prod_dt_model'],
            tags=['training', 'model']
        ),
        node(
            best_model_node,
            inputs=[
                'prod_test_data',
                'prod_lr_model',
                'prod_dt_model',
                'params:prod'
            ],
            outputs=['prod_best_model', 'prod_lr_model_metrics_img', 'prod_dt_model_metrics_img'],
            tags=['prod_best_model', 'model']
        ),
        node(
            generate_model_report,
            inputs=['prod_best_model', 'prod_test_data'],
            outputs='prod_best_model_report',
            tags=['reporting']
        )
    ])
