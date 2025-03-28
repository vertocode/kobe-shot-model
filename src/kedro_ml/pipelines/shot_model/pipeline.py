from kedro.pipeline import node, Pipeline, pipeline
from .nodes.data_acquisition import data_acquisition_node
from .nodes.data_preparation import data_preparation_node
from .nodes.data_feature import data_feature_preparation
from .nodes.data_splitting import data_splitting

def create_pipeline(**kargs) -> Pipeline:
    return pipeline([
        node(
            data_acquisition_node,
            inputs=[],
            outputs=['dev_raw_train', 'prod_raw_train']
        ),
        node(
            data_preparation_node,
            inputs=['dev_raw_train', 'prod_raw_train'],
            outputs=['prepared_dev_raw_train', 'prepared_prod_raw_train']
        ),
        node(
            data_feature_preparation,
            inputs=['prepared_dev_raw_train', 'prepared_prod_raw_train'],
            outputs=['selected_features_dev_raw_train', 'selected_features_prod_raw_train']
        ),
        node(
            data_splitting,
            inputs=['selected_features_dev_raw_train', 'selected_features_prod_raw_train'],
            outputs=[
                'dev_train_data',
                'dev_test_data',
                'prod_train_data',
                'prod_test_data'
            ]
        )
    ])
