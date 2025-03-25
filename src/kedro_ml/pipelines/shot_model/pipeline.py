from kedro.pipeline import node, Pipeline, pipeline
from .nodes.data_loader import data_loader_node

def create_pipeline(**kargs) -> Pipeline:
    return pipeline([
        node(
            data_loader_node,
            inputs=[],
            outputs=['dev_raw_train', 'prod_raw_train']
        )
    ])