from kedro.pipeline import node, Pipeline, pipeline
from .nodes import data_loader

def create_pipeline(**kargs) -> Pipeline:
    return pipeline([
        node(
            data_loader.data_loader_node,
            inputs=[],
            outputs='data'
        )
    ])