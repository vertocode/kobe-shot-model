from .pipelines.shot_model import create_pipeline

def register_pipelines():
    return {
        "__default__": create_pipeline(),
    }