from .pipelines import get_pipelines

def register_pipelines():
    return {
        "__default__": get_pipelines(),
    }