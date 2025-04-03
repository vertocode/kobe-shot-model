from .shot_model_dev import create_pipeline as create_shot_model_dev
from .shot_model_prod import create_pipeline as create_shot_model_prod

def get_pipelines():
    dev_pipeline = create_shot_model_dev()
    prod_pipeline = create_shot_model_prod()

    return dev_pipeline + prod_pipeline