from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import RegressionPipeline


class RFRegressionPipeline(RegressionPipeline):
    """Random Forest Pipeline for regression problems"""
    name = "Random Forest Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'Random Forest Regressor']
    problem_types = ['regression']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
