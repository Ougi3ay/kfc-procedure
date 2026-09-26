import pytest

from kfc_procedure.factory.base import BaseFactory


class BaseModel:
    pass

@pytest.fixture
def model_factory():
    class ModelFactory(BaseFactory[BaseModel]):
        pass

    return ModelFactory

@pytest.fixture
def populated_factory(model_factory):
    @model_factory.register(
        "linear",
        "linear-regression",
        categories={"regression", "supervised"},
        version="1.0",
    )
    class LinearModel(BaseModel):
        def __init__(
            self,
            weight: float = 1.0,
            bias: float = 0.0,
        ):
            self.weight = weight
            self.bias = bias

    @model_factory.register(
        "tree",
        categories={"tree", "supervised"},
        version="2.0",
    )
    class TreeModel(BaseModel):
        def __init__(self, depth: int = 5):
            self.depth = depth

    return {
        "factory" : model_factory,
        "linear" : LinearModel,
        "tree" : TreeModel
    }