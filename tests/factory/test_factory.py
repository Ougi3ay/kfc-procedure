
import pytest


def test_available(populated_factory):
    factory = populated_factory["factory"]

    assert factory.available() == [
        "linear",
        "linear-regression",
        "tree",
    ]


def test_contains(populated_factory):
    factory = populated_factory["factory"]

    assert factory.contains("linear")
    assert factory.contains("LINEAR")
    assert factory.contains(" linear ")
    assert not factory.contains("unknown")


def test_get_class(populated_factory):
    factory = populated_factory["factory"]
    LinearModel = populated_factory["linear"]
    TreeModel = populated_factory["tree"]

    assert factory.get_class("linear") is LinearModel
    assert factory.get_class("linear-regression") is LinearModel
    assert factory.get_class("tree") is TreeModel


def test_get_class_unknown(populated_factory):
    factory = populated_factory["factory"]

    with pytest.raises(KeyError, match="is not registered"):
        factory.get_class("unknown")


def test_create_with_kwargs(populated_factory):
    factory = populated_factory["factory"]
    LinearModel = populated_factory["linear"]

    model = factory.create(
        "linear",
        weight=2.0,
        bias=3.0,
    )

    assert isinstance(model, LinearModel)
    assert model.weight == 2.0
    assert model.bias == 3.0


def test_create_with_args(populated_factory):
    factory = populated_factory["factory"]
    LinearModel = populated_factory["linear"]

    model = factory.create(
        "linear",
        4.0,
        5.0,
    )

    assert isinstance(model, LinearModel)
    assert model.weight == 4.0
    assert model.bias == 5.0


def test_available_categories(populated_factory):
    factory = populated_factory["factory"]

    assert factory.available_categories() == {
        "regression",
        "supervised",
        "tree",
    }


def test_available_by_category(populated_factory):
    factory = populated_factory["factory"]

    assert factory.available_by_category("regression") == [
        "linear",
        "linear-regression",
    ]


def test_supports(populated_factory):
    factory = populated_factory["factory"]

    assert factory.supports("linear", "regression")
    assert factory.supports("tree", "tree")
    assert not factory.supports("linear", "tree")
    assert not factory.supports("unknown", "regression")


def test_find_by_class(populated_factory):
    factory = populated_factory["factory"]
    LinearModel = populated_factory["linear"]

    assert factory.find_by_class(LinearModel) == [
        "linear",
        "linear-regression",
    ]


def test_info(populated_factory):
    factory = populated_factory["factory"]

    info = factory.info("linear")

    assert info["name"] == "linear"
    assert info["class"] == "LinearModel"
    assert info["categories"] == [
        "regression",
        "supervised",
    ]
    assert info["metadata"] == {
        "version": "1.0",
    }


def test_register_requires_name(model_factory):
    with pytest.raises(
        ValueError,
        match="At least one registration name",
    ):

        @model_factory.register()
        class InvalidModel:
            pass


def test_register_duplicate_names(model_factory):
    with pytest.raises(
        ValueError,
        match="Duplicate registration names",
    ):

        @model_factory.register(
            "model",
            "MODEL",
        )
        class InvalidModel:
            pass


def test_register_existing_name(model_factory):
    @model_factory.register("model")
    class ModelOne:
        pass

    with pytest.raises(
        KeyError,
        match="Registration conflict",
    ):

        @model_factory.register("model")
        class ModelTwo:
            pass


def test_registration_is_atomic(model_factory):
    @model_factory.register("existing")
    class ExistingModel:
        pass

    with pytest.raises(KeyError):

        @model_factory.register(
            "new-model",
            "existing",
        )
        class InvalidModel:
            pass

    assert not model_factory.contains("new-model")
    assert model_factory.contains("existing")


def test_unregister(populated_factory):
    factory = populated_factory["factory"]

    factory.unregister("linear")

    assert not factory.contains("linear")
    assert factory.contains("linear-regression")


def test_clear(populated_factory):
    factory = populated_factory["factory"]

    factory.clear()

    assert factory.available() == []
    assert factory.available_categories() == set()


def test_independent_factories(model_factory):
    class AnotherFactory(type(model_factory)):
        pass