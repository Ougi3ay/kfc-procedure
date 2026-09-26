import pytest
from kfc_procedure.factory.base import BaseFactory


def test_factory_register_create_and_aliases():
    class Factory(BaseFactory):
        pass

    @Factory.register("demo", "alias", categories={"regression"}, version="1")
    class Demo:
        def __init__(self, value=1):
            self.value = value

    obj = Factory.create("DEMO", value=42)
    assert isinstance(obj, Demo)
    assert obj.value == 42
    assert Factory.contains(" demo ")
    assert Factory.get_class("alias") is Demo
    assert Factory.supports("demo", "regression")
    assert set(Factory.available()) == {"alias", "demo"}


def test_factory_subclasses_have_independent_registries():
    class A(BaseFactory):
        pass

    class B(BaseFactory):
        pass

    @A.register("x")
    class X:
        pass

    assert A.contains("x")
    assert not B.contains("x")


def test_factory_duplicate_registration_raises():
    class Factory(BaseFactory):
        pass

    @Factory.register("x")
    class X:
        pass

    with pytest.raises(KeyError):
        @Factory.register("x")
        class X2:
            pass


def test_factory_invalid_name_raises():
    class Factory(BaseFactory):
        pass

    with pytest.raises(ValueError):
        Factory.register("")

    with pytest.raises(KeyError):
        Factory.get_class("missing")
