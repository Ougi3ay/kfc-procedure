"""
Factory classes for dynamic instantiation of components by string name.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Type


class BaseFactory(ABC):
    """
    Abstract base class for all component factories.

    A factory maintains a registry mapping string aliases to concrete classes.
    It provides utilities to:
    - register new components
    - instantiate components by name
    - inspect available components

    This pattern is critical for research modularity, allowing different
    implementations of estimators, distances, kernels, aggregators, and
    optimizers to be swapped without modifying core algorithm code.

    Notes
    -----
    - Each subclass has its own independent registry.
    - Aliases are case-insensitive (normalized to lowercase).
    - Designed for extensibility and experimentation workflows.
    """

    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure each concrete factory subclass owns an isolated registry."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, *names: str):
        """
        Register a class under one or more string aliases.

        This method is intended to be used as a decorator.

        Parameters
        ----------
        *names : str
            One or more aliases used to reference the class.
            Aliases are case-insensitive and stored in lowercase.

        Returns
        -------
        decorator : callable
            A decorator that registers the target class.

        Raises
        ------
        KeyError
            If any alias is already registered in this factory.

        Example
        -------
        >>> @DistanceFactory.register("euclidean", "l2")
        ... class EuclideanDistance:
        ...     pass
        """
        def decorator(target_cls: Type) -> Type:
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise KeyError(
                        f"'{key}' is already registered in {cls.__name__}. "
                        f"Existing entry: {cls._registry[key].__name__}"
                    )
                cls._registry[key] = target_cls
            return target_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Instantiate a registered component by its alias.

        Parameters
        ----------
        name : str
            Alias of the component to instantiate (case-insensitive).
        **kwargs
            Keyword arguments passed to the component constructor.

        Returns
        -------
        instance : Any
            Instantiated component.

        Raises
        ------
        KeyError
            If the alias is not found in the registry.

        Example
        -------
        >>> kernel = KernelFactory.create("rbf", bandwidth=0.5)
        >>> distance = DistanceFactory.create("euclidean")
        """
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(
                f"'{name}' is not registered in {cls.__name__}. "
                f"Available: {cls.available()}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """
        List all registered aliases.

        Returns
        -------
        list of str
            Sorted list of available component names.

        Example
        -------
        >>> KernelFactory.available()
        ['rbf', 'epanechnikov', 'indicator']
        """
        return sorted(cls._registry)

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether a component alias exists in the registry.

        Parameters
        ----------
        name : str
            Alias to check.

        Returns
        -------
        bool
            True if the alias is registered, False otherwise.

        Example
        -------
        >>> KernelFactory.contains("rbf")
        True
        """
        return name.lower() in cls._registry
