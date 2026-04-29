"""
Base factory module for dynamic class registration and object creation.

This module provides the ``BaseFactory`` abstract base class, which implements
a registry-based factory pattern. Subclasses can register implementation
classes using decorators and instantiate them dynamically by name.

The factory is designed to support modular and extensible architectures,
especially useful for machine learning pipelines where components such as
splitters, kernels, optimizers, or estimators need to be selected
dynamically.

Examples
--------
>>> class KernelFactory(BaseFactory):
...     pass

>>> @KernelFactory.register("gaussian", "rbf")
... class GaussianKernel:
...     def __init__(self, sigma=1.0):
...         self.sigma = sigma

>>> kernel = KernelFactory.create("gaussian", sigma=2.0)
>>> type(kernel).__name__
'GaussianKernel'
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Type


class BaseFactory(ABC):
    """
    Abstract base class for registry-based factories.

    This class provides a generic mechanism for registering classes
    under string names and creating instances dynamically using those names.

    Each subclass maintains its own independent registry, ensuring that
    different factory types (e.g., kernels, splitters, optimizers)
    do not interfere with each other.

    Attributes
    ----------
    _registry : Dict[str, Any]
        Internal mapping from registered names to implementation classes.

    Notes
    -----
    Registration is case-insensitive. All names are stored in lowercase.

    Subclasses automatically receive a fresh registry through
    ``__init_subclass__()``.

    Examples
    --------
    >>> class DistanceFactory(BaseFactory):
    ...     pass

    >>> @DistanceFactory.register("euclidean")
    ... class EuclideanDistance:
    ...     pass

    >>> DistanceFactory.available()
    ['euclidean']
    """

    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Initialize subclass with an independent registry.

        This ensures that each factory subclass maintains its own
        separate registration dictionary.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to parent classes.
        """
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, *names: str):
        """
        Register a class under one or more names.

        This method is typically used as a decorator.

        Parameters
        ----------
        *names : str
            One or more string names used to register the target class.

        Returns
        -------
        callable
            A decorator that registers the target class.

        Raises
        ------
        KeyError
            If a name is already registered.

        Examples
        --------
        >>> @KernelFactory.register("gaussian", "rbf")
        ... class GaussianKernel:
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
        Create an instance of a registered class.

        Parameters
        ----------
        name : str
            Name of the registered class to instantiate.

        **kwargs : dict
            Keyword arguments passed to the class constructor.

        Returns
        -------
        Any
            Instance of the registered class.

        Raises
        ------
        KeyError
            If the requested name is not registered.

        Examples
        --------
        >>> kernel = KernelFactory.create("gaussian", sigma=1.5)
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
        Return all registered names.

        Returns
        -------
        List[str]
            Sorted list of available registration names.

        Examples
        --------
        >>> KernelFactory.available()
        ['gaussian', 'rbf']
        """
        return sorted(cls._registry)

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether a name is registered.

        Parameters
        ----------
        name : str
            Name to check.

        Returns
        -------
        bool
            True if the name exists in the registry, otherwise False.

        Examples
        --------
        >>> KernelFactory.contains("gaussian")
        True
        """
        return name.lower() in cls._registry