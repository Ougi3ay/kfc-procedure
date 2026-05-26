"""
Registry-based factory infrastructure for dynamic class discovery and creation.

This module defines the ``BaseFactory`` abstract base class, which implements
a flexible registry-driven factory pattern. Subclasses can register
implementation classes under one or more string aliases and later create
instances dynamically from those identifiers.

The design is intended for modular and extensible systems where components
must be selected at runtime, such as:

- machine learning pipelines
- optimization frameworks
- plugin architectures
- configurable backends
- strategy-based systems

Key Features
------------
- Case-insensitive registration
- Multiple aliases per implementation
- Independent registries per factory subclass
- Optional category tagging
- Extensible metadata storage
- Runtime introspection utilities

Examples
--------
>>> class KernelFactory(BaseFactory):
...     pass

>>> @KernelFactory.register(
...     "gaussian",
...     "rbf",
...     categories={"kernel", "distance"},
... )
... class GaussianKernel:
...     def __init__(self, sigma=1.0):
...         self.sigma = sigma

>>> kernel = KernelFactory.create("gaussian", sigma=2.0)

>>> KernelFactory.available()
['gaussian', 'rbf']

>>> KernelFactory.available_categories()
{'kernel', 'distance'}
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Set, Type


class BaseFactory(ABC):
    """
    Abstract base class for registry-driven factories.

    ``BaseFactory`` provides a generic mechanism for registering classes
    under string identifiers and creating instances dynamically at runtime.

    Each subclass maintains an isolated registry, allowing multiple factory
    types to coexist independently without registration conflicts.

    The registry stores:

    - implementation class references
    - category labels
    - optional metadata

    Registration is typically performed using the :meth:`register`
    decorator.

    Notes
    -----
    - Registration names are case-insensitive.
    - All identifiers are normalized to lowercase internally.
    - Factory subclasses automatically receive independent registries
      through ``__init_subclass__``.

    Attributes
    ----------
    _registry : Dict[str, Dict[str, Any]]
        Internal registry mapping normalized names to registration metadata.

        Each entry has the structure::

            {
                "class": Type,
                "categories": Set[str],
                "metadata": Dict[str, Any],
            }

    Examples
    --------
    >>> class OptimizerFactory(BaseFactory):
    ...     pass

    >>> @OptimizerFactory.register(
    ...     "adam",
    ...     categories={"optimizer"},
    ...     version="v1",
    ... )
    ... class Adam:
    ...     pass

    >>> OptimizerFactory.contains("adam")
    True
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Initialize subclass with an independent registry.

        Each subclass receives a fresh registry dictionary to ensure
        registrations remain isolated between factory types.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments forwarded to parent classes.

        Examples
        --------
        >>> class KernelFactory(BaseFactory):
        ...     pass

        >>> class OptimizerFactory(BaseFactory):
        ...     pass

        >>> KernelFactory._registry is OptimizerFactory._registry
        False
        """
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(
        cls,
        *names: str,
        categories: Optional[Set[str] | str] = None,
        **metadata: Any,
    ):
        """
        Register a class under one or more aliases.

        This method is primarily intended for decorator-based registration.

        Parameters
        ----------
        *names : str
            One or more aliases associated with the target class.

            Registration names are normalized to lowercase before storage.

        categories : str or Set[str], optional
            Optional category labels associated with the registration.

            Categories are useful for grouping implementations by capability,
            functionality, or component type.

        **metadata : Any
            Additional arbitrary metadata attached to the registration entry.

            Metadata is stored without interpretation and can later be
            accessed through :meth:`info`.

        Returns
        -------
        callable
            A decorator that registers the target class and returns it
            unchanged.

        Raises
        ------
        KeyError
            If any provided registration name already exists in the registry.

        Notes
        -----
        Multiple aliases may point to the same implementation class.

        Examples
        --------
        >>> @KernelFactory.register(
        ...     "gaussian",
        ...     "rbf",
        ...     categories={"kernel"},
        ...     stationary=True,
        ... )
        ... class GaussianKernel:
        ...     pass
        """
        if isinstance(categories, str):
            categories = {categories}

        categories = set(categories or [])

        def decorator(target_cls: Type) -> Type:
            for name in names:
                key = name.lower()

                if key in cls._registry:
                    raise KeyError(
                        f"'{key}' is already registered in "
                        f"{cls.__name__}. "
                        f"Existing entry: "
                        f"{cls._registry[key]['class'].__name__}"
                    )

                cls._registry[key] = {
                    "class": target_cls,
                    "categories": categories,
                    "metadata": metadata,
                }

            return target_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered implementation.

        Parameters
        ----------
        name : str
            Registered name or alias of the implementation class.

        **kwargs : dict
            Keyword arguments forwarded to the implementation constructor.

        Returns
        -------
        Any
            Instantiated implementation object.

        Raises
        ------
        KeyError
            If the requested name does not exist in the registry.

        Examples
        --------
        >>> kernel = KernelFactory.create(
        ...     "gaussian",
        ...     sigma=1.5,
        ... )
        """
        key = name.lower()

        if key not in cls._registry:
            raise KeyError(
                f"'{name}' is not registered in {cls.__name__}. "
                f"Available: {cls.available()}"
            )

        return cls._registry[key]["class"](**kwargs)

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
        return sorted(cls._registry.keys())

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether a name exists in the registry.

        Parameters
        ----------
        name : str
            Registration name or alias to check.

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

    @classmethod
    def available_categories(cls) -> Set[str]:
        """
        Return all registered category labels.

        Returns
        -------
        Set[str]
            Set of unique category names currently present in the registry.

        Examples
        --------
        >>> KernelFactory.available_categories()
        {'kernel', 'distance'}
        """
        categories: Set[str] = set()

        for meta in cls._registry.values():
            categories.update(meta.get("categories", set()))

        return categories

    @classmethod
    def available_by_category(cls, category: str) -> List[str]:
        """
        Return all registered names associated with a category.

        Parameters
        ----------
        category : str
            Category label used for filtering.

        Returns
        -------
        List[str]
            Sorted list of registered names belonging to the category.

        Examples
        --------
        >>> KernelFactory.available_by_category("kernel")
        ['gaussian', 'rbf']
        """
        category = category.lower()

        return sorted(
            name
            for name, meta in cls._registry.items()
            if category in meta.get("categories", set())
        )

    @classmethod
    def info(cls, name: str) -> Dict[str, Any]:
        """
        Return registration metadata for a registered name.

        Parameters
        ----------
        name : str
            Registered name or alias.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing registration information.

            The returned structure includes::

                {
                    "name": str,
                    "class": str,
                    "categories": List[str],
                    "metadata": Dict[str, Any],
                }

        Raises
        ------
        KeyError
            If the name is not registered.

        Examples
        --------
        >>> KernelFactory.info("gaussian")
        {
            'name': 'gaussian',
            'class': 'GaussianKernel',
            'categories': ['kernel'],
            'metadata': {'stationary': True}
        }
        """
        key = name.lower()

        if key not in cls._registry:
            raise KeyError(
                f"'{name}' not found in {cls.__name__}"
            )

        meta = cls._registry[key]

        return {
            "name": key,
            "class": meta["class"].__name__,
            "categories": sorted(meta.get("categories", set())),
            "metadata": meta.get("metadata", {}),
        }

    @classmethod
    def find_by_class(cls, target_cls: Type) -> List[str]:
        """
        Return all registered aliases associated with a class.

        Parameters
        ----------
        target_cls : Type
            Implementation class to search for.

        Returns
        -------
        List[str]
            Sorted list of aliases registered for the class.

        Examples
        --------
        >>> KernelFactory.find_by_class(GaussianKernel)
        ['gaussian', 'rbf']
        """
        return sorted(
            name
            for name, meta in cls._registry.items()
            if meta["class"] is target_cls
        )

    @classmethod
    def supports(
        cls,
        name: str,
        category: str,
    ) -> bool:
        """
        Check whether a registered implementation belongs to a category.

        Parameters
        ----------
        name : str
            Registered name or alias.

        category : str
            Category label to test.

        Returns
        -------
        bool
            True if the implementation is registered under the specified
            category, otherwise False.

        Examples
        --------
        >>> KernelFactory.supports("gaussian", "kernel")
        True
        """
        key = name.lower()

        if key not in cls._registry:
            return False

        categories = cls._registry[key].get(
            "categories",
            set(),
        )

        return category.lower() in categories
