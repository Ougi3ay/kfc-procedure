"""
Factory infrastructure for dynamic component registration and instantiation.

This module provides the :class:`BaseFactory` abstraction used throughout
the library to implement registry-based discovery and object creation.

Factories decouple component selection from implementation details by
allowing classes to be registered under symbolic names and instantiated
dynamically at runtime. This pattern is used to support extensible
architectures where kernels, estimators, splitters, optimizers, or other
components can be added without modifying existing factory logic.

Notes
-----
Each factory subclass maintains an independent registry and supports:

* Dynamic class registration
* Alias-based lookup
* Category-based filtering
* Metadata storage
* Runtime object creation
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Set, Type


class BaseFactory(ABC):
    """
    Abstract registry-based factory.

    The factory pattern provides a centralized mechanism for registering,
    discovering, and instantiating implementation classes using symbolic
    names rather than direct class references.

    This abstraction enables loosely coupled and extensible software
    architectures where new components can be integrated through
    registration without modifying existing client code.

    Each subclass maintains an independent registry, allowing multiple
    factory types to coexist safely within the same framework.

    Examples
    --------
    Register a component:

    >>> @KernelFactory.register("gaussian")
    ... class GaussianKernel:
    ...     pass

    Create an instance:

    >>> kernel = KernelFactory.create("gaussian")

    Query available implementations:

    >>> KernelFactory.available()
    ['gaussian']

    Notes
    -----
    Registration names are case-insensitive and are stored internally
    in lowercase form.
    """

    _registry: Dict[str, Dict[str, Any]] = {}

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
    def register(
        cls,
        *names: str,
        categories: Optional[Set[str] | str] = None,
        **metadata: Any
    ):
        """
        Register a class under one or more names.

        This method is typically used as a decorator.

        Parameters
        ----------
        *names : str
            One or more string names used to register the target class.
        
        categories:
            Optional single category or set of categories.

        metadata:
            Optional extra info (future extensibility).

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
        >>> @Factory.register("adam", categories={"optimizer"})
        ... class Adam:
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
                        f"'{key}' is already registered in {cls.__name__}. "
                        f"Existing entry: {cls._registry[key]['class'].__name__}"
                    )
                cls._registry[key] = {
                    "class" : target_cls,
                    "categories" : categories,
                    "metadata" : metadata
                }
            return target_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Instantiate a registered implementation.

        Parameters
        ----------
        name : str
            Registered name or alias identifying the desired implementation.

        **kwargs
            Keyword arguments forwarded directly to the implementation
            constructor.

        Returns
        -------
        Any
            Newly created instance of the registered implementation.

        Raises
        ------
        KeyError
            If no implementation is registered under the specified name.

        Notes
        -----
        Factory creation abstracts the concrete implementation class from
        client code, promoting modularity and configurability.
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
    
    @classmethod
    def available_categories(cls) -> Set[str]:
        """
        Return all registered categories.

        Categories provide a lightweight mechanism for grouping related
        implementations within a factory.

        Returns
        -------
        set of str
            Unique category names currently present in the registry.

        Examples
        --------
        >>> KernelFactory.available_categories()
        {'distance', 'similarity'}
        """
        categories: Set[str] = set()
        for meta in cls._registry.values():
            categories.update(meta.get("categories", set()))
        
        return categories
    
    @classmethod
    def available_by_category(cls, category: str) -> List[str]:
        """
        Return registered names belonging to a category.

        Parameters
        ----------
        category : str
            Category label used for filtering.

        Returns
        -------
        list of str
            Sorted registration names associated with the specified category.

        Notes
        -----
        Category matching is case-insensitive.
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
        Return metadata associated with a registered implementation.

        Parameters
        ----------
        name : str
            Registered component name.

        Returns
        -------
        dict
            Registration information containing:

            - ``name`` : normalized registration name
            - ``class`` : implementation class name
            - ``categories`` : associated categories
            - ``metadata`` : user-defined metadata

        Raises
        ------
        KeyError
            If the specified name is not registered.
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
        Find registration names associated with a class.

        Parameters
        ----------
        target_cls : type
            Implementation class to search for.

        Returns
        -------
        list of str
            Sorted registration names mapped to the specified class.

        Notes
        -----
        A single implementation may be registered under multiple aliases.
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
        Determine whether a registered implementation belongs to a category.

        Parameters
        ----------
        name : str
            Registered implementation name.

        category : str
            Category to test.

        Returns
        -------
        bool
            True if the implementation is registered and associated with the
            specified category; otherwise False.

        Notes
        -----
        Both the registration name and category comparison are
        case-insensitive.
        """
        key = name.lower()

        if key not in cls._registry:
            return False
        
        categories = cls._registry[key].get(
            "categories",
            set(),
        )

        return category.lower() in categories
        