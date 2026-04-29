"""
Usage pattern
-------------
1. Subclass ``BaseFactory`` and declare ``_registry: Dict[str, Type[T]] = {}``.
2. Decorate concrete classes with ``@MyFactory.register("name", "alias")``.
3. Call ``MyFactory.create("name", **kwargs)`` to get an instance.

Each subclass **must** declare its own ``_registry`` dict so the registries
stay isolated — a kernel name will never collide with a loss name.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Type


class BaseFactory(ABC):
    """
    Generic registry / factory.

    Sub-classes must define their own class-level ``_registry`` dict.
    Failing to do so means all subclasses share the same dict, which
    causes cross-family name collisions.
    """

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, *names: str):
        """
        Class decorator that registers a class under one or more aliases.

        Parameters
        ----------
        *names : str
            One or more lowercase-normalised aliases.

        Raises
        ------
        KeyError
            If any alias is already taken in this registry.

        Example
        -------
        >>> @MyFactory.register("foo", "f")
        ... class Foo(Base): ...
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
        Instantiate a registered class by name.

        Parameters
        ----------
        name : str
            Registered alias (case-insensitive).
        **kwargs
            Forwarded verbatim to the class constructor.

        Returns
        -------
        An instance of the registered class.

        Raises
        ------
        KeyError
            If *name* is not registered.
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
        """Return a sorted list of all registered aliases."""
        return sorted(cls._registry)

    @classmethod
    def contains(cls, name: str) -> bool:
        """Return ``True`` if *name* is a registered alias."""
        return name.lower() in cls._registry
