"""
BaseFactory
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Generic, Iterable, List, Set, Type
from .entry import T, RegistryEntry



class BaseFactory(ABC, Generic[T]):
    """
    Base class for class-registration and object-creation factories.

    ``BaseFactory`` provides a registry that maps one or more string names
    to implementation classes. Registered classes may also define categories
    and arbitrary metadata.

    Each subclass receives its own independent registry.

    Registered classes can later be retrieved or instantiated by name.

    Type Parameters:
        T:
            The base type represented by this factory.

    Example:
        >>> class Model:
        ...     pass
        ...
        >>> class ModelFactory(BaseFactory[Model]):
        ...     pass
        ...
        >>> @ModelFactory.register(
        ...     "linear",
        ...     categories={"regression"},
        ...     version="1.0",
        ... )
        ... class LinearModel(Model):
        ...     def __init__(self, weight: float = 1.0):
        ...         self.weight = weight
        ...
        >>> model = ModelFactory.create("linear", weight=2.0)
        >>> isinstance(model, LinearModel)
        True
    """

    _registry: Dict[str, RegistryEntry[Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Initialize an independent registry for each factory subclass.

        Args:
            **kwargs:
                Additional arguments forwarded to ``ABC.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)

        # Each subclass maintains an independent registry.
        cls._registry = {}

    @staticmethod
    def _normalize_value(
        value: str,
        *,
        field_name: str,
    ) -> str:
        """
        Normalize and validate a registry string value.

        Values are stripped of surrounding whitespace and converted to
        lowercase.

        Args:
            value:
                String value to normalize.
            field_name:
                Human-readable field name used in error messages.

        Returns:
            The normalized string.

        Raises:
            TypeError:
                If ``value`` is not a string.
            ValueError:
                If ``value`` is empty after stripping whitespace.
        """
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be str, got {type(value).__name__}"
            )

        value = value.strip().lower()

        if not value:
            raise ValueError(f"{field_name} cannot be empty")

        return value

    @classmethod
    def _normalize_categories(
        cls,
        categories: Iterable[str] | str | None,
    ) -> frozenset[str]:
        """
        Normalize one or more category names.

        Args:
            categories:
                A category string, iterable of category strings, or ``None``.

        Returns:
            A frozen set containing normalized category names.

            If ``categories`` is ``None``, an empty ``frozenset`` is
            returned.

        Raises:
            TypeError:
                If a category is not a string.
            ValueError:
                If a category is empty after normalization.
        """
        if categories is None:
            return frozenset()

        if isinstance(categories, str):
            categories = [categories]

        return frozenset(
            cls._normalize_value(
                category,
                field_name="Category",
            )
            for category in categories
        )

    @classmethod
    def register(
        cls,
        *names: str,
        categories: Iterable[str] | str | None = None,
        **metadata: Any,
    ):
        """
        Register a class under one or more names.

        Registration names and categories are normalized by stripping
        whitespace and converting them to lowercase.

        Multiple names may reference the same class, allowing aliases for a
        single implementation.

        Args:
            *names:
                One or more names used to register the class.
            categories:
                Optional category or categories associated with the class.
            **metadata:
                Arbitrary metadata stored with the registration.

        Returns:
            A class decorator that registers the decorated class.

        Raises:
            ValueError:
                If no registration names are provided, or if duplicate names
                occur within the same registration call.
            KeyError:
                If any normalized registration name already exists in the
                factory registry.

        Example:
            >>> @ModelFactory.register(
            ...     "linear",
            ...     "linear-regression",
            ...     categories={"regression", "supervised"},
            ...     version="1.0",
            ... )
            ... class LinearModel(Model):
            ...     pass
        """
        if not names:
            raise ValueError(
                "At least one registration name must be provided"
            )

        normalized_names = [
            cls._normalize_value(
                name,
                field_name="Registry name",
            )
            for name in names
        ]

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(
                f"Duplicate registration names: {normalized_names}"
            )

        normalized_categories = cls._normalize_categories(categories)

        conflicts = [
            name
            for name in normalized_names
            if name in cls._registry
        ]

        if conflicts:
            details = ", ".join(
                (
                    f"{name!r} -> "
                    f"{cls._registry[name].target_cls.__name__}"
                )
                for name in conflicts
            )

            raise KeyError(
                f"Registration conflict in {cls.__name__}: {details}"
            )

        def decorator(target_cls: Type[T]) -> Type[T]:
            """
            Register the decorated class.

            Args:
                target_cls:
                    Class to register.

            Returns:
                The original class unchanged.
            """
            entry = RegistryEntry(
                target_cls=target_cls,
                categories=normalized_categories,
                metadata=dict(metadata),
            )

            for name in normalized_names:
                cls._registry[name] = entry

            return target_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Create an instance of a registered class.

        Positional and keyword arguments are forwarded directly to the
        registered class constructor.

        Args:
            name:
                Registered class name or alias.
            *args:
                Positional arguments passed to the class constructor.
            **kwargs:
                Keyword arguments passed to the class constructor.

        Returns:
            A new instance of the registered class.

        Raises:
            KeyError:
                If ``name`` is not registered.

        Example:
            >>> model = ModelFactory.create(
            ...     "linear",
            ...     weight=2.0,
            ...     bias=1.0,
            ... )
        """
        target_cls = cls.get_class(name)
        return target_cls(*args, **kwargs)

    @classmethod
    def get_class(
        cls,
        name: str,
    ) -> Type[T]:
        """
        Return the class registered under a given name.

        Args:
            name:
                Registered class name or alias.

        Returns:
            The registered class.

        Raises:
            TypeError:
                If ``name`` is not a string.
            ValueError:
                If ``name`` is empty.
            KeyError:
                If ``name`` is not registered.
        """
        key = cls._normalize_value(
            name,
            field_name="Registry name",
        )

        try:
            entry = cls._registry[key]
        except KeyError:
            available = ", ".join(cls.available()) or "<empty>"

            raise KeyError(
                f"{name!r} is not registered in {cls.__name__}. "
                f"Available: {available}"
            ) from None

        return entry.target_cls

    @classmethod
    def available(cls) -> List[str]:
        """
        Return all registered names.

        Returns:
            A sorted list containing every registered name and alias.
        """
        return sorted(cls._registry)

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether a registration name exists.

        Invalid names return ``False`` instead of raising an exception.

        Args:
            name:
                Registration name to check.

        Returns:
            ``True`` if the normalized name exists, otherwise ``False``.
        """
        try:
            key = cls._normalize_value(
                name,
                field_name="Registry name",
            )
        except (TypeError, ValueError):
            return False

        return key in cls._registry

    @classmethod
    def available_categories(cls) -> Set[str]:
        """
        Return all categories used by registered entries.

        Returns:
            A set containing all unique registered categories.
        """
        categories: Set[str] = set()

        for entry in cls._registry.values():
            categories.update(entry.categories)

        return categories

    @classmethod
    def available_by_category(
        cls,
        category: str,
    ) -> List[str]:
        """
        Return registration names associated with a category.

        Args:
            category:
                Category used to filter registry entries.

        Returns:
            A sorted list of registration names whose entries contain the
            requested category.

        Raises:
            TypeError:
                If ``category`` is not a string.
            ValueError:
                If ``category`` is empty.
        """
        category = cls._normalize_value(
            category,
            field_name="Category",
        )

        return sorted(
            name
            for name, entry in cls._registry.items()
            if category in entry.categories
        )

    @classmethod
    def supports(
        cls,
        name: str,
        category: str,
    ) -> bool:
        """
        Check whether a registered name belongs to a category.

        Invalid names or categories return ``False`` instead of raising an
        exception.

        Args:
            name:
                Registration name to check.
            category:
                Category to check against the registered entry.

        Returns:
            ``True`` if the registered entry contains the category,
            otherwise ``False``.
        """
        try:
            key = cls._normalize_value(
                name,
                field_name="Registry name",
            )
            category = cls._normalize_value(
                category,
                field_name="Category",
            )
        except (TypeError, ValueError):
            return False

        entry = cls._registry.get(key)

        if entry is None:
            return False

        return category in entry.categories

    @classmethod
    def find_by_class(
        cls,
        target_cls: Type[Any],
    ) -> List[str]:
        """
        Return all registration names associated with a class.

        This is useful when the same class has been registered under multiple
        aliases.

        Args:
            target_cls:
                Class to search for.

        Returns:
            A sorted list of names associated with ``target_cls``.
        """
        return sorted(
            name
            for name, entry in cls._registry.items()
            if entry.target_cls is target_cls
        )

    @classmethod
    def info(
        cls,
        name: str,
    ) -> Dict[str, Any]:
        """
        Return detailed information about a registered entry.

        Args:
            name:
                Registered class name or alias.

        Returns:
            A dictionary containing:

            - ``name``: normalized registry name.
            - ``class``: registered class name.
            - ``module``: module containing the registered class.
            - ``categories``: sorted category names.
            - ``metadata``: metadata supplied during registration.

        Raises:
            TypeError:
                If ``name`` is not a string.
            ValueError:
                If ``name`` is empty.
            KeyError:
                If ``name`` is not registered.
        """
        key = cls._normalize_value(
            name,
            field_name="Registry name",
        )

        try:
            entry = cls._registry[key]
        except KeyError:
            raise KeyError(
                f"{name!r} not found in {cls.__name__}"
            ) from None

        return {
            "name": key,
            "class": entry.target_cls.__name__,
            "module": entry.target_cls.__module__,
            "categories": sorted(entry.categories),
            "metadata": dict(entry.metadata),
        }

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a registered name from the factory.

        Only the specified name is removed. If the same class is registered
        under additional aliases, those aliases remain registered.

        Args:
            name:
                Registration name to remove.

        Raises:
            TypeError:
                If ``name`` is not a string.
            ValueError:
                If ``name`` is empty.
            KeyError:
                If ``name`` is not registered.
        """
        key = cls._normalize_value(
            name,
            field_name="Registry name",
        )

        if key not in cls._registry:
            raise KeyError(
                f"{name!r} is not registered in {cls.__name__}"
            )

        del cls._registry[key]

    @classmethod
    def clear(cls) -> None:
        """
        Remove all entries from this factory's registry.

        This operation affects only the current factory subclass and does not
        modify registries belonging to other ``BaseFactory`` subclasses.
        """
        cls._registry.clear()
