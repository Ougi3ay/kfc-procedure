
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Type, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    """
    Store information associated with a registered class.

    A registry entry contains the target class, its normalized categories,
    and any optional metadata supplied during registration.

    The dataclass is frozen so its attributes cannot be reassigned after
    creation.

    Attributes:
        target_cls:
            The class associated with this registry entry.
        categories:
            Normalized categories associated with the class.
        metadata:
            Arbitrary metadata associated with the class.

    Note:
        ``frozen=True`` prevents attribute reassignment, but does not make
        mutable objects stored inside ``metadata`` immutable.
    """

    target_cls: Type[T]
    categories: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)
