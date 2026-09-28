# ============================================================
# FILE: gismol/core/repository.py
# ============================================================
"""In-memory (optionally JSON-persisted) store of COH objects."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional

from .exceptions import RepositoryError

__all__ = ["COHRepository"]


class COHRepository:
    """A registry mapping identifiers to :class:`COHObject` instances.

    The repository keeps *references* to the very same objects you register, so
    mutations are visible to every holder.  ``save``/``load`` serialise the
    hierarchy via :meth:`COHObject.to_dict`.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._objects: Dict[str, Any] = {}
        self._tags: Dict[str, set] = {}

    # -- CRUD --------------------------------------------------------------
    def add(self, obj, obj_id: Optional[str] = None, tags: Iterable[str] = ()) -> str:
        oid = obj_id or getattr(obj, "uid", None) or str(uuid.uuid4())
        if oid in self._objects and self._objects[oid] is not obj:
            raise RepositoryError(f"identifier {oid!r} already registered")
        self._objects[oid] = obj
        try:
            obj.uid = oid
        except Exception:
            pass
        self._tags.setdefault(oid, set()).update(tags)
        return oid

    def get(self, obj_id: str):
        try:
            return self._objects[obj_id]
        except KeyError as exc:
            raise RepositoryError(f"unknown object {obj_id!r}") from exc

    def remove(self, obj_id: str):
        self._objects.pop(obj_id, None)
        self._tags.pop(obj_id, None)

    def __contains__(self, obj_id): return obj_id in self._objects

    def __len__(self): return len(self._objects)

    def __iter__(self):
        return iter(self._objects.values())

    def ids(self) -> List[str]:
        return list(self._objects)

    def all(self) -> List[Any]:
        return list(self._objects.values())

    # -- search ------------------------------------------------------------
    def find(self, name: Optional[str] = None, tag: Optional[str] = None,
             predicate: Optional[Any] = None) -> List[Any]:
        out = []
        for oid, obj in self._objects.items():
            if name is not None and getattr(obj, "name", None) != name:
                continue
            if tag is not None and tag not in self._tags.get(oid, set()):
                continue
            if predicate is not None and not predicate(obj):
                continue
            out.append(obj)
        return out

    def tags_of(self, obj_id: str) -> set:
        return set(self._tags.get(obj_id, set()))

    def tag(self, obj_id: str, *tags: str):
        self._tags.setdefault(obj_id, set()).update(tags)

    def untag(self, obj_id: str, *tags: str):
        self._tags.setdefault(obj_id, set()).difference_update(tags)

    def clear(self):
        self._objects.clear()
        self._tags.clear()

    # -- hierarchy ---------------------------------------------------------
    def hierarchy(self, obj_id: str) -> Dict[str, Any]:
        obj = self.get(obj_id)

        def walk(o):
            children = getattr(o, "children", lambda: {})()
            return {"name": getattr(o, "name", None), "children": [walk(c) for c in children.values()]}

        return walk(obj)

    def index_hierarchy(self, obj) -> List[str]:
        """Register *obj* and all of its descendants; return their ids."""
        ids = [self.add(obj)]
        for child in getattr(obj, "children", lambda: {})().values():
            ids.extend(self.index_hierarchy(child))
        return ids

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "objects": {oid: obj.to_dict() for oid, obj in self._objects.items()},
            "tags": {oid: sorted(t) for oid, t in self._tags.items()},
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str, factory=None) -> "COHRepository":
        from .coh_object import COHObject
        factory = factory or COHObject.from_dict
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        repo = cls(payload.get("name", "default"))
        for oid, data in payload.get("objects", {}).items():
            repo.add(factory(data), obj_id=oid, tags=payload.get("tags", {}).get(oid, []))
        return repo

    def __repr__(self):
        return f"COHRepository({self.name!r}, n={len(self._objects)})"
    