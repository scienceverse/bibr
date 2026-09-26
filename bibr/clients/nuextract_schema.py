"""Native-only schema projection for NuExtract's template protocol."""

from __future__ import annotations

import copy
import functools
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel


@dataclass(frozen=True)
class NuExtractSchemaPolicy:
    """Native-only field policy attached to a Pydantic response model."""

    exclude: frozenset[str] = frozenset()
    semantics: Mapping[str, str] = field(default_factory=dict)
    choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeSchemaContract:
    """NuExtract template plus the strict JSON schema for its completion."""

    response_model: type[BaseModel]
    template: dict[str, Any]
    wire_schema: dict[str, Any]
    nullable_paths: frozenset[str]
    # Whether the builder already ran the static wire-shape check below.
    # Builder-made contracts skip the per-parse re-check; hand-assembled
    # ones (tests) still pay it.
    shape_checked: bool = False


class _UnsupportedWireSchema(ValueError):
    """Internal sentinel for a wire/template shape we cannot validate safely."""


@dataclass
class _CollectedPolicy:
    exclude: set[str] = field(default_factory=set)
    semantics: dict[str, str] = field(default_factory=dict)
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _nested_models(annotation: Any, suffix: str = "") -> list[tuple[type[BaseModel], str]]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _nested_models(get_args(annotation)[0], suffix)
    if origin in (Union, types.UnionType):
        nested: list[tuple[type[BaseModel], str]] = []
        for member in get_args(annotation):
            if member is not type(None):
                nested.extend(_nested_models(member, suffix))
        return nested
    if origin in (list, set, tuple, frozenset):
        args = get_args(annotation)
        return _nested_models(args[0], f"{suffix}[]") if args else []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [(annotation, suffix)]
    return []


def _merge_policy_value(
    target: dict[str, Any],
    *,
    path: str,
    value: Any,
    kind: str,
) -> None:
    if path in target and target[path] != value:
        raise ValueError(f"Conflicting NuExtract {kind} policies at {path!r}")
    target[path] = value


def _validation_property_name(field_name: str, model_field: Any) -> str:
    validation_alias = model_field.validation_alias
    if isinstance(validation_alias, str):
        return validation_alias
    if isinstance(validation_alias, AliasChoices):
        for choice in validation_alias.choices:
            if isinstance(choice, str):
                return choice
        return field_name
    if isinstance(validation_alias, AliasPath):
        return field_name
    if validation_alias is not None:
        raise ValueError(
            f"Unsupported Pydantic validation alias for NuExtract field {field_name!r}: "
            f"{validation_alias!r}"
        )
    return model_field.alias if isinstance(model_field.alias, str) else field_name


def _collect_model_policies(
    model: type[BaseModel],
    *,
    prefix: str = "",
    stack: tuple[type[BaseModel], ...] = (),
    collected: _CollectedPolicy | None = None,
) -> _CollectedPolicy:
    if collected is None:
        collected = _CollectedPolicy()
    if model in stack:
        chain = " -> ".join(item.__name__ for item in (*stack, model))
        raise ValueError(f"Cyclic Pydantic model graph in NuExtract contract: {chain}")

    raw_policy = model.__dict__.get("nuextract_policy")
    if raw_policy is not None:
        if not isinstance(raw_policy, NuExtractSchemaPolicy):
            raise ValueError(f"{model.__name__}.nuextract_policy has an invalid type")
        overlap = set(raw_policy.semantics) & set(raw_policy.choices)
        if overlap:
            paths = ", ".join(sorted(overlap))
            raise ValueError(f"NuExtract paths have both semantics and choices: {paths}")
        for local_path in raw_policy.exclude:
            collected.exclude.add(_join_path(prefix, local_path))
        for local_path, semantic in raw_policy.semantics.items():
            _merge_policy_value(
                collected.semantics,
                path=_join_path(prefix, local_path),
                value=semantic,
                kind="semantic",
            )
        for local_path, choices in raw_policy.choices.items():
            _merge_policy_value(
                collected.choices,
                path=_join_path(prefix, local_path),
                value=tuple(choices),
                kind="choice",
            )

    next_stack = (*stack, model)
    for field_name, model_field in model.model_fields.items():
        property_name = _validation_property_name(field_name, model_field)
        field_prefix = _join_path(prefix, property_name)
        for nested_model, suffix in _nested_models(model_field.annotation):
            _collect_model_policies(
                nested_model,
                prefix=f"{field_prefix}{suffix}",
                stack=next_stack,
                collected=collected,
            )
    return collected


def _resolve_json_pointer(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported $ref {ref!r}; only local refs are allowed")
    current: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Could not resolve $ref {ref!r}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"$ref {ref!r} does not resolve to a schema object")
    return current


def _dereference(
    node: Any,
    *,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...] = (),
) -> Any:
    if isinstance(node, list):
        return [_dereference(item, root_schema=root_schema, ref_stack=ref_stack) for item in node]
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    if "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str):
            raise ValueError("$ref must be a string")
        if ref in ref_stack:
            raise ValueError(f"Cyclic $ref detected: {ref!r}")
        merged = {
            **copy.deepcopy(_resolve_json_pointer(ref, root_schema)),
            **{key: copy.deepcopy(value) for key, value in node.items() if key != "$ref"},
        }
        return _dereference(
            merged,
            root_schema=root_schema,
            ref_stack=(*ref_stack, ref),
        )
    return {
        key: _dereference(value, root_schema=root_schema, ref_stack=ref_stack)
        for key, value in node.items()
        if key != "$defs"
    }


def _nullable_branch(node: dict[str, Any]) -> dict[str, Any] | None:
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        non_null = [
            branch
            for branch in any_of
            if not (isinstance(branch, dict) and branch.get("type") == "null")
        ]
        if len(non_null) == 1 and len(non_null) != len(any_of) and isinstance(non_null[0], dict):
            return non_null[0]
    node_type = node.get("type")
    if isinstance(node_type, list):
        non_null_types = [value for value in node_type if value != "null"]
        if len(non_null_types) == 1 and len(non_null_types) != len(node_type):
            branch = copy.deepcopy(node)
            branch["type"] = non_null_types[0]
            return branch
    return None


def _non_null_type(node: dict[str, Any]) -> str | None:
    node_type = node.get("type")
    if isinstance(node_type, str):
        return node_type
    if isinstance(node_type, list):
        non_null_types = [value for value in node_type if value != "null"]
        if len(non_null_types) == 1 and len(non_null_types) != len(node_type):
            return non_null_types[0]
    return None


def _collect_nullable_paths(
    node: dict[str, Any],
    *,
    path: str = "",
    paths: set[str] | None = None,
) -> set[str]:
    if paths is None:
        paths = set()
    nullable = _nullable_branch(node)
    if nullable is not None:
        paths.add(path)
        _collect_nullable_paths(nullable, path=path, paths=paths)
        return paths
    if node.get("type") == "object" or "properties" in node:
        for name, child in node.get("properties", {}).items():
            _collect_nullable_paths(child, path=_join_path(path, name), paths=paths)
    elif node.get("type") == "array" and isinstance(node.get("items"), dict):
        _collect_nullable_paths(node["items"], path=f"{path}[]", paths=paths)
    return paths


def _path_tokens(path: str) -> list[str]:
    if not path:
        raise ValueError("NuExtract policy paths may not be empty")
    tokens: list[str] = []
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid NuExtract policy path {path!r}")
        while part.endswith("[]"):
            base = part[:-2]
            if base:
                tokens.append(base)
            tokens.append("[]")
            part = ""
        if part:
            tokens.append(part)
    return tokens


def _for_navigation(node: dict[str, Any], path: str) -> dict[str, Any]:
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        nullable = _nullable_branch(node)
        if nullable is not None:
            return nullable
        raise ValueError(f"Unsupported union while resolving NuExtract policy path {path!r}")
    if isinstance(node.get("type"), list):
        if _non_null_type(node) is not None:
            return node
        raise ValueError(f"Unsupported union while resolving NuExtract policy path {path!r}")
    return node


def _locate_node(schema: dict[str, Any], path: str) -> dict[str, Any]:
    current = schema
    for segment in _path_tokens(path):
        current = _for_navigation(current, path)
        if segment == "[]":
            if _non_null_type(current) != "array" or not isinstance(current.get("items"), dict):
                raise ValueError(f"NuExtract policy path {path!r} does not select an array item")
            current = current["items"]
            continue
        properties = current.get("properties")
        if not isinstance(properties, dict) or segment not in properties:
            raise ValueError(f"NuExtract policy path {path!r} is missing at {segment!r}")
        current = properties[segment]
    return _for_navigation(current, path)


def _exclude_property(schema: dict[str, Any], path: str) -> None:
    tokens = _path_tokens(path)
    if tokens[-1] == "[]":
        raise ValueError(f"NuExtract exclusion {path!r} must select an object property")
    current = schema
    for segment in tokens[:-1]:
        current = _for_navigation(current, path)
        if segment == "[]":
            if _non_null_type(current) != "array" or not isinstance(current.get("items"), dict):
                raise ValueError(f"NuExtract exclusion {path!r} does not select an array item")
            current = current["items"]
        else:
            properties = current.get("properties")
            if not isinstance(properties, dict) or segment not in properties:
                raise ValueError(f"NuExtract exclusion path {path!r} is missing")
            current = properties[segment]
    current = _for_navigation(current, path)
    properties = current.get("properties")
    name = tokens[-1]
    if not isinstance(properties, dict) or name not in properties:
        raise ValueError(f"NuExtract exclusion path {path!r} is missing")
    del properties[name]
    required = current.get("required")
    if isinstance(required, list):
        current["required"] = [field_name for field_name in required if field_name != name]


def _ensure_string_leaf(node: dict[str, Any], path: str) -> None:
    node_type = node.get("type")
    is_nullable_string = (
        isinstance(node_type, list) and set(node_type) == {"string", "null"} and len(node_type) == 2
    )
    if (
        (node_type != "string" and not is_nullable_string)
        or "properties" in node
        or "items" in node
    ):
        kind = node.get("type", "object" if "properties" in node else "unknown")
        raise ValueError(f"NuExtract policy path {path!r} selects a {kind} node, not a string leaf")


def _apply_policies(schema: dict[str, Any], policies: _CollectedPolicy) -> None:
    for path in sorted(policies.exclude):
        _exclude_property(schema, path)
    for path, semantic in policies.semantics.items():
        node = _locate_node(schema, path)
        _ensure_string_leaf(node, path)
        node.pop("enum", None)
        node.pop("format", None)
        node.pop("x-verbatim", None)
        if semantic == "verbatim-string":
            node["x-verbatim"] = True
        elif semantic != "string":
            node["format"] = semantic
    for path, choices in policies.choices.items():
        if len(choices) < 2 or not all(isinstance(choice, str) for choice in choices):
            raise ValueError(f"NuExtract choices at {path!r} require at least two strings")
        node = _locate_node(schema, path)
        _ensure_string_leaf(node, path)
        node.pop("format", None)
        node.pop("x-verbatim", None)
        node["enum"] = list(choices)


def _without_nulls(node: Any) -> Any:
    if isinstance(node, list):
        return [_without_nulls(item) for item in node]
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    nullable = _nullable_branch(node)
    if nullable is not None:
        return _without_nulls(nullable)
    return {key: _without_nulls(value) for key, value in node.items()}


def _remove_wire_semantics(schema: dict[str, Any], semantic_paths: Mapping[str, str]) -> None:
    for path in semantic_paths:
        node = _locate_node(schema, path)
        node.pop("format", None)
        node.pop("x-verbatim", None)


def _close_wire_schema(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _close_wire_schema(item)
        return
    if not isinstance(node, dict):
        return
    node_type = _non_null_type(node)
    if node_type == "object" or (node_type is None and "properties" in node):
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("Invalid object schema properties")
        if "type" not in node:
            node["type"] = "object"
        node["required"] = list(properties)
        node["additionalProperties"] = False
        for child in properties.values():
            _close_wire_schema(child)
    elif node_type == "array":
        _close_wire_schema(node.get("items"))
    for branch in node.get("anyOf", []):
        _close_wire_schema(branch)


def _schema_leaf_map(
    node: dict[str, Any],
    *,
    path: str = "",
) -> dict[str, str | list[str]]:
    if "anyOf" in node:
        raise ValueError(f"unsupported union in projected NuExtract schema at {path!r}")
    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"Empty or unconstrained object in NuExtract schema at {path!r}")
        flattened: dict[str, str | list[str]] = {}
        for name, child in properties.items():
            flattened.update(_schema_leaf_map(child, path=_join_path(path, name)))
        return flattened
    if node.get("type") == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise ValueError(f"Array without item schema in NuExtract schema at {path!r}")
        return _schema_leaf_map(items, path=f"{path}[]")
    enum = node.get("enum")
    if enum is not None:
        if (
            not isinstance(enum, list)
            or len(enum) < 2
            or not all(isinstance(value, str) for value in enum)
        ):
            raise ValueError(f"Invalid enum in NuExtract schema at {path!r}")
        return {path: enum}
    node_type = node.get("type")
    if not isinstance(node_type, str) or node_type == "null":
        raise ValueError(f"Invalid NuExtract schema leaf at {path!r}")
    decoded = node.get("format", node_type)
    if not isinstance(decoded, str):
        raise ValueError(f"Invalid NuExtract semantic at {path!r}")
    if node.get("x-verbatim"):
        decoded = f"verbatim-{decoded}"
    return {path: decoded}


def _template_leaf_map(value: Any, *, path: str = "") -> dict[str, str | list[str]]:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"Empty NuExtract template object at {path!r}")
        flattened: dict[str, str | list[str]] = {}
        for name, child in value.items():
            flattened.update(_template_leaf_map(child, path=_join_path(path, name)))
        return flattened
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty NuExtract template list at {path!r}")
        if len(value) > 1 and all(isinstance(item, str) for item in value):
            return {path: value}
        if len(value) != 1:
            raise ValueError(f"Invalid NuExtract template list at {path!r}")
        return _template_leaf_map(value[0], path=f"{path}[]")
    if not isinstance(value, str):
        raise ValueError(f"Invalid NuExtract template leaf at {path!r}")
    return {path: value}


def _wire_non_null_branch(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return one supported non-null branch and whether null is allowed."""
    if "anyOf" in schema:
        any_of = schema["anyOf"]
        if not isinstance(any_of, list):
            raise _UnsupportedWireSchema
        null_branches = [
            branch for branch in any_of if isinstance(branch, dict) and branch.get("type") == "null"
        ]
        non_null_branches = [
            branch
            for branch in any_of
            if not (isinstance(branch, dict) and branch.get("type") == "null")
        ]
        if (
            len(null_branches) != 1
            or len(non_null_branches) != 1
            or not isinstance(non_null_branches[0], dict)
        ):
            raise _UnsupportedWireSchema
        return non_null_branches[0], True

    node_type = schema.get("type")
    if isinstance(node_type, list):
        if (
            len(node_type) != 2
            or "null" not in node_type
            or not all(isinstance(value, str) for value in node_type)
        ):
            raise _UnsupportedWireSchema
        non_null_types = [value for value in node_type if value != "null"]
        if len(non_null_types) != 1:
            raise _UnsupportedWireSchema
        non_null_schema = copy.deepcopy(schema)
        non_null_schema["type"] = non_null_types[0]
        return non_null_schema, True

    return schema, False


def _wire_shape_is_supported(schema: dict[str, Any], template: Any) -> bool:
    try:
        schema, _ = _wire_non_null_branch(schema)
    except _UnsupportedWireSchema:
        return False

    node_type = schema.get("type")
    enum = schema.get("enum")
    if enum is not None:
        return (
            node_type == "string"
            and isinstance(enum, list)
            and len(enum) >= 2
            and all(isinstance(choice, str) for choice in enum)
            and isinstance(template, list)
            and template == enum
        )

    if node_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if (
            not isinstance(properties, dict)
            or not all(isinstance(name, str) for name in properties)
            or not isinstance(template, dict)
            or not all(isinstance(name, str) for name in template)
            or set(properties) != set(template)
            or not isinstance(required, list)
            or len(required) != len(properties)
            or not all(isinstance(name, str) for name in required)
            or set(required) != set(properties)
            or schema.get("additionalProperties") is not False
        ):
            return False
        return all(
            isinstance(child_schema, dict)
            and _wire_shape_is_supported(child_schema, template[name])
            for name, child_schema in properties.items()
        )

    if node_type == "array":
        items = schema.get("items")
        return (
            isinstance(items, dict)
            and isinstance(template, list)
            and len(template) == 1
            and _wire_shape_is_supported(items, template[0])
        )

    return node_type in {"string", "integer", "number", "boolean"} and isinstance(template, str)


def _wire_value_matches(value: Any, schema: dict[str, Any], template: Any) -> bool:
    try:
        schema, nullable = _wire_non_null_branch(schema)
    except _UnsupportedWireSchema:
        return False

    if value is None:
        return nullable

    enum = schema.get("enum")
    # Enum *membership* is deliberately not checked here. It is semantics, and
    # the model layer already owns it: a ``mode="before"`` validator maps
    # 'dissertation' onto 'thesis' and anything unrecognised onto 'other'.
    # Rejecting at the wire throws away the entire completion — every other
    # field of every other item in the batch — over one word the next layer
    # would have repaired. The shape is still enforced: an enum slot declared
    # as a string must carry a string.
    if enum is not None and type(value) is not str:
        return False

    node_type = schema.get("type")
    if node_type == "object":
        if type(value) is not dict:
            return False
        properties = schema.get("properties")
        if (
            not isinstance(properties, dict)
            or not isinstance(template, dict)
            or set(value) != set(properties)
            or set(template) != set(properties)
        ):
            return False
        return all(
            isinstance(child_schema, dict)
            and _wire_value_matches(value[name], child_schema, template[name])
            for name, child_schema in properties.items()
        )

    if node_type == "array":
        items = schema.get("items")
        if (
            type(value) is not list
            or not isinstance(items, dict)
            or not isinstance(template, list)
            or len(template) != 1
        ):
            return False
        return all(_wire_value_matches(item, items, template[0]) for item in value)

    if node_type == "string":
        if type(value) is not str:
            # A bare number is tolerated only under the plain "string" tag.
            # Those are the short printed locators — page, volume, issue —
            # typed as strings because "S13"/"e12345"/"iii" are real values,
            # which means a range that really is 100-115 invites the model to
            # emit 100 unquoted; the model layer stringifies it. Every other
            # tag ("verbatim-string", "date", "url") marks a span copied out
            # of the document, where a bare number is not a quoting slip but
            # a wrong answer, so it stays rejected. ``bool`` is an ``int``
            # subclass and is excluded by the exact type checks — a stray
            # ``true`` is not a page number.
            if template != "string":
                return False
            if type(value) is not int and type(value) is not float:
                return False
            if type(value) is float and not math.isfinite(value):
                return False
    elif node_type == "integer":
        if type(value) is not int:
            return False
    elif node_type == "number":
        if type(value) is float and not math.isfinite(value):
            return False
        if type(value) is not int and type(value) is not float:
            return False
    elif node_type == "boolean":
        if type(value) is not bool:
            return False
    else:
        return False

    if enum is not None:
        return isinstance(template, list) and template == enum
    return isinstance(template, str) and value != template


def native_wire_value_is_valid(
    value: Any,
    *,
    wire_schema: dict[str, Any],
    template: dict[str, Any],
) -> bool:
    """Validate one decoded completion against the exact native wire contract.

    This deliberately implements only the closed schema forms emitted by
    :func:`native_contract_for_model`. Unknown unions and shapes fail closed.
    Semantic formats are template instructions, not lexical wire constraints.
    """
    if not isinstance(wire_schema, dict) or not isinstance(template, dict):
        return False
    try:
        if not _wire_shape_is_supported(wire_schema, template):
            return False
        return _wire_value_matches(value, wire_schema, template)
    except Exception:
        return False


def contract_wire_value_is_valid(value: Any, *, contract: NativeSchemaContract) -> bool:
    """Validate one decoded completion against a builder-made contract.

    The builder checks the static wire shape once (recorded on the
    contract), so the per-parse path only matches the value. Hand-assembled
    contracts still pay the shape check, exactly as
    :func:`native_wire_value_is_valid` would.
    """
    if not isinstance(contract.wire_schema, dict) or not isinstance(contract.template, dict):
        return False
    try:
        if not contract.shape_checked and not _wire_shape_is_supported(
            contract.wire_schema, contract.template
        ):
            return False
        return _wire_value_matches(value, contract.wire_schema, contract.template)
    except Exception:
        return False


def _build_native_contract(response_model: type[BaseModel]) -> NativeSchemaContract:
    """Build a faithful, strict NuExtract contract without mutating the model schema."""
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise TypeError("response_model must be a Pydantic BaseModel subclass")

    source_schema = copy.deepcopy(response_model.model_json_schema())
    policies = _collect_model_policies(response_model)
    overlap = set(policies.semantics) & set(policies.choices)
    if overlap:
        paths = ", ".join(sorted(overlap))
        raise ValueError(f"NuExtract paths have both semantics and choices: {paths}")
    projected = _dereference(source_schema, root_schema=source_schema)
    nullable_paths = _collect_nullable_paths(projected)
    _apply_policies(projected, policies)
    excluded_nullable_paths = {
        nullable_path
        for nullable_path in nullable_paths
        if any(
            nullable_path == excluded
            or nullable_path.startswith(f"{excluded}.")
            or nullable_path.startswith(f"{excluded}[]")
            for excluded in policies.exclude
        )
    }
    nullable_paths.difference_update(excluded_nullable_paths)

    converter_schema = _without_nulls(copy.deepcopy(projected))
    expected_leaves = _schema_leaf_map(converter_schema)

    wire_schema = copy.deepcopy(projected)
    _remove_wire_semantics(wire_schema, policies.semantics)
    _close_wire_schema(wire_schema)

    from numind.nuextract_utils import convert_json_schema_to_nuextract_template

    template, dropped_branches = convert_json_schema_to_nuextract_template(
        converter_schema,
        omit_unsupported_branches=True,
    )
    if dropped_branches:
        raise ValueError(
            f"NuExtract converter dropped schema branches for {response_model.__name__}: "
            f"{dropped_branches}"
        )
    if not isinstance(template, dict):
        raise ValueError(f"{response_model.__name__} did not render to an object template")
    actual_leaves = _template_leaf_map(template)
    if actual_leaves != expected_leaves:
        raise ValueError(
            f"NuExtract template leaf mismatch for {response_model.__name__}: "
            f"expected={expected_leaves!r}, actual={actual_leaves!r}"
        )

    return NativeSchemaContract(
        response_model=response_model,
        template=copy.deepcopy(template),
        wire_schema=wire_schema,
        nullable_paths=frozenset(nullable_paths),
        shape_checked=_wire_shape_is_supported(wire_schema, template),
    )


@functools.cache
def _cached_native_contract(response_model: type[BaseModel]) -> NativeSchemaContract:
    """One built contract per response model; callers get copies, never this."""
    return _build_native_contract(response_model)


def native_contract_for_model(response_model: type[BaseModel]) -> NativeSchemaContract:
    """Return the NuExtract contract for *response_model*, memoized per model.

    The expensive projection (schema dereference plus the numind converter)
    runs once per model; every call returns an independent deep copy, so
    mutating the result never affects later calls.
    """
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise TypeError("response_model must be a Pydantic BaseModel subclass")
    return copy.deepcopy(_cached_native_contract(response_model))
