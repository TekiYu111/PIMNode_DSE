from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml


YamlLike = Union[str, Path]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    values: List[Any]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.values, list) or len(self.values) == 0:
            raise ValueError(f"Parameter '{self.name}' must have a non-empty list of values.")


@dataclass(frozen=True)
class ConfigGroupSpec:
    name: str
    configs: List[Dict[str, Any]]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.configs, list) or len(self.configs) == 0:
            raise ValueError(f"Config group '{self.name}' must have a non-empty list of configs.")
        for idx, cfg in enumerate(self.configs):
            if not isinstance(cfg, dict) or not cfg:
                raise ValueError(f"Config group '{self.name}' config #{idx} must be a non-empty dict.")


@dataclass(frozen=True)
class StaticConstraint:
    name: str
    when: Dict[str, Any]
    require: Dict[str, Any] = field(default_factory=dict)
    require_any: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SectionSpace:
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    config_groups: Dict[str, ConfigGroupSpec] = field(default_factory=dict)

    def parameter_names(self) -> List[str]:
        return list(self.parameters.keys())

    def config_group_names(self) -> List[str]:
        return list(self.config_groups.keys())


@dataclass
class DesignSpace:
    hardware: SectionSpace
    fusion: SectionSpace
    placement: SectionSpace
    tiling: SectionSpace
    search: Dict[str, Any]
    constraints: List[StaticConstraint]
    notes: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    def expand_independent_section(self, section_name: str) -> List[Dict[str, Any]]:
        """
        Expand only value-based parameters for a section (ignores config-groups).
        Example: fusion / placement / tiling.
        """
        section = getattr(self, section_name)
        if not isinstance(section, SectionSpace):
            raise ValueError(f"Section '{section_name}' is not a SectionSpace.")

        if not section.parameters:
            return [{}]

        keys = list(section.parameters.keys())
        value_lists = [section.parameters[k].values for k in keys]
        out: List[Dict[str, Any]] = []
        for combo in product(*value_lists):
            out.append(dict(zip(keys, combo)))
        return out

    def expand_hardware(self) -> List[Dict[str, Any]]:
        """
        Expand mixed hardware space:
          - config_groups are selected as a single coupled config
          - independent parameters are cartesian-product expanded

        This matches the current design_space_v2.yaml style:
          - hardware.dram.configs
          - hardware.de.configs
          - hardware.sram.<values>
          - hardware.pe.<values>
        """
        hw = self.hardware

        # First expand all config groups.
        config_group_expansions: List[List[Tuple[str, Dict[str, Any]]]] = []
        for group_name, group in hw.config_groups.items():
            group_items = []
            for cfg in group.configs:
                group_items.append((group_name, cfg))
            config_group_expansions.append(group_items)

        if not config_group_expansions:
            config_products = [()]
        else:
            config_products = product(*config_group_expansions)

        # Then expand independent scalar/value parameters.
        if hw.parameters:
            scalar_keys = list(hw.parameters.keys())
            scalar_values = [hw.parameters[k].values for k in scalar_keys]
            scalar_products = product(*scalar_values)
        else:
            scalar_keys = []
            scalar_products = [()]

        out: List[Dict[str, Any]] = []
        for cfg_combo in config_products:
            base: Dict[str, Any] = {}
            for group_name, cfg in cfg_combo:
                base[group_name] = dict(cfg)

            for scalar_combo in scalar_products:
                item = dict(base)
                if scalar_keys:
                    for k, v in zip(scalar_keys, scalar_combo):
                        item[k] = v
                out.append(item)

        return out

    def apply_static_constraints(
        self,
        hardware_candidates: Optional[List[Dict[str, Any]]] = None,
        fusion_candidates: Optional[List[Dict[str, Any]]] = None,
        placement_candidates: Optional[List[Dict[str, Any]]] = None,
        tiling_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply static constraints section-wise in a conservative way.
        This is intentionally lightweight and does not replace Stage A dynamic pruning.

        Current behavior:
          - hardware constraints are applied to hardware candidates
          - constraints referencing other sections are left for higher-level orchestration
            because they usually need joint candidate assembly
        """
        result = {
            "hardware": hardware_candidates or [],
            "fusion": fusion_candidates or [],
            "placement": placement_candidates or [],
            "tiling": tiling_candidates or [],
        }

        if result["hardware"]:
            filtered_hw = []
            for hw in result["hardware"]:
                if self._hardware_candidate_satisfies_static_constraints(hw):
                    filtered_hw.append(hw)
            result["hardware"] = filtered_hw

        return result

    def _hardware_candidate_satisfies_static_constraints(self, hw: Dict[str, Any]) -> bool:
        """
        Check only static constraints that can be evaluated from hardware fields alone.
        Constraints involving multiple sections are deferred to orchestrator / Stage A.
        """
        for c in self.constraints:
            # We only enforce constraints whose 'require' keys are hardware-only.
            if c.require and all(k.startswith("hardware.") for k in c.require.keys()):
                # If 'when' references non-hardware sections, skip here.
                if any(not k.startswith("hardware.") for k in c.when.keys()):
                    continue
                if self._matches_prefixed_mapping(hw, c.when, prefix="hardware."):
                    if not self._matches_prefixed_mapping(hw, c.require, prefix="hardware."):
                        return False

            if c.require_any:
                # Same rule: only apply if both when/require_any are hardware-only.
                if any(not k.startswith("hardware.") for k in c.when.keys()):
                    continue
                if not all(
                    isinstance(item, dict) and all(key.startswith("hardware.") for key in item.keys())
                    for item in c.require_any
                ):
                    continue
                if self._matches_prefixed_mapping(hw, c.when, prefix="hardware."):
                    ok_any = any(self._matches_prefixed_mapping(hw, req_any, prefix="hardware.") for req_any in c.require_any)
                    if not ok_any:
                        return False

        return True

    @staticmethod
    def _matches_prefixed_mapping(candidate: Dict[str, Any], mapping: Dict[str, Any], prefix: str) -> bool:
        for full_key, expected in mapping.items():
            if not full_key.startswith(prefix):
                return False
            subkey = full_key[len(prefix):]
            actual = _lookup_dotted(candidate, subkey)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True


def load_design_space(path: YamlLike) -> DesignSpace:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("design space file must load into a mapping/dict.")

    hardware = _parse_section_space(raw.get("hardware", {}), section_name="hardware")
    fusion = _parse_section_space(raw.get("fusion", {}), section_name="fusion")
    placement = _parse_section_space(raw.get("placement", {}), section_name="placement")
    tiling = _parse_section_space(raw.get("tiling", {}), section_name="tiling")

    constraints = _parse_constraints(raw.get("constraints", []))
    notes = raw.get("notes", []) or []
    search = raw.get("search", {}) or {}

    return DesignSpace(
        hardware=hardware,
        fusion=fusion,
        placement=placement,
        tiling=tiling,
        search=search,
        constraints=constraints,
        notes=notes,
        raw=raw,
    )


def _parse_section_space(section_raw: Dict[str, Any], section_name: str) -> SectionSpace:
    if not isinstance(section_raw, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping.")

    parameters: Dict[str, ParameterSpec] = {}
    config_groups: Dict[str, ConfigGroupSpec] = {}

    for key, value in section_raw.items():
        # Non-parameter metadata (e.g., description) at section level
        if key in {"description"}:
            continue

        if not isinstance(value, dict):
            # Allow plain scalar search settings only outside SectionSpace; here we skip.
            continue

        if "configs" in value:
            description = str(value.get("description", ""))
            configs = value["configs"]
            config_groups[key] = ConfigGroupSpec(name=key, configs=configs, description=description)
            continue

        if "values" in value:
            description = str(value.get("description", ""))
            parameters[key] = ParameterSpec(name=key, values=list(value["values"]), description=description)
            continue

        # Nested non-leaf mapping: flatten one level with dotted names, recursively.
        nested = _flatten_nested_values(value, prefix=key)
        for flat_name, spec in nested.items():
            parameters[flat_name] = spec

    return SectionSpace(parameters=parameters, config_groups=config_groups)


def _flatten_nested_values(obj: Dict[str, Any], prefix: str) -> Dict[str, ParameterSpec]:
    out: Dict[str, ParameterSpec] = {}
    for key, value in obj.items():
        full = f"{prefix}.{key}"
        if not isinstance(value, dict):
            continue

        if "values" in value:
            out[full] = ParameterSpec(
                name=full,
                values=list(value["values"]),
                description=str(value.get("description", "")),
            )
        else:
            out.update(_flatten_nested_values(value, prefix=full))
    return out


def _parse_constraints(raw_constraints: Iterable[Any]) -> List[StaticConstraint]:
    constraints: List[StaticConstraint] = []
    for idx, item in enumerate(raw_constraints):
        if not isinstance(item, dict):
            raise ValueError(f"Constraint #{idx} must be a mapping.")
        constraints.append(
            StaticConstraint(
                name=str(item.get("name", f"constraint_{idx}")),
                when=dict(item.get("when", {}) or {}),
                require=dict(item.get("require", {}) or {}),
                require_any=list(item.get("require_any", []) or []),
            )
        )
    return constraints


def _lookup_dotted(mapping: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def dump_example_expansion(space: DesignSpace, max_items: int = 5) -> Dict[str, Any]:
    """
    Convenience helper for quick debugging / notebooks.
    """
    hw = space.expand_hardware()[:max_items]
    fusion = space.expand_independent_section("fusion")[:max_items]
    placement = space.expand_independent_section("placement")[:max_items]
    tiling = space.expand_independent_section("tiling")[:max_items]
    return {
        "hardware": hw,
        "fusion": fusion,
        "placement": placement,
        "tiling": tiling,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and inspect design_space.yaml.")
    parser.add_argument("path", type=str, help="Path to design_space YAML file.")
    parser.add_argument("--max-items", type=int, default=3, help="Maximum examples to print per section.")
    args = parser.parse_args()

    ds = load_design_space(args.path)
    preview = dump_example_expansion(ds, max_items=args.max_items)
    print(json.dumps(preview, indent=2, ensure_ascii=False))
