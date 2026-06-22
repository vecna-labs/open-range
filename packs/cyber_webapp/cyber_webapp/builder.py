"""Procedural Builder for the cyber webapp pack."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping

from openrange_pack_sdk import (
    BuildResult,
    Manifest,
    PackError,
    PackPrior,
    ProceduralBuilder,
    TaskSpec,
)

from cyber_webapp.difficulty import world_difficulty
from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.priors import default_prior
from cyber_webapp.sampling import (
    _DEFAULT_LOOT_WEIGHTS,
    _DEFAULT_VULN_KIND_WEIGHTS,
    _INTERNAL_ONLY_KINDS,
    sample_graph,
)
from cyber_webapp.vulnerabilities import CATALOG as VULN_CATALOG

_RENAMED_KEYS: Mapping[str, str] = {
    "company": 'use topology: "company"',
    "lateral_movement": 'use topology: "chain"',
    "vuln_kinds": "use vuln: {weights: {...}} (bias) or vuln: {pin: [{kind: ...}]}",
    "loot_shapes": "use loot: {db: n, file: n}",
    "recon_disclosure": 'use recon: "full" | "none"',
    "difficulty": "use instruction_tier: ...",
}
_GENERATE_MODES = (False, "vuln", "novel", "service", "world")


def _as_int(value: object, ctx: str) -> int:
    if type(value) is not int:
        raise PackError(f"{ctx} must be an integer, got {value!r}")
    return value


def _validate_vuln_kinds(kinds: object) -> None:
    for kind in kinds:
        if kind not in VULN_CATALOG:
            raise PackError(f"unknown vuln kind {kind!r}; not in the catalog")
        if kind in _INTERNAL_ONLY_KINDS:
            raise PackError(
                f"vuln kind {kind!r} is internal-only -- it is composed by topology "
                "'company'/'chain', not directly pinnable"
            )


def _apply_vuln_knob(
    vuln: object,
    topology: dict,
    kind_weights: dict,
    count_ranges: dict,
) -> None:
    if not isinstance(vuln, Mapping):
        raise PackError("vuln must be a mapping with 'weights' or 'pin'")
    weights, pin = vuln.get("weights"), vuln.get("pin")
    if (weights is None) == (pin is None):
        raise PackError("vuln takes exactly one of 'weights' or 'pin'")
    if weights is not None:
        if not isinstance(weights, Mapping):
            raise PackError("vuln.weights must be a mapping of kind -> int")
        biased = {
            str(k): _as_int(v, f"vuln.weights[{k!r}]") for k, v in weights.items()
        }
        _validate_vuln_kinds(biased)
        kind_weights["vuln_kinds"] = {**_DEFAULT_VULN_KIND_WEIGHTS, **biased}
    else:
        if not isinstance(pin, list | tuple) or not pin:
            raise PackError("vuln.pin must be a non-empty list of {kind: ...} entries")
        pinned: list[str] = []
        for entry in pin:
            if not isinstance(entry, Mapping) or "kind" not in entry:
                raise PackError("each vuln.pin entry must be a mapping with a 'kind'")
            pinned.append(str(entry["kind"]))
        if len(set(pinned)) != len(pinned):
            raise PackError("vuln.pin kinds must be distinct (it places one of each)")
        _validate_vuln_kinds(pinned)
        topology["vuln_pin"] = pinned
        count_ranges["vuln_count"] = {"min": len(pinned), "max": len(pinned)}


class WebappBuilder(ProceduralBuilder):
    def __init__(self, prior: PackPrior | None = None) -> None:
        super().__init__(prior if prior is not None else default_prior())

    def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
        prior = self._effective_prior(manifest)
        generate = manifest.get("generate", False)
        if generate not in _GENERATE_MODES:
            raise PackError(
                f"generate must be one of {_GENERATE_MODES}, got {generate!r}"
            )
        graph = sample_graph(rng, prior)
        tasks: list[TaskSpec] = []
        tasks.extend(WebappBuild().generate(graph, manifest, prior))
        tasks.extend(WebappPentest().generate(graph, manifest, prior))
        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta={
                "builder": "cyber.webapp.v2",
                "seed": self.current_seed,
                "prior_source": prior.source,
                "manifest_keys": sorted(manifest.keys()),
                "generate": generate,
                "world_difficulty": float(world_difficulty(graph)),
            },
        )

    def _effective_prior(self, manifest: Manifest) -> PackPrior:
        base = self.prior if self.prior is not None else default_prior()
        for old, hint in _RENAMED_KEYS.items():
            if old in manifest:
                raise PackError(f"manifest key {old!r} was renamed -- {hint}")

        topo = manifest.get("topology", "flat")
        if topo not in ("flat", "company", "chain"):
            raise PackError(f"topology must be flat|company|chain, got {topo!r}")
        company = topo in ("company", "chain")
        lateral = topo == "chain"
        scale = manifest.get("scale")
        vuln = manifest.get("vuln")
        loot = manifest.get("loot")
        chain = manifest.get("chain")
        recon = manifest.get("recon")

        if company and (vuln is not None or loot is not None):
            raise PackError(
                f"topology {topo!r} forces its vuln/loot shape; "
                "set vuln/loot only on topology 'flat'"
            )
        if recon is not None and not company:
            raise PackError("recon applies only to topology 'company' or 'chain'")
        if chain is not None and not lateral:
            raise PackError("chain applies only to topology 'chain'")
        no_knobs = all(x is None for x in (scale, vuln, loot, chain, recon))
        if not company and no_knobs:
            return base

        topology = dict(base.topology)
        count_ranges = dict(topology.get("count_ranges") or {})
        kind_weights = dict(topology.get("kind_weights") or {})

        if company:
            topology["preset"] = "company"
            if lateral:
                topology["lateral"] = True
            count_ranges.setdefault("service_count", {"min": 6, "max": 8})
            count_ranges.setdefault("vuln_count", {"min": 3, "max": 6})
            kind_weights["loot_shapes"] = {"db": 1, "file": 0}
            kind_weights["vuln_kinds"] = {"ssrf": 1, "path_traversal": 3, "xxe": 2}
        if recon is not None:
            if recon not in ("full", "none"):
                raise PackError(f"recon must be 'full' or 'none', got {recon!r}")
            topology["recon_disclosure"] = recon
        if scale is not None:
            if not isinstance(scale, Mapping):
                raise PackError("scale must be a mapping of count-key -> {min, max}")
            for key, spec in scale.items():
                if not isinstance(spec, Mapping):
                    raise PackError(f"scale[{key!r}] must be a {{min, max}} mapping")
                lo = _as_int(spec.get("min"), f"scale[{key!r}].min")
                hi = _as_int(spec.get("max"), f"scale[{key!r}].max")
                if not 0 <= lo <= hi:
                    raise PackError(
                        f"scale[{key!r}] needs 0 <= min <= max, got {lo}..{hi}"
                    )
                count_ranges[str(key)] = {"min": lo, "max": hi}
        if chain is not None:
            depth = chain.get("depth") if isinstance(chain, Mapping) else None
            if not isinstance(depth, Mapping):
                raise PackError("chain must be {depth: {min: int, max: int}}")
            lo = _as_int(depth.get("min"), "chain.depth.min")
            hi = _as_int(depth.get("max"), "chain.depth.max")
            if not 1 <= lo <= hi:
                raise PackError(f"chain.depth needs 1 <= min <= max, got {lo}..{hi}")
            topology["chain_depth"] = {"min": lo, "max": hi}
        if vuln is not None:
            _apply_vuln_knob(vuln, topology, kind_weights, count_ranges)
        if loot is not None:
            if not isinstance(loot, Mapping):
                raise PackError("loot must be a mapping like {db: int, file: int}")
            kind_weights["loot_shapes"] = {
                **_DEFAULT_LOOT_WEIGHTS,
                **{str(k): _as_int(v, f"loot[{k!r}]") for k, v in loot.items()},
            }
        if count_ranges:
            topology["count_ranges"] = count_ranges
        if kind_weights:
            topology["kind_weights"] = kind_weights
        return dataclasses.replace(base, topology=topology)
