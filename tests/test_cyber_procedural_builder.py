"""Tests for the procedural cyber :class:`WebappBuilder`.

Covers:

  - ``WebappBuilder(prior)`` produces a :class:`BuildResult` whose
    graph carries the expected node kinds.
  - Determinism in ``manifest["seed"]``: same seed → same
    ``content_hash()``.
  - Different seeds produce different content hashes.
  - ``repair`` resamples with a perturbed seed.
  - End-to-end ``admit(pack, manifest)`` succeeds against
    :class:`WebappPack`.
  - :func:`default_prior` flows when ``WebappBuilder(prior=None)``.
"""

from __future__ import annotations

from cyber_webapp import WebappBuilder, WebappPack
from cyber_webapp.priors import default_prior
from openrange_pack_sdk import BuildResult, PackPrior, Snapshot

from openrange.core.admit import admit


def test_builder_returns_build_result() -> None:
    """``WebappBuilder.build`` returns a fully-populated ``BuildResult``."""
    builder = WebappBuilder(default_prior())
    result = builder.build({"seed": 0})
    assert isinstance(result, BuildResult)
    assert result.graph.ontology == "cyber.webapp@v2"
    # Both task families contribute.
    assert len(result.tasks) >= 2
    assert {t.feasibility_check for t in result.tasks} == {
        "webapp.build",
        "webapp.pentest",
    }


def test_builder_graph_carries_expected_kinds() -> None:
    """A seeded build emits every node kind a fully-shaped webapp needs.

    The sampler always lays down at least one of each core kind so the
    invariants pass and both families can bind tasks. The set we assert
    on is the intersection of "always present" kinds across the
    sampler — ``account`` / ``credential`` are sampled in batches >= 1,
    ``network`` is single-instance, etc.
    """
    builder = WebappBuilder(default_prior())
    result = builder.build({"seed": 0})
    kinds = {n.kind for n in result.graph.nodes.values()}
    expected = {
        "host",
        "service",
        "endpoint",
        "data_store",
        "record",
        "secret",
        "vulnerability",
        "account",
        "credential",
        "network",
    }
    missing = expected - kinds
    assert not missing, f"seed 0 missing kinds: {missing} (got {kinds})"


def test_builder_admission_meta_carries_provenance() -> None:
    """``BuildResult.admission_meta`` records builder id + seed + prior."""
    builder = WebappBuilder(default_prior())
    result = builder.build({"seed": 3})
    meta = dict(result.admission_meta)
    assert meta["builder"] == "cyber.webapp.v2"
    assert meta["seed"] == 3
    assert "prior_source" in meta


def test_same_seed_same_content_hash() -> None:
    """Identical ``manifest["seed"]`` → identical graph content hash.

    Two fresh builders consume the same seed and emit graphs whose
    ``content_hash()`` is byte-identical. This is the property that
    powers the content-addressed snapshot id.
    """
    builder_a = WebappBuilder(default_prior())
    builder_b = WebappBuilder(default_prior())
    hash_a = builder_a.build({"seed": 11}).graph.content_hash()
    hash_b = builder_b.build({"seed": 11}).graph.content_hash()
    assert hash_a == hash_b


def test_different_seeds_different_content_hashes() -> None:
    """Different seeds → different content hashes (worlds diverge)."""
    builder = WebappBuilder(default_prior())
    hash_a = builder.build({"seed": 0}).graph.content_hash()
    # Reset attempts by using a fresh builder for the second seed.
    hash_b = WebappBuilder(default_prior()).build({"seed": 42}).graph.content_hash()
    assert hash_a != hash_b


def test_seed_sweep_yields_distinct_flag_values() -> None:
    """Sweeping seeds yields graphs with distinct embedded flag values.

    The sampler picks a fresh ``flag`` secret value per build; over a
    sweep of seeds we should see at least two different values. This is
    the procedural-variety contract.
    """
    flags: set[str] = set()
    for seed in range(5):
        result = WebappBuilder(default_prior()).build({"seed": seed})
        secrets = [n for n in result.graph.nodes.values() if n.kind == "secret"]
        # Every build has at least one flag secret.
        assert secrets, f"seed {seed}: no secret node"
        flags.add(str(secrets[0].attrs["value_ref"]))
    assert len(flags) >= 2, f"seed sweep produced only one flag: {flags}"


def test_missing_seed_falls_back_to_zero() -> None:
    """A manifest without ``seed`` builds deterministically against seed=0."""
    explicit = WebappBuilder(default_prior()).build({"seed": 0}).graph.content_hash()
    default = WebappBuilder(default_prior()).build({}).graph.content_hash()
    assert explicit == default


def test_repair_resamples_a_different_graph() -> None:
    """``repair`` perturbs the seed: the follow-up build differs.

    The repair policy is a perturbed-seed resample; the v1 attempt
    counter ticks so each ``repair → build`` cycle samples a fresh
    world. Two consecutive builds from the same builder (with a repair
    in between) must therefore differ.
    """
    builder = WebappBuilder(default_prior())
    first = builder.build({"seed": 0})
    repaired = builder.repair(first, errors=[], infeasible=[])
    assert isinstance(repaired, BuildResult)
    assert first.graph.content_hash() != repaired.graph.content_hash()


def test_repair_preserves_pentest_family() -> None:
    """Repair must still produce a pentest task — that's the family the
    infeasible report targeted. Build is conditional on world shape (api
    service present) and may legitimately not appear in either result."""
    builder = WebappBuilder(default_prior())
    first = builder.build({"seed": 0})
    repaired = builder.repair(first, errors=[], infeasible=["webapp.pentest.0"])
    repaired_families = {t.feasibility_check for t in repaired.tasks}
    assert "webapp.pentest" in repaired_families, repaired_families


def test_builder_falls_back_to_default_prior_when_none() -> None:
    """``WebappBuilder(prior=None)`` still admits — uses the default prior."""
    builder = WebappBuilder(prior=None)
    result = builder.build({"seed": 0})
    assert "hand-authored" in str(result.admission_meta["prior_source"])


def test_pack_make_builder_returns_webapp_builder() -> None:
    """``WebappPack.make_builder(prior=None)`` constructs a ``WebappBuilder``."""
    pack = WebappPack()
    builder = pack.make_builder(prior=None)
    assert isinstance(builder, WebappBuilder)


def test_pack_make_builder_threads_caller_supplied_prior() -> None:
    """A caller-supplied ``PackPrior`` reaches the builder verbatim."""
    pack = WebappPack()
    prior: PackPrior = default_prior()
    # Tag the prior with a recognisable source so we can confirm it's
    # the one threaded into admission_meta.
    prior.source = "test-marker :: external"
    builder = pack.make_builder(prior)
    result = builder.build({"seed": 0})
    assert result.admission_meta["prior_source"] == "test-marker :: external"


def test_builder_world_admits_through_admit() -> None:
    """The builder's world graph passes every admission layer.

    Structural + ontology + pack invariants + task-binding + task
    feasibility — all five gates must be satisfied for ``admit`` to
    return a ``Snapshot`` rather than an ``AdmissionFailure``.
    """
    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0}, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    # The procedural sampler emits a non-trivial graph.
    assert len(snap.graph.nodes) >= 10
    assert len(snap.graph.edges) >= 8


def test_admission_yields_both_task_families() -> None:
    """An admitted snapshot carries one task per family."""
    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0}, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    families = {t.feasibility_check for t in snap.tasks}
    assert families == {"webapp.build", "webapp.pentest"}
    # Different entrypoint kinds — build family entrypoints a service,
    # pentest family entrypoints an endpoint.
    entrypoint_kinds = {snap.graph.nodes[t.entrypoints[0]].kind for t in snap.tasks}
    assert entrypoint_kinds == {"service", "endpoint"}


# A manifest scale override that widens the sampler's count ranges well
# past the default 2-5 service band.
SCALE_UP = {
    "service_count": {"min": 8, "max": 10},
    "vuln_count": {"min": 4, "max": 6},
}


def _service_count(result: BuildResult) -> int:
    return sum(1 for n in result.graph.nodes.values() if n.kind == "service")


def test_manifest_scale_grows_the_world() -> None:
    """``manifest["scale"]`` widens the sampler count ranges, so the world
    carries more services than the default 2-5 band — scale from the
    manifest, no hand-built ``PackPrior``."""
    default = WebappBuilder(default_prior()).build({"seed": 5})
    scaled = WebappBuilder(default_prior()).build({"seed": 5, "scale": SCALE_UP})
    assert _service_count(scaled) > _service_count(default)
    assert _service_count(scaled) >= 8


def test_manifest_scale_preserves_determinism() -> None:
    """Same scale + same seed → identical content hash; a scaled world
    diverges from the default-scale world. Scaling stays reproducible."""
    a = WebappBuilder(default_prior()).build({"seed": 5, "scale": SCALE_UP})
    b = WebappBuilder(default_prior()).build({"seed": 5, "scale": SCALE_UP})
    base = WebappBuilder(default_prior()).build({"seed": 5})
    assert a.graph.content_hash() == b.graph.content_hash()
    assert a.graph.content_hash() != base.graph.content_hash()


def test_absent_scale_is_a_noop() -> None:
    """Omitting ``scale`` (or passing an empty mapping) leaves the world
    byte-identical to a build that never mentioned scale."""
    plain = WebappBuilder(default_prior()).build({"seed": 7}).graph.content_hash()
    empty = (
        WebappBuilder(default_prior())
        .build({"seed": 7, "scale": {}})
        .graph.content_hash()
    )
    assert plain == empty


def test_scale_ignores_non_mapping_entries() -> None:
    """A non-mapping ``scale`` value is skipped, not crashed on — the
    world falls back to the default range for that key."""
    plain = WebappBuilder(default_prior()).build({"seed": 7}).graph.content_hash()
    junk = (
        WebappBuilder(default_prior())
        .build({"seed": 7, "scale": {"service_count": "lots"}})
        .graph.content_hash()
    )
    assert plain == junk


def test_scaled_world_still_admits() -> None:
    """A scaled-up world passes all five admission layers — solvable by
    construction holds at larger scale, not just the default band."""
    snap = admit(WebappPack(), manifest={"seed": 5, "scale": SCALE_UP}, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    services = sum(1 for n in snap.graph.nodes.values() if n.kind == "service")
    assert services >= 8


def test_admission_snapshot_id_is_deterministic() -> None:
    """Same seed → same snapshot id (content-addressed identity)."""
    pack = WebappPack()
    snap_a = admit(pack, manifest={"seed": 7})
    snap_b = admit(pack, manifest={"seed": 7})
    assert isinstance(snap_a, Snapshot)
    assert isinstance(snap_b, Snapshot)
    assert snap_a.snapshot_id == snap_b.snapshot_id
