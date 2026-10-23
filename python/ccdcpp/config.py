"""The algorithm's configuration: the shared values, for each stage the strategy that
decides it and that strategy's parameters, and each procedure's preprocessing.

A ``Config`` is immutable, and checked and compiled when it is built. It holds a dict laid out
as its TOML file::

    {"shared": {name: value, table: {name: value}},
     stage: {"strategy": name, parameter: value, ...},          # one per stage
     "preprocess": {procedure: {"thermal": "celsius" or "stored",
                                "steps": [{"step": name, parameter: value}, ...]}}}

A value is a bool, an int, a float, a str, or a list of names (band roles, QA code names), of
ints (band rows, QA codes), of numbers (a range's [min, max]) or of bools (its ends'). Pass it
as ``config=`` to ``ccdcpp.detect`` and ``ccdcpp.detect_cube``: it is checked when it is built
(unknown stages, strategies, steps, parameters or shared values, values out of range, what a
strategy needs from another) and resolved against each data spec it runs on once (band roles
against the spec's rows). A resume record of a configured run continues only under the same
configuration.

    import ccdcpp

    cfg = ccdcpp.config("ccdc")                         # the preset: pyccd's algorithm
    cfg = cfg.replace(shared__change_probability=0.95,
                      refit__factor=2.0)                 # shared__value=, stage__parameter=
    print(cfg)                                          # every shared value, stage and parameter
    out = ccdcpp.detect(dates, spectral, qas, spec=ccdcpp.presets.landsat_c2(), config=cfg)

A configuration is written in physical terms, so it means the same on every data spec: the shared
``lasso.alpha`` and ``lasso.coef_floor`` and the preprocessing's ranges and offsets in
reflectance (the run multiplies them by the spec's ``reflectance_scale``), the thermal range
in degrees Celsius, the change and outlier thresholds as probabilities (the run takes their
chi-squared quantiles for the number of detection bands), bands by role, QA codes by name. The
values several stages read are written once, in ``shared`` (``ccdcpp.shared_values()`` lists
them); a stage cannot set them. Swapping a stage's strategy resets that stage's parameters to
the new strategy's defaults (``cfg.replace(outlier="floored")``). ``ccdcpp.strategies()``
lists every stage's strategies and their parameters (unit, bounds, source, default, meaning).

The preprocessing (``ccdcpp.Preprocess``, ``ProcedureChain`` and the step classes;
``ccdcpp.steps()`` lists the step types) is edited with ``replace()`` too::

    cfg.replace(preprocess__insufficient_clear__green_median__offset=0.05)  # one step
    cfg.replace(preprocess__thermal_range__celsius=[-80.0, 70.7])   # every thermal_range step
    cfg.replace(preprocess__standard=ccdcpp.ProcedureChain(steps=(...), thermal="celsius"))

Presets are TOML files (``ccdcpp/configs/<name>.toml``, or any path), laid out as the dict:
``Config.to_toml()`` writes one. A file of your own lists only what it changes::

    extends = "ccdc"            # optional: start from another preset or file

    [shared]
    change_probability = 0.995

    [shared.lasso]
    alpha = 0.0002

    [refit]
    factor = 2.0

    [preprocess.snow]
    steps = [{ step = "keep_qa", codes = ["clear", "water"] }, { step = "one_per_date" }]

A stage a file names keeps its base's strategy and parameters but for the ones it gives; with
another ``strategy``, the stage starts from that strategy's defaults. A shared table changes
the values it gives. A procedure a file names takes the file's ``thermal`` and ``steps`` where
it gives them, its base's otherwise. ``statistics_end`` is ``"none"`` (no statistics cutoff), a
date ``"YYYY-MM-DD"`` or an ordinal day; a lookback's and a stability rule's ``peek`` is
``"change_test"`` (the change test's) or a number.
"""
from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Sequence
from typing import Any

from . import ccdcpp as _extension
from .preprocess import PROCEDURES, Preprocess, ProcedureChain, step_dict

__all__ = ["Config", "config", "shared_values", "steps", "strategies"]

_PRESETS = os.path.join(os.path.dirname(__file__), "configs")

# The library registry's schemas, read once.
_SCHEMA = _extension._strategies()
_STEPS = _extension._steps()
_SHARED = {v["name"]: v for v in _extension._shared()}
_TABLES = _extension._shared_tables()
# The stages, in the order a pixel meets them, with what each decides.
_STAGES = _extension._stages()
# The sections of a configuration's dict that are not stages.
_SECTIONS = ("shared", "preprocess")


def strategies() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every stage's strategies and their parameters, from the library's registry:
    {stage: {strategy: [{"name", "unit", "bounds", "doc", "source", "from", "default"},
    ...]}}. A parameter whose source is "shared" or "chi2" reads the shared value "from" (its
    chi-squared quantile for "chi2") and is not set in its stage."""
    return copy.deepcopy(_SCHEMA)


def steps() -> dict[str, list[dict[str, Any]]]:
    """Every preprocessing step type and its parameters, from the library's registry:
    {step: [{"name", "unit", "bounds", "doc", "source", "from", "default"}, ...]}. A parameter
    whose source is "shared" reads that shared value and is not set in the step."""
    return copy.deepcopy(_STEPS)


def shared_values() -> list[dict[str, Any]]:
    """The shared values of a configuration: [{"name", "unit", "bounds", "doc", "default"},
    ...], the defaults the "ccdc" preset's, in physical terms; a table's value by its path,
    "lasso.alpha" (``Config.replace(shared__lasso__alpha=...)``)."""
    return copy.deepcopy(list(_SHARED.values()))


def _chain(value: object, base: dict[str, Any] | None = None) -> object:
    """A procedure as a configuration's dict holds it, from a ProcedureChain, a list of steps
    (``base``'s thermal) or a dict (missing keys from ``base``); the steps as dicts.
    Anything else goes to the extension as it is, which refuses it."""
    if isinstance(value, ProcedureChain):
        value = value.to_dict()
    elif isinstance(value, (list, tuple)):
        value = {"steps": list(value)}
    if not isinstance(value, dict):
        return value
    chain: dict[str, Any] = dict(base or {})
    chain.update(value)
    if isinstance(chain.get("steps"), (list, tuple)):
        chain["steps"] = [step_dict(s) for s in chain["steps"]]
    return chain


def _set_step(tree: dict[str, Any], procedures: Sequence[str], step: str, param: str,
              value: Any) -> None:
    """Sets `param` of every step of type `step` in `procedures` (one step per procedure when
    a procedure is named, ValueError if it has none or several)."""
    found = 0
    for procedure in procedures:
        matches = [s for s in tree["preprocess"][procedure]["steps"] if s["step"] == step]
        if len(procedures) == 1 and len(matches) != 1:
            raise ValueError(f"preprocess {procedure} has {len(matches)} '{step}' steps; "
                             f"replace its chain (preprocess__{procedure}=...) instead")
        for s in matches:
            s[param] = value
            found += 1
    if not found:
        raise ValueError(f"the configuration's preprocess has no '{step}' step")


# What the comments of a preset file say of a value's unit (to_toml()).
_UNIT_NOTES = {"observations": "observations", "days": "days", "reflectance": "reflectance",
               "reflectance_delta": "reflectance", "celsius": "degrees Celsius",
               "date": 'a date "YYYY-MM-DD", an ordinal day, or "none"'}
_THERMAL_NOTES = {"celsius": "through the spec's thermal transform (degrees Celsius)",
                  "stored": "as stored; the range still compares in degrees Celsius"}
# The column a preset file's comments start at, and its section banners' width.
_NOTE_COLUMN = 33
_BANNER_WIDTH = 92


def _text(v: Any) -> str:
    """A value as TOML writes it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_text(b) for b in v) + "]"
    return str(v)


def _unit_note(info: dict[str, Any]) -> str:
    """A parameter's unit, as a comment says it ("" for none), with the name it takes for a
    value (its keyword)."""
    note = _UNIT_NOTES.get(info.get("unit", ""), "")
    keyword = info.get("keyword")
    if keyword and keyword not in note:
        note = f'{note}, or "{keyword}"' if note else f'or "{keyword}"'
    return note


def _noted(line: str, note: str) -> str:
    """`line` with `note` as its comment, at the comment column."""
    if not note:
        return line
    return f"{line.ljust(_NOTE_COLUMN - 1)} # {note}" if len(line) < _NOTE_COLUMN - 1 \
        else f"{line}  # {note}"


def _banner(title: str) -> str:
    head = f"# --- {title} "
    return head + "-" * max(3, _BANNER_WIDTH - len(head))


def _left_out(value: Any, info: dict[str, Any], full: bool) -> bool:
    """Whether to_toml() leaves the value out: a flag left off at its default."""
    default = info.get("default")
    off = default is False or (isinstance(default, list) and bool(default)
                               and all(v is False for v in default))
    return not full and off and value == default


def _step_row(step: dict[str, Any], full: bool) -> tuple[str, str]:
    """A step as a chain's line, in its schema's order, and its units' comment."""
    info = {p["name"]: p for p in _STEPS.get(step["step"], ())}
    fields = [f"step = {_text(step['step'])}"]
    notes: list[str] = []
    for name in list(info) + sorted(set(step) - set(info) - {"step"}):
        if name not in step or _left_out(step[name], info.get(name, {}), full):
            continue
        fields.append(f"{name} = {_text(step[name])}")
        note = _unit_note(info.get(name, {}))
        if note and note not in notes:
            notes.append(note)
    return "    { " + ", ".join(fields) + " },", "; ".join(notes)


def _paths(shared: dict[str, Any]) -> list[tuple[str, Any]]:
    """The shared values by their paths: a table's as "lasso.alpha"."""
    return [(f"{k}.{n}", v) for k, t in shared.items() if isinstance(t, dict)
            for n, v in t.items()] + [(k, v) for k, v in shared.items()
                                      if not isinstance(v, dict)]


class Config:
    """An algorithm configuration (see the module's documentation). Building one checks it
    (ValueError for what a run would refuse on any data spec) and compiles it once: every call
    that runs it reuses the compilation, and its resolution against each data spec it meets. It
    holds the configuration in full, as the extension writes it out: every shared value and
    every stage's and step's parameters spelled out, a whole number given for a real as the
    real, so equal configurations compare and hash equal.

    What the data spec says and what the configuration says: ``ccdcpp.DataSpec``'s documentation.
    docs/units.md states the boundary: a configuration is in the physical domain
    (reflectance levels and differences, degrees Celsius), and a run converts it once into
    the data's storage encoding (on Collection-1 DN, band_range 0–1 is 0–10000, -93.2–70.7
    degrees Celsius is -9320–7070, alpha 1e-4 is 1.0, the green offset 0.04 is 400).

    Example: ``cfg = ccdcpp.config("ccdc").replace(shared__change_probability=0.995)``
    """

    __slots__ = ("_tree", "_algorithm")
    _tree: dict[str, Any]
    _algorithm: _extension.Algorithm

    def __init__(self, tree: Config | dict[str, Any]) -> None:
        if isinstance(tree, Config):
            tree = tree._tree
        # The compiled configuration (an _extension.Algorithm): what the entry points run.
        # It checks the tree, and holds it in full: every value spelled out (config()).
        algorithm = _extension.Algorithm(tree)
        object.__setattr__(self, "_algorithm", algorithm)
        object.__setattr__(self, "_tree", algorithm.config())

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a Config is immutable; replace() returns a changed copy")

    def stages(self) -> list[str]:
        """The stage names, in the order a pixel meets them."""
        return [s["name"] for s in _STAGES if s["name"] in self._tree]

    def strategy(self, stage: str) -> str:
        return self._tree[stage]["strategy"]

    def params(self, stage: str) -> dict[str, Any]:
        """A copy of the stage's parameters."""
        return {k: copy.deepcopy(v) for k, v in self._tree[stage].items() if k != "strategy"}

    def shared(self) -> dict[str, Any]:
        """A copy of the shared values: {name: value, table: {name: value}}."""
        return copy.deepcopy(self._tree["shared"])

    def preprocess(self, procedure: str) -> dict[str, Any]:
        """A copy of a procedure's preprocessing: {"thermal", "steps"}."""
        return copy.deepcopy(self._tree["preprocess"][procedure])

    def replace(self, **changes: Any) -> Config:
        """A copy with changes:

        - ``shared__name=value`` sets a shared value, ``shared__table__name=value`` one of a
          table's (``shared__lasso__alpha=1e-5``);
        - ``stage="strategy"`` swaps a stage's strategy (its parameters become the new
          strategy's defaults), ``stage__param=value`` sets one parameter
          (``stage__table__name=value`` the one named for a shared table's value,
          ``table.name``);
        - ``preprocess=ccdcpp.Preprocess(...)`` replaces the procedures it gives,
          ``preprocess__procedure=chain`` one procedure (a ``ccdcpp.ProcedureChain``, a list of
          steps, or a dict), ``preprocess__procedure__thermal="celsius"`` (or ``"stored"``) how it
          reads the thermal row,
          ``preprocess__procedure__step__param=value`` a parameter of its one step of that
          type, and ``preprocess__step__param=value`` that parameter of every step of that
          type.

        Strategy swaps and whole procedures apply before parameter changes."""
        tree = copy.deepcopy(self._tree)
        if "preprocess" in changes:
            given = changes.pop("preprocess")
            given = given.to_dict() if isinstance(given, Preprocess) else given
            if not isinstance(given, dict):
                raise TypeError("preprocess= is a ccdcpp.Preprocess or a dict of procedures")
            for procedure, chain in given.items():
                tree["preprocess"][procedure] = _chain(chain, tree["preprocess"].get(procedure))
        swaps = {k: v for k, v in changes.items() if "__" not in k}
        sets = {k: v for k, v in changes.items() if "__" in k}
        for stage, strategy in swaps.items():
            if stage in _SECTIONS:
                raise ValueError(f"{stage}= is not a stage; write {stage}__name=value")
            tree[stage] = {"strategy": strategy}
        chains = {k: v for k, v in sets.items()
                  if k.startswith("preprocess__") and k.count("__") == 1}
        for key, value in chains.items():
            procedure = key.split("__", 1)[1]
            if procedure not in PROCEDURES:
                raise ValueError(f"preprocess has no procedure '{procedure}'; the procedures "
                                 f"are {', '.join(PROCEDURES)}")
            tree["preprocess"][procedure] = _chain(value, tree["preprocess"].get(procedure))
        for key, value in sets.items():
            if key in chains:
                continue
            stage, param = key.split("__", 1)
            if stage == "preprocess":
                parts = param.split("__")
                if parts[0] in PROCEDURES:
                    if parts[0] not in tree["preprocess"]:
                        raise ValueError(f"the configuration has no preprocess '{parts[0]}'")
                    if parts[1:] == ["thermal"]:
                        tree["preprocess"][parts[0]]["thermal"] = value
                    elif len(parts) == 3:
                        _set_step(tree, [parts[0]], parts[1], parts[2], value)
                    else:
                        raise ValueError(f"{key}: write preprocess__procedure__step__param")
                elif len(parts) == 2:
                    _set_step(tree, list(tree["preprocess"]), parts[0], parts[1], value)
                else:
                    raise ValueError(f"{key}: write preprocess__step__param or "
                                     f"preprocess__procedure__step__param")
                continue
            if stage == "shared":
                if "__" in param:   # a table's value: shared__lasso__alpha
                    table, param = param.split("__", 1)
                    if not isinstance(tree["shared"].get(table), dict):
                        raise ValueError(f"shared has no table '{table}'; its tables are "
                                         f"{', '.join(t['name'] for t in _TABLES)}")
                    tree["shared"][table][param] = value
                else:
                    tree["shared"][param] = value
                continue
            if stage not in tree:
                raise ValueError(f"the configuration has no stage '{stage}'")
            if param == "strategy":
                raise ValueError(f"{key}: swap a strategy with {stage}=\"...\"")
            tree[stage][param.replace("__", ".")] = value
        return Config(tree)

    def check(self, spec: _extension.DataSpec | None = None) -> None:
        """Raises ValueError for what a run would refuse; with ``spec`` (a ccdcpp.DataSpec),
        what it refuses against that spec too (a role it lacks, a row off its bands)."""
        _extension._config_check(self._tree, spec)

    def resolve(self, spec: _extension.DataSpec) -> dict[str, Any]:
        """The configuration resolved against ``spec`` (a ccdcpp.DataSpec), as the run runs
        it, laid out as the configuration (shared, each stage, preprocess) with its "digest",
        every value in the spec's units, rows and codes."""
        return self._algorithm.resolve(spec)

    def hash(self) -> int:
        """The configuration's digest: equal for equal configurations, on any platform."""
        return _extension._config_digest(self._tree)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._tree)

    def to_toml(self, full: bool = False) -> str:
        """The configuration as a preset file: the shared values, each procedure's
        preprocessing, then the stages in the order a pixel meets them; as comments from the
        registry's schemas, each table with what it decides, each shared value with what it
        is, and every other value with its unit. A flag
        left off at its default (a range's ``inclusive = [false, false]``) is left out, unless
        ``full``."""
        tree = self._tree
        lines = [_banner("Values several stages share"), "", "[shared]"]
        shared = tree["shared"]
        tables = {t["name"]: t["doc"] for t in _TABLES}
        for name, info in _SHARED.items():
            if "." not in name and name in shared:
                lines.append(_noted(f"{name} = {_text(shared[name])}", info["doc"]))
        for table, doc in tables.items():
            if not isinstance(shared.get(table), dict):
                continue
            lines += ["", _noted(f"[shared.{table}]", doc)]
            for path, info in _SHARED.items():
                name = path.removeprefix(f"{table}.")
                if name != path and name in shared[table]:
                    lines.append(_noted(f"{name} = {_text(shared[table][name])}", info["doc"]))
        lines += ["", _banner("Preprocessing: which observations each procedure keeps")]
        for procedure in PROCEDURES:
            if procedure not in tree["preprocess"]:
                continue
            chain = tree["preprocess"][procedure]
            lines += ["", f"[preprocess.{procedure}]",
                      _noted(f"thermal = {_text(chain['thermal'])}", _THERMAL_NOTES.get(
                          chain["thermal"], "")), "steps = ["]
            rows = [_step_row(step, full) for step in chain["steps"]]
            width = max((len(row) for row, _ in rows), default=0) + 2
            lines += [row.ljust(width) + f"# {note}" if note else row for row, note in rows]
            lines.append("]")
        lines += ["", _banner("The stages, in the order a pixel meets them")]
        for stage in _STAGES:
            name = stage["name"]
            if name not in tree:
                continue
            entry = tree[name]
            info = {p["name"]: p for p in _SCHEMA.get(name, {}).get(entry["strategy"], [])}
            lines += ["", _noted(f"[{name}]", stage["purpose"]),
                      f"strategy = {_text(entry['strategy'])}"]
            for param, p in info.items():
                if param in entry and not _left_out(entry[param], p, full):
                    lines.append(_noted(f"{param} = {_text(entry[param])}", _unit_note(p)))
        return "\n".join(lines) + "\n"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Config) and self._tree == other._tree

    def __hash__(self) -> int:
        return self.hash()

    def __repr__(self) -> str:
        rows = ["  shared: " + ", ".join(f"{k}={v!r}"
                                         for k, v in sorted(_paths(self._tree["shared"])))]
        for stage in self.stages():
            params = self.params(stage)
            rows.append(f"  {stage}: {self.strategy(stage)}("
                        + ", ".join(f"{k}={params[k]!r}" for k in sorted(params)) + ")")
        for procedure in PROCEDURES:
            if procedure not in self._tree["preprocess"]:
                continue
            chain = self._tree["preprocess"][procedure]
            steps = "; ".join(
                s["step"] + "(" + ", ".join(f"{k}={s[k]!r}" for k in sorted(s) if k != "step")
                + ")" for s in chain["steps"])
            rows.append(f"  preprocess {procedure}: thermal {chain['thermal']}, {steps}")
        return "Config(\n" + "\n".join(rows) + "\n)"

    def __reduce__(self) -> tuple[type[Config], tuple[dict[str, Any]]]:
        return (Config, (self._tree,))


def _load(path: str, seen: frozenset[str]) -> Config:
    real = os.path.realpath(path)
    if real in seen:
        raise ValueError(f"{path}: its extends chain comes back to it")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    base = data.pop("extends", None)
    if isinstance(base, str) and (base.endswith(".toml") or os.sep in base) \
            and not os.path.isabs(base):
        base = os.path.join(os.path.dirname(os.path.abspath(path)), base)   # the file's own dir
    tree = copy.deepcopy(_resolve(base, seen | {real})._tree) if base is not None \
        else {"shared": {}, "preprocess": {}}
    shared = data.pop("shared", {})
    if not isinstance(shared, dict):
        raise ValueError(f"{path}: [shared] is a table of values")
    for name, value in shared.items():
        if isinstance(value, dict) and isinstance(tree["shared"].get(name), dict):
            tree["shared"][name].update(value)   # a table: the file's values over the base's
        else:
            tree["shared"][name] = value
    preprocess = data.pop("preprocess", {})
    if not isinstance(preprocess, dict):
        raise ValueError(f"{path}: [preprocess] is a table of procedures")
    for procedure, chain in preprocess.items():
        tree["preprocess"][procedure] = _chain(chain, tree["preprocess"].get(procedure))
    for stage, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: stage '{stage}' is a table, [{stage}]")
        base_entry = tree.get(stage)
        if isinstance(base_entry, dict) and entry.get("strategy", base_entry["strategy"]) \
                == base_entry["strategy"]:
            base_entry.update(entry)    # the base's strategy: the file's parameters over its
        elif "strategy" in entry:
            tree[stage] = dict(entry)   # another strategy: its defaults, and the file's
        else:
            raise ValueError(f"{path}: stage '{stage}' needs strategy = \"...\"")
    try:
        return Config(tree)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


# The presets by name, built once: they are fixed.
_NAMED: dict[str, Config] = {}


def _resolve(source: Config | dict[str, Any] | str | os.PathLike[str],
             seen: frozenset[str] = frozenset()) -> Config:
    if isinstance(source, Config):
        return source
    if isinstance(source, dict):
        return Config(source)
    name = str(source)
    if name.endswith(".toml") or os.sep in name:
        if not os.path.exists(name):
            raise ValueError(f"no configuration file '{name}'")
        return _load(name, seen)
    if name in _NAMED:
        return _NAMED[name]
    # A preset: the package's file, else the library's registry's (an extension's).
    files = {f[:-5] for f in os.listdir(_PRESETS) if f.endswith(".toml")}
    if name in files:
        cfg = _load(os.path.join(_PRESETS, name + ".toml"), seen)
    elif name in set(_extension._presets()):
        cfg = Config(_extension._config_preset(name))
    else:
        raise ValueError(f"no configuration '{name}'; the presets are "
                         f"{', '.join(sorted(files | set(_extension._presets())))}")
    _NAMED[name] = cfg
    return cfg


def config(source: Config | dict[str, Any] | str | os.PathLike[str] = "ccdc") -> Config:
    """A configuration: a preset's name (``ccdcpp/configs/<name>.toml``, or one of the
    library's registry, an extension's; built once and shared), a TOML file's path, a
    configuration's dict (``{"shared": ..., <stage>: {"strategy": ..., ...}, "preprocess":
    ...}``) or a Config."""
    return _resolve(source)
