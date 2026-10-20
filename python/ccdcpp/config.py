"""The algorithm's configuration: the shared values, and for each stage the strategy that
decides it and that strategy's parameters.

A ``Config`` is immutable. It holds a dict ``{"shared": {name: value}, "stages": {stage:
{"strategy": name, "params": {name: value}}}}``; a value is a bool, an int, a float, a str, a
list of band roles (strs) or a list of band rows (ints). Pass it as ``config=`` to
``ccdcpp.detect`` and the other entry points: it is checked when it is first run (unknown
stages, strategies, parameters or shared values, values out of range, what a strategy needs from
another) and resolved against each input it runs on (band roles against the input's rows). A
resume record of a configured run continues only under the same configuration.

    import ccdcpp

    cfg = ccdcpp.config("ccdc")                         # the preset: pyccd's algorithm
    cfg = cfg.replace(shared__change_probability=0.95,
                      refit__factor=2.0)                 # shared__value=, stage__parameter=
    print(cfg)                                          # every shared value, stage and parameter
    out = ccdcpp.detect(dates, spectral, qas, input=ccdcpp.presets.landsat_c2(), config=cfg)

A configuration is written in physical terms, so it means the same on every input: the shared
``alpha``, ``coef_floor`` and ``green_filter`` in reflectance (the run multiplies them by the
input's ``reflectance_unit``), the change and outlier thresholds as probabilities (the run takes
their chi-squared quantiles for the number of detection bands), bands by role. The values
several stages read are written once, in ``shared`` (``ccdcpp.shared_values()`` lists them); a
stage cannot set them. Swapping a stage's strategy resets that stage's parameters to the new
strategy's defaults (``cfg.replace(outlier="floored")``). ``ccdcpp.strategies()`` lists every
stage's strategies and their parameters (unit, bounds, source, default, meaning).

Presets are TOML files (``ccdcpp/configs/<name>.toml``, or any path): a ``[shared]`` table and
one table per stage:

    extends = "ccdc"            # optional: start from another preset or file

    [shared]
    change_probability = 0.995

    [refit]
    strategy = "span-factor"
    [refit.params]
    factor = 2.0

A stage a file names with the base's strategy changes only the parameters it gives; with
another strategy, the stage starts from that strategy's defaults. ``stat_ord`` with no
statistics cutoff is ``NO_STAT_CUTOFF``, 9223372036854775807 (``None`` in ``replace()``).
"""

import copy
import os
import tomllib

from . import ccdcpp as _extension

__all__ = ["Config", "config", "shared_values", "strategies"]

_PRESETS = os.path.join(os.path.dirname(__file__), "configs")

# The library registry's schemas, read once.
_SCHEMA = _extension._strategies()
_SHARED = {v["name"]: v for v in _extension._shared()}


def strategies():
    """Every stage's strategies and their parameters, from the library's registry:
    {stage: {strategy: [{"name", "unit", "bounds", "doc", "source", "from", "default"},
    ...]}}. A parameter whose source is "shared" or "chi2" reads the shared value "from" (its
    chi-squared quantile for "chi2") and is not set in its stage."""
    return copy.deepcopy(_SCHEMA)


def shared_values():
    """The shared values of a configuration: [{"name", "unit", "bounds", "doc", "default"},
    ...], the defaults the "ccdc" preset's, in physical terms."""
    return copy.deepcopy(list(_SHARED.values()))


def _parameters(stage, strategy):
    if stage not in _SCHEMA:
        raise ValueError(f"there is no stage '{stage}'; the stages are {', '.join(sorted(_SCHEMA))}")
    if strategy not in _SCHEMA[stage]:
        raise ValueError(f"{stage} has no strategy '{strategy}'; it has "
                         f"{', '.join(sorted(_SCHEMA[stage]))}")
    return _SCHEMA[stage][strategy]


def _defaults(stage, strategy):
    """The parameters a stage of `strategy` sets, at their defaults."""
    return {p["name"]: copy.deepcopy(p["default"]) for p in _parameters(stage, strategy)
            if p["source"] == "own"}


def _value(v, default=None):
    """A value as a configuration holds it: a tuple as a list, numpy scalars as Python's,
    and a whole number given for a real value (its default a float) as a float, so equal
    configurations hash equal."""
    if getattr(v, "ndim", 0) > 0 and hasattr(v, "tolist"):
        v = v.tolist()   # a numpy array: a band list
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        if all(isinstance(b, str) for b in v):
            return [str(b) for b in v]
        return [int(b) for b in v] if all(hasattr(b, "__index__") and not isinstance(b, bool)
                                           for b in v) else v
    if not isinstance(v, (bool, int, float, str)) and hasattr(v, "item"):
        v = v.item()   # a numpy scalar
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def _shared_value(name, v):
    if name == "stat_ord" and v is None:
        return _extension.NO_STAT_CUTOFF
    known = _SHARED.get(name)
    return _value(v, known["default"] if known else None)


def _normalized(stage, strategy, params):
    try:
        defaults = _defaults(stage, strategy)
    except ValueError:
        defaults = {}   # unknown names: the run refuses them, naming them
    return {str(k): _value(v, defaults.get(str(k))) for k, v in params.items()}


class Config:
    """An algorithm configuration (see the module's documentation)."""

    __slots__ = ("_tree", "_compiled")

    def __init__(self, tree):
        if isinstance(tree, Config):
            tree = tree._tree
        if not isinstance(tree, dict) or not set(tree) <= {"shared", "stages"}:
            raise TypeError('a Config is built from a dict {"shared": {...}, "stages": {...}}')
        shared = tree.get("shared", {})
        stages = tree.get("stages", {})
        if not isinstance(shared, dict):
            raise ValueError("shared is a dict of values")
        if not isinstance(stages, dict):
            raise ValueError("stages is a dict of stages")
        clean = {}
        for stage, entry in stages.items():
            if not isinstance(entry, dict) or "strategy" not in entry:
                raise ValueError(f"stage '{stage}' is a dict {{'strategy': ..., 'params': ...}}")
            params = entry.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"stage '{stage}' params is a dict of parameters")
            clean[str(stage)] = {
                "strategy": str(entry["strategy"]),
                "params": _normalized(str(stage), str(entry["strategy"]), params),
            }
        object.__setattr__(self, "_tree", {
            "shared": {str(k): _shared_value(str(k), v) for k, v in shared.items()},
            "stages": clean,
        })
        object.__setattr__(self, "_compiled", None)

    def __setattr__(self, name, value):
        raise AttributeError("a Config is immutable; replace() returns a changed copy")

    @property
    def _algorithm(self):
        """The compiled configuration (an _extension.Algorithm), built on first use: what the
        entry points run. ValueError for what it refuses."""
        if self._compiled is None:
            object.__setattr__(self, "_compiled", _extension.Algorithm(self._tree))
        return self._compiled

    def stages(self):
        """The stage names, sorted."""
        return sorted(self._tree["stages"])

    def strategy(self, stage):
        return self._tree["stages"][stage]["strategy"]

    def params(self, stage):
        """A copy of the stage's parameters."""
        return copy.deepcopy(self._tree["stages"][stage]["params"])

    def shared(self):
        """A copy of the shared values."""
        return copy.deepcopy(self._tree["shared"])

    def replace(self, **changes):
        """A copy with changes: ``shared__name=value`` sets a shared value (``stat_ord=None``
        for no statistics cutoff), ``stage="strategy"`` swaps a stage's strategy (its
        parameters become the new strategy's defaults), ``stage__param=value`` sets one
        parameter. Strategy swaps apply before parameter changes."""
        tree = copy.deepcopy(self._tree)
        swaps = {k: v for k, v in changes.items() if "__" not in k}
        sets = {k: v for k, v in changes.items() if "__" in k}
        for stage, strategy in swaps.items():
            tree["stages"][stage] = {"strategy": strategy, "params": _defaults(stage, strategy)}
        for key, value in sets.items():
            stage, param = key.split("__", 1)
            if stage == "shared":
                if param not in _SHARED:
                    raise ValueError(f"there is no shared value '{param}'; the shared values "
                                     f"are {', '.join(_SHARED)}")
                tree["shared"][param] = _shared_value(param, value)
                continue
            if stage not in tree["stages"]:
                raise ValueError(f"the configuration has no stage '{stage}'")
            entry = tree["stages"][stage]
            info = {p["name"]: p for p in _parameters(stage, entry["strategy"])}
            known = _defaults(stage, entry["strategy"])
            if param in info and info[param]["source"] != "own":
                source = info[param]["from"]
                raise ValueError(f"{stage} '{entry['strategy']}' parameter '{param}' is the "
                                 f"shared value '{source}'; set shared__{source} instead")
            if param not in known:
                raise ValueError(f"{stage} '{entry['strategy']}' has no parameter '{param}'; "
                                 f"it has {', '.join(known)}")
            entry["params"][param] = _value(value, known[param])
        return Config(tree)

    def check(self, input=None):
        """Raises ValueError for what a run would refuse; with ``input`` (a ccdcpp.Input),
        what it refuses against that input too (a role it lacks, a row off its bands)."""
        _extension._config_check(self._tree, input)

    def resolve(self, input):
        """The configuration resolved against ``input`` (a ccdcpp.Input), as the run runs it:
        {"stages", "shared", "digest"}, every value in the input's units and rows."""
        return self._algorithm.resolve(input)

    def hash(self):
        """The configuration's digest: equal for equal configurations, on any platform."""
        return _extension._config_digest(self._tree)

    def to_dict(self):
        return copy.deepcopy(self._tree)

    def to_toml(self):
        """The configuration as a preset file (a [shared] table, then every stage)."""
        def text(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, str):
                return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
            if isinstance(v, float):
                return repr(v)
            if isinstance(v, list):
                return "[" + ", ".join(text(b) if isinstance(b, str) else str(int(b))
                                       for b in v) + "]"
            return str(v)

        lines = ["[shared]"]
        for name in sorted(self._tree["shared"]):
            lines.append(f"{name} = {text(self._tree['shared'][name])}")
        lines.append("")
        for stage in self.stages():
            entry = self._tree["stages"][stage]
            lines.append(f"[{stage}]")
            lines.append(f"strategy = {text(entry['strategy'])}")
            if entry["params"]:
                lines.append(f"[{stage}.params]")
                for name in sorted(entry["params"]):
                    lines.append(f"{name} = {text(entry['params'][name])}")
            lines.append("")
        return "\n".join(lines)

    def __eq__(self, other):
        return isinstance(other, Config) and self._tree == other._tree

    def __hash__(self):
        return self.hash()

    def __repr__(self):
        rows = ["  shared: " + ", ".join(f"{k}={self._tree['shared'][k]!r}"
                                         for k in sorted(self._tree["shared"]))]
        for stage in self.stages():
            entry = self._tree["stages"][stage]
            params = ", ".join(f"{k}={entry['params'][k]!r}" for k in sorted(entry["params"]))
            rows.append(f"  {stage}: {entry['strategy']}({params})")
        return "Config(\n" + "\n".join(rows) + "\n)"

    def __reduce__(self):
        return (Config, (self._tree,))


def _load(path, seen):
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
        else {"shared": {}, "stages": {}}
    shared = data.pop("shared", {})
    if not isinstance(shared, dict):
        raise ValueError(f"{path}: [shared] is a table of values")
    tree["shared"].update(shared)
    for stage, entry in data.items():
        if not isinstance(entry, dict) or "strategy" not in entry:
            raise ValueError(f"{path}: stage '{stage}' needs strategy = \"...\"")
        strategy = entry["strategy"]
        params = dict(entry.get("params", {}))
        stages = tree["stages"]
        if stage in stages and stages[stage]["strategy"] == strategy:
            stages[stage]["params"].update(params)
        else:
            merged = _defaults(stage, strategy)
            merged.update(params)
            stages[stage] = {"strategy": strategy, "params": merged}
    return Config(tree)


# The presets by name, built once: they are fixed.
_NAMED = {}


def _resolve(source, seen=frozenset()):
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


def config(source="ccdc"):
    """A configuration: a preset's name (``ccdcpp/configs/<name>.toml``, or one of the
    library's registry, an extension's; built once and shared), a TOML file's path, a dict
    ``{"shared": ..., "stages": ...}`` or a Config."""
    return _resolve(source)
