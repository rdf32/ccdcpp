"""The algorithm's configuration: for each stage, the strategy that decides it and that
strategy's parameters.

A ``Config`` is immutable and holds a dict of stages, each ``{"strategy": name, "params":
{name: value}}``; a value is a bool, an int, a float, a str or a list of band indices. Pass it
as ``config=`` to ``ccdcpp.detect`` and the other entry points: the run uses its strategies and
checks it before any pixel runs (unknown stages, strategies or parameters, values out of range,
band lists against the input's bands, what a strategy needs from another). A resume record of a
configured run continues only under the same configuration.

    import ccdcpp

    cfg = ccdcpp.config("ccdc")                         # the preset: pyccd's algorithm
    cfg = cfg.replace(change_test__change_probability=0.95,
                      refit__factor=2.0)                 # stage__parameter=value
    print(cfg)                                          # every stage and parameter
    out = ccdcpp.detect(dates, spectral, qas, profile=ccdcpp.presets.landsat_c2(), config=cfg)

Swapping a stage's strategy resets that stage's parameters to the new strategy's defaults
(``cfg.replace(outlier="floored")``). ``ccdcpp.strategies()`` lists every stage's strategies
and their parameters (unit, bounds, default, meaning).

A configuration's parameters are in the input's units: the Lasso's ``alpha`` and
``coef_floor`` are 1.0 for Collection-1 DN and 1e-4 for reflectance. A sensor profile writes
a preset for its input (``profile.config("ccdc")``: its units, its Tmask bands, its cadence,
its thresholds for its detection-band count), which is what a run with ``profile=`` and no
``config=`` uses.

Presets are TOML files (``ccdcpp/configs/<name>.toml``, or any path), one table per stage:

    extends = "ccdc"            # optional: start from another preset or file

    [refit]
    strategy = "span-factor"
    [refit.params]
    factor = 2.0

A stage a file names with the base's strategy changes only the parameters it gives; with
another strategy, the stage starts from that strategy's defaults.
"""

import copy
import os
import tomllib

from . import ccdcpp as _extension

__all__ = ["Config", "config", "strategies"]

_PRESETS = os.path.join(os.path.dirname(__file__), "configs")


def strategies():
    """Every stage's strategies and their parameters, from the library's registry:
    {stage: {strategy: [{"name", "unit", "bounds", "doc", "default"}, ...]}}."""
    return _extension._strategies()


def _defaults(stage, strategy):
    table = strategies()
    if stage not in table:
        raise ValueError(f"there is no stage '{stage}'; the stages are {', '.join(sorted(table))}")
    if strategy not in table[stage]:
        raise ValueError(f"{stage} has no strategy '{strategy}'; it has "
                         f"{', '.join(sorted(table[stage]))}")
    return {p["name"]: p["default"] for p in table[stage][strategy]}


def _value(v, default=None):
    """A parameter's value as a configuration holds it: a tuple as a list, numpy
    scalars as Python's, and a whole number given for a real parameter (its
    default a float) as a float, so equal configurations hash equal."""
    if getattr(v, "ndim", 0) > 0 and hasattr(v, "tolist"):
        v = v.tolist()   # a numpy array: a band list
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [int(b) for b in v] if all(hasattr(b, "__index__") and not isinstance(b, bool)
                                           for b in v) else v
    if not isinstance(v, (bool, int, float, str)) and hasattr(v, "item"):
        v = v.item()   # a numpy scalar
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def _normalized(stage, strategy, params):
    try:
        defaults = _defaults(stage, strategy)
    except ValueError:
        defaults = {}   # unknown names: the run refuses them, naming them
    return {str(k): _value(v, defaults.get(str(k))) for k, v in params.items()}


class Config:
    """An algorithm configuration (see the module's documentation)."""

    __slots__ = ("_tree",)

    def __init__(self, tree):
        if isinstance(tree, Config):
            tree = tree._tree
        if not isinstance(tree, dict):
            raise TypeError("a Config is built from a dict of stages")
        clean = {}
        for stage, entry in tree.items():
            if not isinstance(entry, dict) or "strategy" not in entry:
                raise ValueError(f"stage '{stage}' is a dict {{'strategy': ..., 'params': ...}}")
            params = entry.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"stage '{stage}' params is a dict of parameters")
            clean[str(stage)] = {
                "strategy": str(entry["strategy"]),
                "params": _normalized(str(stage), str(entry["strategy"]), params),
            }
        object.__setattr__(self, "_tree", clean)

    def __setattr__(self, name, value):
        raise AttributeError("a Config is immutable; replace() returns a changed copy")

    def stages(self):
        """The stage names, sorted."""
        return sorted(self._tree)

    def strategy(self, stage):
        return self._tree[stage]["strategy"]

    def params(self, stage):
        """A copy of the stage's parameters."""
        return copy.deepcopy(self._tree[stage]["params"])

    def replace(self, **changes):
        """A copy with changes: ``stage="strategy"`` swaps a stage's strategy (its
        parameters become the new strategy's defaults), ``stage__param=value`` sets one
        parameter. Strategy swaps apply before parameter changes."""
        tree = copy.deepcopy(self._tree)
        swaps = {k: v for k, v in changes.items() if "__" not in k}
        sets = {k: v for k, v in changes.items() if "__" in k}
        for stage, strategy in swaps.items():
            tree[stage] = {"strategy": strategy, "params": _defaults(stage, strategy)}
        for key, value in sets.items():
            stage, param = key.split("__", 1)
            if stage not in tree:
                raise ValueError(f"the configuration has no stage '{stage}'")
            entry = tree[stage]
            known = _defaults(stage, entry["strategy"])
            if param not in known:
                raise ValueError(f"{stage} '{entry['strategy']}' has no parameter '{param}'; "
                                 f"it has {', '.join(known)}")
            entry["params"][param] = _value(value, known[param])
        return Config(tree)

    def check(self, bands=None):
        """Raises ValueError for what a run would refuse; with ``bands``, band lists are
        checked against an input of that many bands too."""
        _extension._config_check(self._tree, -1 if bands is None else int(bands))

    def hash(self):
        """The configuration's digest: equal for equal configurations, on any platform."""
        return _extension._config_digest(self._tree)

    def to_dict(self):
        return copy.deepcopy(self._tree)

    def to_toml(self):
        """The configuration as a preset file (every stage and parameter)."""
        def text(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, str):
                return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
            if isinstance(v, float):
                return repr(v)
            if isinstance(v, list):
                return "[" + ", ".join(str(int(b)) for b in v) + "]"
            return str(v)

        lines = []
        for stage in self.stages():
            entry = self._tree[stage]
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
        rows = []
        for stage in self.stages():
            entry = self._tree[stage]
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
    tree = _resolve(base, seen | {real})._tree if base is not None else {}
    tree = copy.deepcopy(tree)
    for stage, entry in data.items():
        if not isinstance(entry, dict) or "strategy" not in entry:
            raise ValueError(f"{path}: stage '{stage}' needs strategy = \"...\"")
        strategy = entry["strategy"]
        params = dict(entry.get("params", {}))
        if stage in tree and tree[stage]["strategy"] == strategy:
            tree[stage]["params"].update(params)
        else:
            merged = _defaults(stage, strategy)
            merged.update(params)
            tree[stage] = {"strategy": strategy, "params": merged}
    return Config(tree)


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
    # A preset: the package's file, else the library's registry's (an extension's).
    files = {f[:-5] for f in os.listdir(_PRESETS) if f.endswith(".toml")}
    if name in files:
        return _load(os.path.join(_PRESETS, name + ".toml"), seen)
    built_in = set(_extension._presets())
    if name in built_in:
        return Config(_extension._config_preset(name))
    raise ValueError(f"no configuration '{name}'; the presets are "
                     f"{', '.join(sorted(files | built_in))}")


def config(source="ccdc"):
    """A configuration: a preset's name (``ccdcpp/configs/<name>.toml``, or one of the
    library's registry, an extension's), a TOML file's path, a dict of stages or a Config."""
    return _resolve(source)
