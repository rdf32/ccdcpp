"""The algorithm's configuration: the shared values, for each stage the strategy that
decides it and that strategy's parameters, and each procedure's preprocessing.

A ``Config`` is immutable. It holds a dict ``{"shared": {name: value}, "stages": {stage:
{"strategy": name, "params": {name: value}}}, "preprocess": {procedure: {"thermal_transform":
bool, "steps": [{"step": name, name: value, ...}, ...]}}}``; a value is a bool, an int, a
float, a str, a list of names (band roles, QA code names) or a list of ints (band rows, QA
codes). Pass it as ``config=`` to
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
``alpha`` and ``coef_floor`` and the preprocessing's bounds and offsets in reflectance (the run
multiplies them by the input's ``reflectance_unit``), the thermal bounds in degrees Celsius, the
change and outlier thresholds as probabilities (the run takes their chi-squared quantiles for
the number of detection bands), bands by role, QA codes by name. The values
several stages read are written once, in ``shared`` (``ccdcpp.shared_values()`` lists them); a
stage cannot set them. Swapping a stage's strategy resets that stage's parameters to the new
strategy's defaults (``cfg.replace(outlier="floored")``). ``ccdcpp.strategies()`` lists every
stage's strategies and their parameters (unit, bounds, source, default, meaning).

The preprocessing (``ccdcpp.Preprocess``, ``ProcedureChain`` and the step classes;
``ccdcpp.steps()`` lists the step types) is edited with ``replace()`` too::

    cfg.replace(preprocess__insufficient_clear__green_median__offset=0.05)  # one step
    cfg.replace(preprocess__thermal_range__min=-80.0)      # every thermal_range step
    cfg.replace(preprocess__standard=ccdcpp.ProcedureChain(steps=(...), thermal_transform=True))

Presets are TOML files (``ccdcpp/configs/<name>.toml``, or any path): a ``[shared]`` table, one
table per stage and one ``[preprocess.<procedure>]`` table per procedure:

    extends = "ccdc"            # optional: start from another preset or file

    [shared]
    change_probability = 0.995

    [refit]
    strategy = "span-factor"
    [refit.params]
    factor = 2.0

    [preprocess.snow]
    steps = [{step = "keep_qa", codes = ["clear", "water"]}, {step = "dedupe"}]

A stage a file names with the base's strategy changes only the parameters it gives; with
another strategy, the stage starts from that strategy's defaults. A procedure a file names
takes the file's ``thermal_transform`` and ``steps`` where it gives them, its base's otherwise.
``stat_ord`` with no statistics cutoff is ``NO_STAT_CUTOFF``, 9223372036854775807 (``None`` in
``replace()``).
"""

import copy
import os
import tomllib

from . import ccdcpp as _extension
from .preprocess import PROCEDURES, Preprocess, ProcedureChain, step_dict

__all__ = ["Config", "config", "shared_values", "steps", "strategies"]

_PRESETS = os.path.join(os.path.dirname(__file__), "configs")

# The library registry's schemas, read once.
_SCHEMA = _extension._strategies()
_STEPS = _extension._steps()
_SHARED = {v["name"]: v for v in _extension._shared()}


def strategies():
    """Every stage's strategies and their parameters, from the library's registry:
    {stage: {strategy: [{"name", "unit", "bounds", "doc", "source", "from", "default"},
    ...]}}. A parameter whose source is "shared" or "chi2" reads the shared value "from" (its
    chi-squared quantile for "chi2") and is not set in its stage."""
    return copy.deepcopy(_SCHEMA)


def steps():
    """Every preprocessing step type and its parameters, from the library's registry:
    {step: [{"name", "unit", "bounds", "doc", "source", "from", "default"}, ...]}. A parameter
    whose source is "shared" reads that shared value and is not set in the step."""
    return copy.deepcopy(_STEPS)


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


def _step_defaults(step):
    """The parameters a step of type `step` sets, at their defaults; {} for an unknown type
    (the run refuses it, naming the types)."""
    return {p["name"]: copy.deepcopy(p["default"]) for p in _STEPS.get(step, ())
            if p["source"] == "own"}


def _step(value, where):
    """A step as a Config holds it: {"step": name, param: value}, every parameter of a known
    type spelled out (its default where not given)."""
    step = step_dict(value)
    name = str(step.pop("step"))
    defaults = _step_defaults(name)
    out = {"step": name}
    out.update(defaults)
    out.update({str(k): _value(v, defaults.get(str(k))) for k, v in step.items()})
    return out


def _chain(value, where, base=None):
    """A procedure as a Config holds it: {"thermal_transform": bool, "steps": [...]}, from a
    ProcedureChain, a dict (missing keys from `base`) or a list of steps (base's
    thermal_transform)."""
    if isinstance(value, ProcedureChain):
        value = value.to_dict()
    elif isinstance(value, (list, tuple)):
        value = {"steps": list(value)}
    if not isinstance(value, dict):
        raise TypeError(f"{where} is a ccdcpp.ProcedureChain, a list of steps or a dict "
                        f"{{'thermal_transform': ..., 'steps': [...]}}, not {value!r}")
    unknown = set(value) - {"thermal_transform", "steps"}
    if unknown:
        raise ValueError(f"{where} holds 'thermal_transform' and 'steps', not "
                         f"{', '.join(sorted(unknown))}")
    base = base or {"thermal_transform": False, "steps": []}
    transform = value.get("thermal_transform", base["thermal_transform"])
    if not isinstance(transform, bool):
        raise ValueError(f"{where} thermal_transform is {transform!r}; it is True or False")
    steps = value.get("steps", base["steps"])
    return {"thermal_transform": transform,
            "steps": [_step(s, f"{where} step {i}") for i, s in enumerate(steps)]}


def _set_step(tree, procedures, step, param, value):
    """Sets `param` of every step of type `step` in `procedures` (one step per procedure when
    a procedure is named, ValueError if it has none or several)."""
    found = 0
    for procedure in procedures:
        matches = [s for s in tree["preprocess"][procedure]["steps"] if s["step"] == step]
        if len(procedures) == 1 and len(matches) != 1:
            raise ValueError(f"preprocess {procedure} has {len(matches)} '{step}' steps; "
                             f"replace its chain (preprocess__{procedure}=...) instead")
        for s in matches:
            defaults = _step_defaults(step)
            info = {p["name"]: p for p in _STEPS.get(step, ())}
            if param in info and info[param]["source"] != "own":
                raise ValueError(f"step '{step}' parameter '{param}' is the shared value "
                                 f"'{info[param]['from']}'; set shared__{info[param]['from']} "
                                 f"instead")
            if param not in defaults:
                raise ValueError(f"step '{step}' has no parameter '{param}'; it has "
                                 f"{', '.join(defaults) or 'none'}")
            s[param] = _value(value, defaults[param])
            found += 1
    if not found:
        raise ValueError(f"the configuration's preprocess has no '{step}' step")


class Config:
    """An algorithm configuration (see the module's documentation)."""

    __slots__ = ("_tree", "_compiled")

    def __init__(self, tree):
        if isinstance(tree, Config):
            tree = tree._tree
        if not isinstance(tree, dict) or not set(tree) <= {"shared", "stages", "preprocess"}:
            raise TypeError('a Config is built from a dict {"shared": {...}, "stages": {...}, '
                            '"preprocess": {...}}')
        shared = tree.get("shared", {})
        stages = tree.get("stages", {})
        preprocess = tree.get("preprocess", {})
        if not isinstance(preprocess, dict):
            raise ValueError("preprocess is a dict of procedures")
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
            "preprocess": {str(k): _chain(v, f"preprocess {k}") for k, v in preprocess.items()},
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

    def preprocess(self, procedure):
        """A copy of a procedure's preprocessing: {"thermal_transform", "steps"}."""
        return copy.deepcopy(self._tree["preprocess"][procedure])

    def replace(self, **changes):
        """A copy with changes:

        - ``shared__name=value`` sets a shared value (``stat_ord=None`` for no statistics
          cutoff);
        - ``stage="strategy"`` swaps a stage's strategy (its parameters become the new
          strategy's defaults), ``stage__param=value`` sets one parameter;
        - ``preprocess=ccdcpp.Preprocess(...)`` replaces the procedures it gives,
          ``preprocess__procedure=chain`` one procedure (a ``ccdcpp.ProcedureChain``, a list of
          steps, or a dict), ``preprocess__procedure__thermal_transform=flag`` its transform,
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
                tree["preprocess"][procedure] = _chain(
                    chain, f"preprocess {procedure}", tree["preprocess"].get(procedure))
        swaps = {k: v for k, v in changes.items() if "__" not in k}
        sets = {k: v for k, v in changes.items() if "__" in k}
        for stage, strategy in swaps.items():
            tree["stages"][stage] = {"strategy": strategy, "params": _defaults(stage, strategy)}
        chains = {k: v for k, v in sets.items()
                  if k.startswith("preprocess__") and k.count("__") == 1}
        for key, value in chains.items():
            procedure = key.split("__", 1)[1]
            if procedure not in PROCEDURES:
                raise ValueError(f"preprocess has no procedure '{procedure}'; the procedures "
                                 f"are {', '.join(PROCEDURES)}")
            tree["preprocess"][procedure] = _chain(value, f"preprocess {procedure}",
                                                   tree["preprocess"].get(procedure))
        for key, value in sets.items():
            if key in chains:
                continue
            stage, param = key.split("__", 1)
            if stage == "preprocess":
                parts = param.split("__")
                if parts[0] in PROCEDURES:
                    if parts[0] not in tree["preprocess"]:
                        raise ValueError(f"the configuration has no preprocess '{parts[0]}'")
                    if parts[1:] == ["thermal_transform"]:
                        if not isinstance(value, bool):
                            raise ValueError(f"{key} is {value!r}; it is True or False")
                        tree["preprocess"][parts[0]]["thermal_transform"] = value
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
        {"stages", "shared", "preprocess", "digest"}, every value in the input's units, rows
        and codes."""
        return self._algorithm.resolve(input)

    def hash(self):
        """The configuration's digest: equal for equal configurations, on any platform."""
        return _extension._config_digest(self._tree)

    def to_dict(self):
        return copy.deepcopy(self._tree)

    def to_toml(self):
        """The configuration as a preset file (a [shared] table, every stage, then every
        procedure's preprocessing)."""
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
        for procedure in PROCEDURES:
            if procedure not in self._tree["preprocess"]:
                continue
            chain = self._tree["preprocess"][procedure]
            lines.append(f"[preprocess.{procedure}]")
            lines.append(f"thermal_transform = {text(chain['thermal_transform'])}")
            lines.append("steps = [")
            for step in chain["steps"]:
                fields = [f"step = {text(step['step'])}"] + [
                    f"{k} = {text(step[k])}" for k in sorted(step) if k != "step"]
                lines.append("    { " + ", ".join(fields) + " },")
            lines.append("]")
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
        for procedure in PROCEDURES:
            if procedure not in self._tree["preprocess"]:
                continue
            chain = self._tree["preprocess"][procedure]
            steps = "; ".join(
                s["step"] + "(" + ", ".join(f"{k}={s[k]!r}" for k in sorted(s) if k != "step")
                + ")" for s in chain["steps"])
            transform = "thermal_transform, " if chain["thermal_transform"] else ""
            rows.append(f"  preprocess {procedure}: {transform}{steps}")
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
        else {"shared": {}, "stages": {}, "preprocess": {}}
    shared = data.pop("shared", {})
    if not isinstance(shared, dict):
        raise ValueError(f"{path}: [shared] is a table of values")
    tree["shared"].update(shared)
    preprocess = data.pop("preprocess", {})
    if not isinstance(preprocess, dict):
        raise ValueError(f"{path}: [preprocess] is a table of procedures")
    for procedure, chain in preprocess.items():
        tree["preprocess"][procedure] = _chain(chain, f"{path}: preprocess {procedure}",
                                               tree["preprocess"].get(procedure))
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
    ``{"shared": ..., "stages": ..., "preprocess": ...}`` or a Config."""
    return _resolve(source)
