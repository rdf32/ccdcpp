"""A new strategy's files, scaffolded from its stage's interface.

    python tools/new_strategy.py STAGE NAME --into DIR [--class CLASS]
    python tools/new_strategy.py --list

writes DIR/STAGE/FILE.hpp and DIR/STAGE/FILE.cpp (FILE is NAME with '-' as '_'): a class CLASS
(NAME in CamelCase) for the stage's interface in include/ccd/strategies/stages.hpp, with an empty
Params and schema, every method the interface leaves to its strategies declared, and each body
throwing until it is written; and a registration function registering it as NAME. It adds the
source and the registration to DIR/extension.cmake (written when DIR has none), so that
`cmake -DCCD_EXTENSIONS=DIR` builds the strategy into the library. Existing files are never
overwritten. --list prints the stages and their interfaces.

The interface is read from stages.hpp, so the scaffold follows it; the build checks every
stage's (tests/ccd/strategies/test_scaffold.cpp).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

STAGES_HPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "include",
                          "ccd", "strategies", "stages.hpp")


@dataclass(frozen=True)
class Method:
    """A method the interface leaves to its strategies: `virtual RETURNS NAME(PARAMS) QUALIFIERS
    = 0`."""
    returns: str
    name: str
    params: list[str]
    qualifiers: str


@dataclass(frozen=True)
class Interface:
    """A stage's interface: its struct, the stage's name and its pure virtual methods."""
    struct: str
    stage: str
    methods: list[Method]


def _without_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _body(text: str, open_brace: int) -> str:
    """The text between the brace at `open_brace` and its match."""
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise ValueError("unbalanced braces in stages.hpp")


def _split_params(params: str) -> list[str]:
    """`params` split at its top-level commas (a template's are not)."""
    out: list[str] = []
    depth = 0
    current = ""
    for c in params:
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        out.append(current.strip())
    return out


def _method(declaration: str) -> Method | None:
    """The method of `declaration` (from `virtual` to its `;`), or None for a destructor or a
    method with a default."""
    text = " ".join(declaration.split())
    m = re.fullmatch(r"virtual (.+?) (\w+)\((.*)\)\s*(const)?\s*=\s*0", text)
    if not m:
        return None
    return Method(m.group(1), m.group(2), _split_params(m.group(3)), m.group(4) or "")


def interfaces(path: str = STAGES_HPP) -> list[Interface]:
    """Every stage interface of stages.hpp, in its order there."""
    with open(path, encoding="utf-8") as f:
        text = _without_comments(f.read())
    found: list[Interface] = []
    for m in re.finditer(r"\bstruct (\w+)\s*\{", text):
        body = _body(text, m.end() - 1)
        stage = re.search(r'static constexpr const char\* stage = "(\w+)";', body)
        if not stage:
            continue
        methods: list[Method] = []
        for declaration in re.findall(r"\bvirtual\b[^;{]*", body):
            method = _method(declaration)
            if method:
                methods.append(method)
        found.append(Interface(m.group(1), stage.group(1), methods))
    return found


def _unnamed(param: str) -> str:
    """`param` with its name as a comment, for a body that does not read it yet."""
    m = re.fullmatch(r"(.*?[\s&*>])(\w+)", param)
    return f"{m.group(1).rstrip()} /*{m.group(2)}*/" if m else param


def _signature(method: Method, prefix: str, params: list[str]) -> str:
    qualifiers = f" {method.qualifiers}" if method.qualifiers else ""
    return f"{method.returns} {prefix}{method.name}({', '.join(params)}){qualifiers}"


def header(interface: Interface, name: str, cls: str, registration: str) -> str:
    """The strategy's header."""
    declarations = "\n".join(f"    {_signature(m, '', m.params)} override;"
                             for m in interface.methods)
    return f"""#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{{
namespace config
{{
class Registry;
}}

namespace strategies
{{

//------------------------------------------------------------------------------
// {interface.stage} "{name}": what it decides, in a sentence or two.
//------------------------------------------------------------------------------
struct {cls} final : {interface.struct}
{{
    struct Params
    {{
    }};

    static const std::vector<config::Field<Params>>& schema();

    explicit {cls}(const Params& p) : p_(p) {{}}

{declarations}

private:
    [[maybe_unused]] Params p_;   // drop [[maybe_unused]] once a method reads it
}};

// Registers {cls} as {interface.stage} "{name}".
void {registration}(config::Registry& registry);

}} // namespace strategies
}} // namespace ccd
"""


def source(interface: Interface, name: str, cls: str, registration: str, file: str) -> str:
    """The strategy's source: its schema, a body per method and its registration."""
    bodies = "\n\n".join(
        f"""{_signature(m, cls + '::', [_unnamed(p) for p in m.params])}
{{
    throw std::logic_error("{interface.stage} '{name}': {m.name}() is not written yet");
}}""" for m in interface.methods)
    return f"""#include "{file}.hpp"

#include <stdexcept>

#include "ccd/config/registry.hpp"
#include "ccd/strategies/model.hpp"

namespace ccd
{{
namespace strategies
{{

const std::vector<config::Field<{cls}::Params>>& {cls}::schema()
{{
    static const std::vector<config::Field<Params>> s = {{
        // {{"name", &Params::member, config::Unit::None, config::any(), "what it sets"}},
    }};
    return s;
}}

{bodies}

void {registration}(config::Registry& registry)
{{
    registry.add<{interface.struct}, {cls}>("{name}");
}}

}} // namespace strategies
}} // namespace ccd
"""


EXTENSION_HEADER = """\
# A strategy extension: built into the library by
#   cmake -DCCD_EXTENSIONS=<this directory, absolute or relative to the source directory>
# CMake sets CCD_EXTENSION_DIR to this directory while it reads this file.
list(APPEND CCD_EXTENSION_INCLUDE_DIRS ${CCD_EXTENSION_DIR})
"""


def write(stage: str, name: str, into: str, cls: str | None = None,
          path: str = STAGES_HPP) -> list[str]:
    """Writes the strategy's files into `into` and adds them to its extension.cmake. Returns
    the paths written. Raises ValueError for an unknown stage, a bad name or class, and
    FileExistsError for a file that exists."""
    by_stage = {i.stage: i for i in interfaces(path)}
    if stage not in by_stage:
        raise ValueError(f"no stage '{stage}'; the stages are {', '.join(by_stage)}")
    if not re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", name):
        raise ValueError(f"'{name}' is not a strategy name: lower case words joined by '-'")
    file = name.replace("-", "_")
    cls = cls or "".join(word.capitalize() for word in name.split("-"))
    if not re.fullmatch(r"[A-Z]\w*", cls):
        raise ValueError(f"'{cls}' is not a class name")
    registration = f"register_{stage}_{file}"
    interface = by_stage[stage]

    directory = os.path.join(into, stage)
    hpp = os.path.join(directory, file + ".hpp")
    cpp = os.path.join(directory, file + ".cpp")
    for p in (hpp, cpp):
        if os.path.exists(p):
            raise FileExistsError(f"{p} exists")
    os.makedirs(directory, exist_ok=True)
    with open(hpp, "w", encoding="utf-8") as f:
        f.write(header(interface, name, cls, registration))
    with open(cpp, "w", encoding="utf-8") as f:
        f.write(source(interface, name, cls, registration, file))

    extension = os.path.join(into, "extension.cmake")
    new = not os.path.exists(extension)
    with open(extension, "a", encoding="utf-8") as f:
        if new:
            f.write(EXTENSION_HEADER)
        f.write(f"list(APPEND CCD_EXTENSION_SOURCES ${{CCD_EXTENSION_DIR}}/{stage}/{file}.cpp)\n"
                f"list(APPEND CCD_EXTENSION_REGISTRATIONS {registration})\n")
    return [hpp, cpp, extension]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", help="the stage, as a configuration names it")
    p.add_argument("name", nargs="?", help="the strategy's name, e.g. residual-cap")
    p.add_argument("--into", help="the extension directory the files go in")
    p.add_argument("--class", dest="cls", help="the class name (default: NAME in CamelCase)")
    p.add_argument("--list", action="store_true", help="print the stages and their interfaces")
    args = p.parse_args(argv)
    if args.list:
        for i in interfaces():
            print(f"{i.stage:16} {i.struct}: {', '.join(m.name + '()' for m in i.methods)}")
        return 0
    if not (args.stage and args.name and args.into):
        p.error("STAGE, NAME and --into are required")
    try:
        written = write(args.stage, args.name, args.into, args.cls)
    except (ValueError, FileExistsError) as e:
        print(f"new_strategy.py: {e}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
