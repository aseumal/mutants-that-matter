"""YAML reads an unquoted " #" as a comment, which silently truncates a rule or probe."""

import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')


@pytest.mark.parametrize("name", ["rules.yaml", "filler.yaml"])
def test_no_value_is_cut_off_by_a_yaml_comment(name):
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()
    cut = [f"{name}:{i}: {line.strip()}" for i, line in enumerate(lines, 1)
           if not line.lstrip().startswith("#") and " #" in _QUOTED.sub("", line)]
    assert not cut, "quote these values:\n" + "\n".join(cut)
