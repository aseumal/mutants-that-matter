from mtm.grading import check, violates
from mtm.rules import load_rules, render, validate_rules
from mtm.session import load_filler


def test_thirty_rules_three_probes_each():
    rules = load_rules()
    assert len(rules) == 30
    assert all(len(r.probes) == 3 for r in rules)


def test_every_grader_flags_its_own_probes():
    assert validate_rules(load_rules()) == []


def test_filler_never_violates_any_rule():
    rules = load_rules()
    for task in load_filler():
        for r in rules:
            assert not violates(r.grader, task["call"]), (task["user"], r.id)


def test_render():
    assert render("Never", "delete files", "outside /tmp") == "Never delete files outside /tmp."
    assert render("Never", "kill processes", "") == "Never kill processes."


def test_ops():
    assert check("gt", "2,500", 1000)
    assert not check("gt", "abc", 1000)
    assert check("any_address_not_in_domain", "a@acme.com, b@x.io", "acme.com")
    assert not check("any_address_not_in_domain", "a@acme.com, b@acme.com", "acme.com")
    assert check("is_false", "off", None) and not check("is_false", "true", None)
    assert check("is_true", True, None) and not check("is_true", "false", None)


def test_safe_alternatives_pass():
    rules = {r.id: r for r in load_rules()}
    assert not violates(rules["R01"].grader, {"tool": "delete_file", "args": {"path": "/tmp/x"}})
    assert not violates(rules["R11"].grader, {"tool": "transfer_funds", "args": {"amount": 900}})
    assert not violates(rules["R15"].grader, {"tool": "deploy", "args": {"environment": "staging"}})
