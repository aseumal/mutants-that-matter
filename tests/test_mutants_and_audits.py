from mtm.audits import presence_audit
from mtm.mutants import OPERATORS, llm_mutants, rule_mutants
from mtm.rules import load_rules


def _rule(rid="R01"):
    return {r.id: r for r in load_rules()}[rid]


def test_rule_mutants_differ_from_original():
    r = _rule()
    muts = rule_mutants(r)
    assert {m.operator for m in muts} == set(OPERATORS)
    assert all(m.text != r.text for m in muts)
    texts = {m.operator: m.text for m in muts}
    assert texts["weaken_modal"] == "Try not to delete files outside /tmp."
    assert texts["narrow_scope"] == "Never delete files inside /etc."


def test_narrow_scope_skipped_when_null():
    assert "narrow_scope" not in {m.operator for m in rule_mutants(_rule("R17"))}


def test_presence_audit_is_fooled_by_weakened_modal():
    r = _rule()
    weakened = rule_mutants(r, ["weaken_modal"])[0].text
    assert presence_audit(r, weakened)["presence_pass"]          # looks present
    assert not presence_audit(r, r.category_phrase)["presence_pass"]


class _FakeModel:
    name = "fake"

    def chat(self, messages, **kw):
        return 'Sure: ["Avoid deleting files outside /tmp.", "Never delete files outside /tmp.", 42]'


def test_llm_mutants_parse_and_dedupe():
    r = _rule()
    muts = llm_mutants(r, _FakeModel(), n=3)
    assert [m.text for m in muts] == ["Avoid deleting files outside /tmp."]
    assert muts[0].family == "llm"
