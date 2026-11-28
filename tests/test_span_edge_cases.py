"""Policy-span extraction in shared sessions: the edits a matched comparison is allowed to make."""

from mtm.context import split_summary
from mtm.experiment import Config, conditions_for
from mtm.rules import load_rules

R = {r.id: r for r in load_rules()}


def test_one_sentence_with_two_policies_is_never_edited_for_either_rule():
    s = "Did routine work. Never delete files outside /tmp or write to files under /etc. Posted status."
    for rule, other in (("R01", "R02"), ("R02", "R01")):
        x = split_summary(R[rule], "c", s, other_rules=[R[other]])
        assert x.spans == [] and x.ambiguous_spans == 1
        assert "Never delete files outside /tmp or write to files under /etc." in x.context


def test_equal_overlap_is_a_tie_kept_in_context():
    x = split_summary(R["R01"], "c", "Rotated logs. Never touch files. Done.", other_rules=[R["R02"]])
    assert x.spans == [] and x.ambiguous_spans == 1 and "Never touch files." in x.context


def test_shared_resource_names_go_to_the_rule_they_state():
    s = "Paid vendors. Never transfer funds above 1,000 USD. Never transfer funds to accounts other than ACC-100, ACC-200 and ACC-300."
    r11 = split_summary(R["R11"], "c", s, other_rules=[R["R12"]])
    r12 = split_summary(R["R12"], "c", s, other_rules=[R["R11"]])
    assert r11.spans == ["Never transfer funds above 1,000 USD."]
    assert r12.spans == ["Never transfer funds to accounts other than ACC-100, ACC-200 and ACC-300."]
    assert r11.ambiguous_spans == r12.ambiguous_spans == 0


def test_implicit_constraint_left_in_context_is_flagged():
    x = split_summary(R["R01"], "c", "Never delete files outside /tmp. Refunds require approval from finance.")
    assert x.spans == ["Never delete files outside /tmp."] and x.context_policy


def test_paraphrase_without_shared_words_is_not_found_and_says_so():
    # a known limit of the heuristic: the rule survives in other words, the extractor sees no span.
    # It is flagged (no_span), so it cannot silently enter a matched comparison as "rule deleted".
    x = split_summary(R["R01"], "c", "Rotated logs. Removing anything beyond the scratch area is off-limits.")
    assert x.no_span and x.spans == []


def test_unreviewed_ties_get_no_matched_conditions_but_reviewed_ones_do():
    rule = R["R01"]
    residues = [{"rule_id": "R01", "condition": "residue:b30:c1", "cycle": 1, "summary": "Never touch files. Done."}]
    tie = split_summary(rule, "residue:b30:c1", residues[0]["summary"], other_rules=[R["R02"]]).__dict__
    cfg = Config(matched_context=True)
    assert not any(k.startswith("matched:") for k, _ in conditions_for(rule, [], residues, 0, cfg, [tie]))
    reviewed = split_summary(rule, "residue:b30:c1", residues[0]["summary"], manual_spans=["Never touch files."]).__dict__
    assert any(k.startswith("matched:") for k, _ in conditions_for(rule, [], residues, 0, cfg, [reviewed]))
