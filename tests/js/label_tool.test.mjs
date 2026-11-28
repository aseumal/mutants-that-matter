// Unit tests for mtm/label_tool.js. Run by tests/test_labeltool.py through Node.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
const require = createRequire(import.meta.url);
const LT = require(process.argv[2]);

const header = ["residue_id", "rule_id", "original_rule", "summary", ...LT.LABEL_KEYS, "notes", "annotator", "label_schema"];
const DELETED = { surface_presence: "absent", semantic_preservation: "absent", obligation_strength: "not_applicable",
                  scope: "not_applicable", exceptions: "not_applicable", contradiction: "no", overall_damage: "deletion",
                  damage_type: "deleted", annotation_confidence: "high" };
const INTACT = { surface_presence: "present", semantic_preservation: "intact", obligation_strength: "preserved",
                 scope: "preserved", exceptions: "preserved", contradiction: "no", overall_damage: "intact",
                 damage_type: "none", annotation_confidence: "high" };
const tricky = {
  residue_id: "r001", rule_id: "R01", original_rule: 'Never "delete", files outside /tmp.',
  summary: 'Line one, with comma\nLine two "quoted" — accénts 你好\r\nLine three', ...INTACT,
  notes: 'said "keep", ok', annotator: "AS", label_schema: "2",
};
const plain = Object.fromEntries(header.map((h) => [h, ""]));
Object.assign(plain, { residue_id: "r002", rule_id: "R02", original_rule: "Never write to files under /etc.", summary: "Tasks done." });

// 1. round trip preserves every character
const csv = LT.toCSV([tricky, plain], header);
const back = LT.parseCSV(csv);
assert.deepEqual(back.header, header);
assert.deepEqual(back.records[0], tricky);
assert.deepEqual(back.records[1], plain);
assert.ok(csv.endsWith("\r\n"));

// 2. BOM and LF-only input still parse
assert.deepEqual(LT.parseCSV("﻿a,b\n1,2\n").records, [{ a: "1", b: "2" }]);

// 3. import merges only label columns, refuses changed text and unknown ids
const base = [Object.assign({}, plain, { residue_id: "r001", summary: "S1", rule_id: "R01", original_rule: "Rule 1" }),
              Object.assign({}, plain)];
const imported = [
  Object.assign({}, base[0], DELETED, { notes: "task list only" }),
  Object.assign({}, base[1], { summary: "EDITED", damage_type: "none" }),
  Object.assign({}, base[1], { residue_id: "r999" }),
];
const m = LT.mergeImport(base, imported);
assert.equal(m.merged, 1);
assert.equal(m.refused.length, 2);
assert.equal(m.rows[0].damage_type, "deleted");
assert.equal(m.rows[0].summary, "S1");
assert.equal(m.rows[1].damage_type, "");
assert.equal(base[0].damage_type, "", "mergeImport must not mutate its input");

// 4. completeness and warnings
assert.ok(LT.isComplete(tricky));
assert.ok(!LT.isComplete(plain) && !LT.isStarted(plain));
const w = (o) => LT.warnings(Object.assign({}, plain, o));
assert.equal(w({}).length, 0);
assert.ok(w({ damage_type: "deleted" }).some((x) => x.includes("still empty")));
assert.ok(w(Object.assign({}, DELETED, { semantic_preservation: "intact" })).length >= 1);
assert.ok(w(Object.assign({}, DELETED, { scope: "narrowed" })).some((x) => x.includes("not_applicable")));
assert.ok(w(Object.assign({}, INTACT, { damage_type: "other" })).some((x) => x.includes("note")));
assert.ok(w(Object.assign({}, INTACT, { contradiction: "yes" })).some((x) => x.includes("note")));
assert.equal(w(DELETED).length, 0);
assert.equal(w(INTACT).length, 0);

// 5. invalid values are reported
const allowed = Object.fromEntries(LT.LABEL_KEYS.map((k) => [k, [INTACT[k]]]));
assert.deepEqual(LT.invalidValues(Object.assign({}, plain, { damage_type: "nonsense" }), allowed), ["damage_type=nonsense"]);

// 6. export of a fully labelled embedded sheet, for the Python round-trip test
if (process.argv[3]) {
  const fs = await import("node:fs");
  const html = fs.readFileSync(process.argv[3], "utf8");
  const json = html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
  const data = JSON.parse(json);
  const rows = data.rows.map((r, k) => Object.assign({}, r,
    Object.fromEntries(data.label_keys.map((key) => [key, data.allowed[key][k % data.allowed[key].length]])),
    { notes: k % 2 ? 'quote " and, comma' : "", annotator: "AS", label_schema: data.schema_version }));
  fs.writeFileSync(process.argv[4], LT.toCSV(rows, data.fields));
}
console.log("label_tool.js: all tests passed");
