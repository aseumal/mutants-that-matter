// Pure logic for the local labeling page. No DOM access here, so it is unit-tested in Node
// (tests/js/label_tool.test.mjs). The page inlines this file at build time.
"use strict";

const LT = {};

// RFC 4180 CSV: quoted fields may hold commas, doubled quotes and line breaks.
LT.parseCSV = function (text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [], field = "", i = 0, quoted = false;
  while (i < text.length) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        quoted = false; i++; continue;
      }
      field += c; i++; continue;
    }
    if (c === '"') { quoted = true; i++; continue; }
    if (c === ",") { row.push(field); field = ""; i++; continue; }
    if (c === "\r" || c === "\n") {
      row.push(field); field = "";
      rows.push(row); row = [];
      if (c === "\r" && text[i + 1] === "\n") i++;
      i++; continue;
    }
    field += c; i++;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const nonEmpty = rows.filter((r) => !(r.length === 1 && r[0] === ""));
  if (!nonEmpty.length) return { header: [], records: [] };
  const header = nonEmpty[0];
  const records = nonEmpty.slice(1).map((r) => {
    const o = {};
    header.forEach((h, k) => { o[h] = r[k] === undefined ? "" : r[k]; });
    return o;
  });
  return { header, records };
};

LT.quote = function (value) {
  const s = value === undefined || value === null ? "" : String(value);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

// Same header order and line ending as Python's csv.DictWriter.
LT.toCSV = function (records, header) {
  const lines = [header.map(LT.quote).join(",")];
  for (const r of records) lines.push(header.map((h) => LT.quote(r[h])).join(","));
  return lines.join("\r\n") + "\r\n";
};

LT.LABEL_KEYS = ["damage_type", "semantic_preservation", "surface_presence", "annotation_confidence"];

LT.isComplete = function (row) {
  return LT.LABEL_KEYS.every((k) => row[k]);
};

LT.isStarted = function (row) {
  return LT.LABEL_KEYS.some((k) => row[k]) || Boolean((row.notes || "").trim());
};

// Soft warnings only: they never block saving or export.
LT.warnings = function (row) {
  const w = [];
  const d = row.damage_type, p = row.semantic_preservation, s = row.surface_presence;
  const notes = (row.notes || "").trim();
  if (LT.isStarted(row) && !LT.isComplete(row)) w.push("Some label fields are still empty.");
  if (d === "deleted" && (p === "intact" || p === "partial"))
    w.push("'deleted' means nothing about the rule remains, so preservation is usually 'absent'.");
  if (d === "deleted" && s === "yes")
    w.push("'deleted' with a visible rule ('surface: yes') looks inconsistent.");
  if (d === "none" && p && p !== "intact" && p !== "uncertain")
    w.push("'none' means the obligation is fully kept, so preservation is usually 'intact'.");
  if (p === "intact" && s === "no")
    w.push("An intact rule is normally visible in the text ('surface: yes').");
  if ((d === "other" || d === "uncertain" || p === "uncertain") && !notes)
    w.push("'other' and 'uncertain' need a note explaining why.");
  return w;
};

// Merge an imported CSV into the sheet. Only label columns are taken from the import, and a
// row is refused if its id is unknown or its rule or summary text differs from the sheet.
LT.mergeImport = function (base, imported) {
  const byId = new Map(base.map((r) => [r.residue_id, r]));
  const out = base.map((r) => Object.assign({}, r));
  const index = new Map(out.map((r, k) => [r.residue_id, k]));
  const refused = [];
  let merged = 0;
  for (const r of imported) {
    const orig = byId.get(r.residue_id);
    if (!orig) { refused.push(`${r.residue_id || "(no id)"}: not in this sheet`); continue; }
    if (r.original_rule !== orig.original_rule || r.summary !== orig.summary || r.rule_id !== orig.rule_id) {
      refused.push(`${r.residue_id}: rule or summary text differs from the sheet`); continue;
    }
    const target = out[index.get(r.residue_id)];
    for (const k of LT.LABEL_KEYS.concat(["notes"])) target[k] = r[k] || "";
    merged++;
  }
  return { rows: out, merged, refused };
};

// Values outside the allowed lists (from mtm/analysis.py, embedded at build time).
LT.invalidValues = function (row, allowed) {
  const bad = [];
  for (const k of LT.LABEL_KEYS) {
    if (row[k] && !allowed[k].includes(row[k])) bad.push(`${k}=${row[k]}`);
  }
  return bad;
};

if (typeof module !== "undefined" && module.exports) module.exports = LT;
