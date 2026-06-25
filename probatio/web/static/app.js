// probatio citation-check audit UI — no build, no deps.
const VERDICTS = ["supported", "partially", "overstated", "unsupported",
                  "not_found", "not_a_claim"];
const LABEL = {
  supported: "supported", partially: "partially", overstated: "overstated",
  unsupported: "unsupported", not_found: "not found", not_a_claim: "not a claim",
  no_pdf: "no pdf", ambiguous: "ambiguous", unresolved_marker: "unresolved",
  unreadable_source: "unreadable", unchecked: "unchecked",
};
const PROBLEMS = new Set(["unsupported", "overstated", "not_found", "partially"]);
const ORDER = { unsupported: 0, overstated: 1, not_found: 2, partially: 3,
  supported: 4, ambiguous: 5, no_pdf: 6, unresolved_marker: 7,
  unreadable_source: 8, not_a_claim: 9, unchecked: 10 };
const RESOLUTION_WHY = {
  no_pdf: "No reference PDF was found in the refs folder for this citation.",
  ambiguous: "The marker matched more than one reference PDF, so it couldn't be resolved.",
  unresolved_marker: "The in-text marker couldn't be matched to a bibliography entry.",
  unreadable_source: "The reference PDF couldn't be read.",
};
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
// Display-only cleanup of a verbatim PDF snippet: first join a word/compound that was
// hyphenated across a PDF line break (e.g. "multi- fidelity" -> "multi-fidelity") —
// conservative: only when a letter directly precedes the hyphen, so standalone dashes
// and ranges are left alone and the hyphen is always kept. Then collapse the PDF's
// physical line breaks into flowing prose. Never mutates the stored snippet (the
// highlight search depends on the original text); the page image keeps the true layout.
const cleanSnippet = (s) => esc(
  String(s == null ? "" : s)
    .replace(/([A-Za-z])-\s+([A-Za-z])/g, "$1-$2")
    .replace(/\s+/g, " ")
    .trim()
);

const api = {
  async citations() { return (await fetch("/api/citations")).json(); },
  async override(body) {
    return (await fetch("/api/override", { method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body) })).json();
  },
  pageImage(pid) { return `/api/page-image/${encodeURIComponent(pid)}`; },
};

const state = {
  manuscript: "", coverage: {}, checks: [],
  filters: { statuses: new Set(), section: "", query: "",
             preset: "problems", sort: "problems" },
  selectedId: null, passageIndex: 0, helpOpen: false,
};

const $ = (id) => document.getElementById(id);
const reviewedCount = () => state.checks.filter((c) => c.reviewed).length;
const selected = () => state.checks.find((c) => c.id === state.selectedId) || null;

function derive() {
  const f = state.filters;
  const q = f.query.trim().toLowerCase();
  const rows = state.checks.filter((c) => {
    if (f.preset === "problems" && !PROBLEMS.has(c.status)) return false;
    if (f.preset === "unreviewed" && c.reviewed) return false;
    if (f.statuses.size && !f.statuses.has(c.status)) return false;
    if (f.section && c.section !== f.section) return false;
    if (q) {
      const hay = [c.claim, c.ref_key, c.reference && c.reference.title]
        .filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  const cmp = f.sort === "manuscript"
    ? (a, b) => (a.manuscript_page ?? 1e9) - (b.manuscript_page ?? 1e9) ||
                a.id.localeCompare(b.id)
    : (a, b) => (ORDER[a.status] ?? 99) - (ORDER[b.status] ?? 99) ||
                a.id.localeCompare(b.id);
  return rows.sort(cmp);
}

// ---- actions ----
function select(id) { if (id) { state.selectedId = id; state.passageIndex = 0; } render(); }
function move(d) {
  const rows = derive(); if (!rows.length) return;
  let i = rows.findIndex((c) => c.id === state.selectedId);
  i = i < 0 ? 0 : Math.min(Math.max(i + d, 0), rows.length - 1);
  select(rows[i].id);
}
function nextUnreviewed() {
  const rows = derive(); if (!rows.length) return;
  const i = rows.findIndex((c) => c.id === state.selectedId);
  for (let k = 1; k <= rows.length; k++) {
    const c = rows[(i + k) % rows.length];
    if (!c.reviewed) { select(c.id); return; }
  }
}
function navPassage(d) {
  const c = selected(); const ps = c ? (c.passages || []) : [];
  if (ps.length <= 1) return;
  state.passageIndex = Math.min(Math.max(state.passageIndex + d, 0), ps.length - 1);
  renderEvidence();
}
async function setVerdict(v) {
  const c = selected(); if (!c) return;
  c.human_override = v; c.reviewed = true; render();
  const res = await api.override({ id: c.id, verdict: v, note: c.note || "" });
  if (res) {
    if (res.status) c.status = res.status;
    if (typeof res.reviewed === "boolean") c.reviewed = res.reviewed;
    render();
  }
}
async function clearOverride() {
  const c = selected(); if (!c) return;
  c.human_override = null; render();
  const res = await api.override({ id: c.id, clear_override: true, note: c.note || "" });
  if (res) {
    if (res.status) c.status = res.status;
    if (typeof res.reviewed === "boolean") c.reviewed = res.reviewed;
    render();
  }
}
async function toggleReviewed() {
  const c = selected(); if (!c) return;
  c.reviewed = !c.reviewed; render();
  const res = await api.override({ id: c.id, reviewed: c.reviewed, note: c.note || "" });
  if (res && typeof res.reviewed === "boolean") { c.reviewed = res.reviewed; render(); }
}
async function saveNote(text) {
  const c = selected(); if (!c) return;
  c.note = text;
  await api.override({ id: c.id, note: text });
}
function clearFilters() {
  state.filters.statuses.clear();
  state.filters.section = ""; state.filters.query = ""; state.filters.preset = "all";
  render();
}

// ---- render ----
const chip = (s) =>
  `<span class="chip v-${esc(s)}"><span class="dot"></span>${esc(LABEL[s] || s)}</span>`;

function render() {
  renderHeader(); renderFilters(); renderQueue(); renderDetail();
  renderEvidence(); renderHelp();
}
function renderHeader() {
  const total = Object.values(state.coverage).reduce((a, b) => a + b, 0) || 1;
  $("coverage").innerHTML = Object.entries(state.coverage)
    .sort((a, b) => (ORDER[a[0]] ?? 99) - (ORDER[b[0]] ?? 99))
    .map(([k, v]) => `<span class="seg v-${esc(k)}" style="width:${(100 * v / total)
      .toFixed(2)}%" title="${esc(k)}=${v}"></span>`).join("");
  const rev = reviewedCount();
  $("progress-text").textContent = `${state.checks.length} citations · ${rev} reviewed`;
  $("progress-fill").style.width = state.checks.length
    ? `${Math.round(100 * rev / state.checks.length)}%` : "0%";
}
function renderFilters() {
  for (const b of document.querySelectorAll(".preset"))
    b.classList.toggle("on", b.dataset.preset === state.filters.preset);
  const counts = {};
  for (const c of state.checks) counts[c.status] = (counts[c.status] || 0) + 1;
  $("status-pills").innerHTML = Object.keys(counts)
    .sort((a, b) => (ORDER[a] ?? 99) - (ORDER[b] ?? 99))
    .map((s) => `<button class="pill v-${esc(s)} ${state.filters.statuses.has(s)
      ? "on" : ""}" data-status="${esc(s)}"><span class="dot"></span>${esc(LABEL[s] || s)} <b>${counts[s]}</b></button>`).join("");
  for (const b of document.querySelectorAll("#status-pills .pill"))
    b.onclick = () => {
      const set = state.filters.statuses; const s = b.dataset.status;
      set.has(s) ? set.delete(s) : set.add(s); render();
    };
  $("section-filter").value = state.filters.section;
  $("sort-mode").value = state.filters.sort;
  const sb = $("search");
  if (document.activeElement !== sb) sb.value = state.filters.query;
}
function renderQueue() {
  const ol = $("rows");
  if (!state.checks.length) { ol.innerHTML = `<li class="empty">No citations in this report.</li>`; return; }
  const rows = derive();
  if (!rows.length) {
    ol.innerHTML = `<li class="empty">No citations match.<br><button id="clear-filters">Clear filters</button></li>`;
    $("clear-filters").onclick = clearFilters; return;
  }
  ol.innerHTML = rows.map((c) => `
    <li class="row ${c.id === state.selectedId ? "sel" : ""} ${c.kind === "non_checkable" || c.status === "not_a_claim" ? "muted" : ""}" data-id="${esc(c.id)}">
      <div class="row-top">${chip(c.status)}<span class="rk">[${esc(c.ref_key)}]</span>${c.reviewed ? '<span class="rev">✓</span>' : ""}</div>
      <div class="row-claim">${esc(c.claim)}</div>
      <div class="row-meta">${esc(c.section || "—")}</div>
    </li>`).join("");
  for (const li of ol.querySelectorAll(".row")) li.onclick = () => select(li.dataset.id);
  const sel = ol.querySelector(".row.sel"); if (sel) sel.scrollIntoView({ block: "nearest" });
}
function renderDetail() {
  const d = $("detail"); const c = selected();
  if (!c) { d.innerHTML = `<div class="placeholder">Select a citation.</div>`; return; }
  const conf = c.confidence
    ? `<span class="conf"><span class="conf-bar"><span style="width:${Math.round(100 * c.confidence)}%"></span></span>${Math.round(100 * c.confidence)}%</span>` : "";
  const overridden = c.human_override
    ? `<div class="ov-note">overridden from <b>${esc(LABEL[c.verdict] || c.verdict)}</b></div>` : "";
  d.innerHTML = `
    <div class="claim">${esc(c.claim)}</div>
    <div class="claim-meta">${esc(c.section || "—")} · p.${c.manuscript_page ?? "—"} · ${esc(c.kind)}</div>
    <div class="vblock">
      <div class="vhead">${chip(c.status)}${conf}</div>
      ${overridden}
      ${c.rationale ? `<div class="rationale">${esc(c.rationale)}</div>` : ""}
      <div class="overrides">
        ${VERDICTS.map((v, i) => `<button class="ovb ${c.human_override === v ? "on" : ""}" data-v="${v}"><kbd>${i + 1}</kbd>${esc(LABEL[v])}</button>`).join("")}
        <button class="ovb" data-clear><kbd>0</kbd>clear</button>
        <button class="ovb confirm ${c.reviewed ? "on" : ""}" data-confirm><kbd>r</kbd>${c.reviewed ? "reviewed ✓" : "confirm"}</button>
      </div>
      <textarea id="note" placeholder="note…" rows="2">${esc(c.note || "")}</textarea>
    </div>
    ${referenceCard(c)}`;
  for (const b of d.querySelectorAll(".ovb[data-v]")) b.onclick = () => setVerdict(b.dataset.v);
  d.querySelector("[data-clear]").onclick = clearOverride;
  d.querySelector("[data-confirm]").onclick = toggleReviewed;
  const note = $("note");
  note.onblur = () => saveNote(note.value);
  note.onkeydown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") note.blur();
    if (e.key === "Escape") { note.value = c.note || ""; note.blur(); }
  };
  const doi = d.querySelector(".doi");
  if (doi) doi.onclick = () => navigator.clipboard && navigator.clipboard.writeText(doi.dataset.doi);
}
function referenceCard(c) {
  if (c.resolution !== "resolved") {
    return `<div class="refcard unresolved"><div class="rc-title">No source to check</div>
      <div class="rc-why">${esc(RESOLUTION_WHY[c.resolution] || c.resolution)}</div>
      <div class="rc-line">resolution: <code>${esc(c.resolution)}</code></div></div>`;
  }
  const r = c.reference || {};
  const authors = (r.authors && r.authors.length)
    ? (r.authors.length > 1 ? `${esc(r.authors[0])} et al.` : esc(r.authors[0])) : "";
  return `<div class="refcard"><div class="rc-key">reference [${esc(c.ref_key)}]</div>
    <div class="rc-title">${esc(r.title || r.raw || "—")}</div>
    <div class="rc-line">${authors}${r.year ? ` (${esc(r.year)})` : ""}</div>
    ${r.doi ? `<div class="rc-line"><code class="doi" data-doi="${esc(r.doi)}" title="click to copy">${esc(r.doi)}</code></div>` : ""}
    <div class="rc-line src">source: ${esc(c.source || "—")}</div></div>`;
}
function renderEvidence() {
  const e = $("evidence"); const c = selected();
  if (!c) { e.innerHTML = ""; return; }
  const ps = c.passages || [];
  if (!ps.length) {
    const msg = c.resolution !== "resolved"
      ? "No source resolved for this citation."
      : "No supporting passage was retrieved from the source.";
    e.innerHTML = `<div class="ev-empty">${esc(msg)}</div>`; return;
  }
  state.passageIndex = Math.min(state.passageIndex, ps.length - 1);
  const p = ps[state.passageIndex];
  const nav = ps.length > 1
    ? `<div class="passage-nav"><button data-pp="-1">‹</button><span>Passage ${state.passageIndex + 1} of ${ps.length} · score ${(p.score || 0).toFixed(2)}</span><button data-pp="1">›</button></div>`
    : `<div class="passage-nav single"><span>Passage 1 of 1 · score ${(p.score || 0).toFixed(2)}</span></div>`;
  e.innerHTML = `${nav}
    <div class="ev-image"><img alt="source page"></div>
    <div class="snippet">${cleanSnippet(p.snippet)}</div>
    ${p.rcs_summary ? `<details class="summary"><summary>contextual summary</summary><div>${cleanSnippet(p.rcs_summary)}</div></details>` : ""}`;
  const img = e.querySelector(".ev-image img");
  img.onerror = () => { img.closest(".ev-image").innerHTML =
    `<div class="img-fail">Couldn't render this page — the verbatim snippet below is the source.</div>`; };
  img.src = api.pageImage(p.id);
  for (const b of e.querySelectorAll("[data-pp]"))
    b.onclick = () => navPassage(parseInt(b.dataset.pp, 10));
}

const HELP = [
  ["j / k", "next / prev citation"], ["[ / ]", "prev / next passage"],
  ["1–6", "set verdict override"], ["0", "clear override"],
  ["r", "confirm / toggle reviewed"], ["n", "next unreviewed"],
  ["/", "focus search"], ["e / Enter", "edit note (⌘/Ctrl+Enter save, Esc cancel)"],
  ["?", "toggle this help"], ["Esc", "blur input / close help"],
];
function toggleHelp(force) {
  state.helpOpen = force === undefined ? !state.helpOpen : force; renderHelp();
}
function renderHelp() {
  const o = $("help-overlay"); o.hidden = !state.helpOpen;
  if (!state.helpOpen) return;
  o.innerHTML = `<div class="help-card"><h2>Keyboard</h2><table>${
    HELP.map(([k, v]) => `<tr><td><kbd>${esc(k)}</kbd></td><td>${esc(v)}</td></tr>`).join("")
  }</table><button id="help-close">Close</button></div>`;
  $("help-close").onclick = () => toggleHelp(false);
  o.onclick = (ev) => { if (ev.target === o) toggleHelp(false); };
}

const typing = () => {
  const a = document.activeElement;
  return a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT");
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.helpOpen) { toggleHelp(false); return; }
  if (typing()) return;
  if (/^[1-6]$/.test(e.key)) { setVerdict(VERDICTS[parseInt(e.key, 10) - 1]); return; }
  switch (e.key) {
    case "j": case "ArrowDown": e.preventDefault(); move(1); break;
    case "k": case "ArrowUp": e.preventDefault(); move(-1); break;
    case "[": navPassage(-1); break;
    case "]": navPassage(1); break;
    case "n": nextUnreviewed(); break;
    case "r": toggleReviewed(); break;
    case "0": clearOverride(); break;
    case "/": e.preventDefault(); $("search").focus(); break;
    case "e": case "Enter": { const n = $("note"); if (n) { e.preventDefault(); n.focus(); } break; }
    case "?": toggleHelp(); break;
  }
});

function wire() {
  for (const b of document.querySelectorAll(".preset"))
    b.onclick = () => { state.filters.preset = b.dataset.preset; render(); };
  $("search").addEventListener("input", (e) => { state.filters.query = e.target.value; render(); });
  $("section-filter").addEventListener("change", (e) => { state.filters.section = e.target.value; render(); });
  $("sort-mode").addEventListener("change", (e) => { state.filters.sort = e.target.value; render(); });
  $("help-btn").onclick = () => toggleHelp();
}
async function init() {
  wire();
  const d = await api.citations();
  state.manuscript = d.manuscript; state.coverage = d.coverage || {};
  state.checks = d.checks || [];
  document.title = `check · ${d.manuscript}`;
  const secs = [...new Set(state.checks.map((c) => c.section).filter(Boolean))].sort();
  $("section-filter").innerHTML = `<option value="">All sections</option>` +
    secs.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  const first = derive()[0]; state.selectedId = first ? first.id : null;
  render();
}
init();
