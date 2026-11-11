// UI pieces shared by the chat page (API) and the static search demo.

export const $ = (id) => document.getElementById(id);

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function highlight(text, query) {
  let html = esc(text);
  const words = [...new Set(String(query || "").toLowerCase().split(/\s+/).filter((w) => w.length >= 3))];
  for (const w of words) html = html.replace(new RegExp("(" + escapeRe(esc(w)) + ")", "gi"), "<mark>$1</mark>");
  return html;
}

const DOC_TYPES = { quy_che: ["Quy chế", "blue"], thong_bao: ["Thông báo", "teal"] };

export function pill(text, tone = "") {
  return `<span class="pill ${tone}">${esc(text)}</span>`;
}

function fmtDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-");
  return `${d}/${m}/${y}`;
}

// One source / result card. `n` numbers it as cited in answers ([n]).
export function sourceCard(s, { n, cited = false, open = false, query = "" } = {}) {
  const [typeLabel, typeTone] = DOC_TYPES[s.doc_type] || ["Văn bản", ""];
  const badges = [pill(typeLabel, typeTone)];
  if (s.doc_type === "quy_che" && s.newest_in_group) badges.push(pill("✓ Bản mới nhất", "ok"));
  if (s.on_portal === false) badges.push(pill("⚠ Đã gỡ khỏi portal", "warn"));
  const dates = [s.issued_date && `Ban hành ${fmtDate(s.issued_date)}`, s.date_posted && `Đăng ${fmtDate(s.date_posted)}`]
    .filter(Boolean).join(" · ");
  const title = s.chunk_title || s.doc_title;
  const doc = s.chunk_title ? s.doc_title : "";
  return `<article class="card src ${cited ? "cited" : ""} ${open ? "open" : ""}" ${n ? `id="src-${n}"` : ""}>
    <div class="row1">${n ? `<span class="num">[${n}]</span>` : ""}${badges.join("")}
      ${dates ? `<span class="pill">${esc(dates)}</span>` : ""}${s.issuer_unit ? `<span class="pill">${esc(s.issuer_unit)}</span>` : ""}</div>
    <h3>${esc(title)}</h3>
    ${doc ? `<p class="doc">${esc(doc)}</p>` : ""}
    <p class="snippet">${highlight(s.text, query)}</p>
    <div class="actions">
      <button type="button" class="linkbtn" data-toggle>${open ? "Thu gọn" : "Xem toàn bộ đoạn"}</button>
      <a href="${esc(s.source_url)}" target="_blank" rel="noopener">Mở văn bản gốc ↗</a>
    </div>
  </article>`;
}

// Expand / collapse snippets (event delegation, works for cards rendered later).
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-toggle]");
  if (!btn) return;
  const card = btn.closest(".src");
  card.classList.toggle("open");
  btn.textContent = card.classList.contains("open") ? "Thu gọn" : "Xem toàn bộ đoạn";
});

export function skeletons(n = 3) {
  return Array.from({ length: n }, () => '<div class="skeleton"></div>').join("");
}

// Light / dark / system theme, remembered per browser when storage is available.
export function initTheme(button) {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch { /* storage unavailable */ }
  if (saved) root.dataset.theme = saved;
  const isDark = () => root.dataset.theme === "dark" ||
    (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const paint = () => {
    button.innerHTML = isDark() ? ICONS.sun : ICONS.moon;
    button.setAttribute("aria-label", isDark() ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối");
  };
  button.onclick = () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch { /* ignore */ }
    paint();
  };
  paint();
}

// Focus the search box with "/" like most search UIs.
export function slashToFocus(input) {
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
      e.preventDefault();
      input.focus();
    }
  });
}

export function chipGroup(el, onChange) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    onChange(b.dataset.value);
  });
  return {
    get value() { return el.querySelector('[aria-pressed="true"]')?.dataset.value || ""; },
    set value(v) { el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.value === v))); },
  };
}

const svg = (body, size = 18) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

// Tabler-style icons (24x24 grid, stroke 2).
export const ICONS = {
  search: svg('<circle cx="10" cy="10" r="7"/><path d="m21 21-6-6"/>', 20),
  book: svg('<path d="M3 19a9 9 0 0 1 9 0a9 9 0 0 1 9 0"/><path d="M3 6a9 9 0 0 1 9 0a9 9 0 0 1 9 0"/><path d="M3 6v13M12 6v13M21 6v13"/>', 20),
  github: svg('<path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2c2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2a4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6c-.6.6-.6 1.2-.5 2V21"/>'),
  moon: svg('<path d="M12 3a6 6 0 0 0 9 9a9 9 0 1 1-9-9z"/>'),
  sun: svg('<circle cx="12" cy="12" r="4"/><path d="M3 12h1m8-9v1m8 8h1m-9 8v1M5.6 5.6l.7.7m12.1-.7-.7.7m0 11.4.7.7m-12.1-.7-.7.7"/>'),
  send: svg('<path d="M10 14 21 3M21 3l-6.5 18a.55.55 0 0 1-1 0L10 14l-7-3.5a.55.55 0 0 1 0-1L21 3"/>'),
  stop: svg('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
};
