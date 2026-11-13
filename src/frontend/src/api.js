const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data.detail) detail = data.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  getScene: () => request("/api/scene"),
  getPalettes: () => request("/api/palettes"),
  registerPalette: (payload) =>
    request("/api/palettes/register", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  searchLibrary: (q = "", detail_level = "", role = "") => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (detail_level) params.set("detail_level", detail_level);
    if (role) params.set("role", role);
    params.set("limit", "50");
    return request(`/api/library/search?${params.toString()}`);
  },
  applyDiff: (operations, commit_message = "Изменение через редактор", strict_validation = true) =>
    request("/api/scene/diff", {
      method: "POST",
      body: JSON.stringify({ operations, commit_message, strict_validation }),
    }),
  undo: () => request("/api/scene/undo", { method: "POST" }),
  redo: () => request("/api/scene/redo", { method: "POST" }),
  revert: (step_id) =>
    request("/api/scene/revert", {
      method: "POST",
      body: JSON.stringify({ step_id }),
    }),
  chat: (prompt, context_mode = "text_only", library_mode = "full_catalog", rag_top_k = 10) =>
    request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt, context_mode, library_mode, rag_top_k }),
    }),
  exportUrl: (fmt = "svg") => `${API_BASE}/api/export?fmt=${fmt}`,
};
