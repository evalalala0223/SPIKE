"""Knowledge base + QA over an offline Stardew Valley wiki (and optional web).

Source wiki:
    https://github.com/cristinakity/offline-stardew-valley-wiki
    (tree: src/stardewvalleywiki.com -> the mirrored html/markdown pages)

Strategy (offline-first, no heavy deps):
  1. ``sync()`` shallow-clones the wiki repo into a local cache directory.
  2. Pages are indexed by filename + a stripped-text token index.
  3. ``retrieve()`` does lightweight scored substring/token matching to fetch the
     most relevant page snippets for a query.
  4. ``answer()`` (optional) feeds the retrieved snippets to the LLM for a concise
     grounded answer; if no LLM is available it returns the raw snippets.
  5. ``web_search`` hook: a caller-supplied callable used as a fallback when the
     offline wiki has no good hit. Kept as a hook so the env agent stays
     dependency-light and the search backend is pluggable.
"""

from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

WIKI_REPO = "https://github.com/cristinakity/offline-stardew-valley-wiki"
WIKI_SUBDIR = "src/stardewvalleywiki.com"

# callable(query: str, k: int) -> list[{"title","url","snippet"}]
WebSearchFn = Callable[[str, int], list[dict[str, str]]]


@dataclass
class Snippet:
    source: str          # "wiki:<file>" or "web:<url>"
    title: str
    text: str
    score: float


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KnowledgeBase:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        llm: Any = None,
        web_search: Optional[WebSearchFn] = None,
        max_files: int = 4000,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.llm = llm
        self.web_search = web_search
        self.max_files = max_files
        self._index: list[tuple[Path, str]] = []  # (path, lowered filename tokens)
        self._loaded = False

    # -- syncing -------------------------------------------------------------
    @property
    def wiki_root(self) -> Path:
        return self.cache_dir / "offline-stardew-valley-wiki" / WIKI_SUBDIR

    def is_synced(self) -> bool:
        return self.wiki_root.exists() and any(self.wiki_root.rglob("*"))

    def sync(self, *, depth: int = 1) -> str:
        """Shallow-clone the offline wiki. Returns a status string.

        Network/git operations are explicit and only run when this is called.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / "offline-stardew-valley-wiki"
        if dest.exists():
            return f"wiki already present at {dest}"
        cmd = ["git", "clone", "--depth", str(depth), WIKI_REPO, str(dest)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{proc.stderr[-800:]}")
        return f"cloned wiki into {dest}"

    # -- indexing ------------------------------------------------------------
    def _ensure_index(self) -> None:
        if self._loaded:
            return
        if self.is_synced():
            exts = {".html", ".htm", ".md", ".txt"}
            for p in self.wiki_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    self._index.append((p, p.stem.lower().replace("_", " ")))
                    if len(self._index) >= self.max_files:
                        break
        self._loaded = True

    # -- retrieval -----------------------------------------------------------
    def retrieve(self, query: str, *, k: int = 4, snippet_chars: int = 1200) -> list[Snippet]:
        self._ensure_index()
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, Path]] = []
        for path, fname in self._index:
            fname_score = len(q_tokens & set(_tokens(fname))) * 3.0
            # cheap filename pre-filter; only read files with any hint
            if fname_score == 0 and not any(t in fname for t in q_tokens):
                continue
            scored.append((fname_score, path))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[Snippet] = []
        for base_score, path in scored[: k * 4]:
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = _strip_html(raw) if path.suffix.lower() in {".html", ".htm"} else raw
            body_tokens = _tokens(text)
            overlap = sum(body_tokens.count(t) for t in q_tokens)
            score = base_score + overlap
            if score <= 0:
                continue
            snippet = self._best_window(text, q_tokens, snippet_chars)
            results.append(
                Snippet(source=f"wiki:{path.name}", title=path.stem, text=snippet, score=score)
            )
        results.sort(key=lambda s: s.score, reverse=True)
        results = results[:k]

        # web fallback when offline wiki yields nothing useful
        if not results and self.web_search is not None:
            for hit in self.web_search(query, k):
                results.append(
                    Snippet(
                        source=f"web:{hit.get('url','')}",
                        title=hit.get("title", ""),
                        text=hit.get("snippet", ""),
                        score=1.0,
                    )
                )
        return results

    @staticmethod
    def _best_window(text: str, q_tokens: set[str], width: int) -> str:
        low = text.lower()
        best_pos, best_hits = 0, -1
        step = max(width // 2, 200)
        for start in range(0, max(len(text) - width, 1), step):
            window = low[start : start + width]
            hits = sum(window.count(t) for t in q_tokens)
            if hits > best_hits:
                best_hits, best_pos = hits, start
        return text[best_pos : best_pos + width].strip()

    # -- QA ------------------------------------------------------------------
    def answer(self, question: str, *, k: int = 4) -> dict[str, Any]:
        """Grounded QA. Returns {answer, sources, snippets}."""
        snippets = self.retrieve(question, k=k)
        sources = [s.source for s in snippets]
        if not snippets:
            return {"answer": "", "sources": [], "snippets": []}

        context = "\n\n".join(f"[{i+1}] ({s.source}) {s.text}" for i, s in enumerate(snippets))
        if self.llm is None or not getattr(self.llm, "available", False):
            # no LLM: return the raw context as the "answer"
            return {"answer": context, "sources": sources, "snippets": [s.text for s in snippets]}

        system = (
            "You are a Stardew Valley game-knowledge assistant. Answer ONLY from "
            "the provided context. Be concise and factual. If the context does not "
            "contain the answer, say so."
        )
        user = f"Question: {question}\n\nContext:\n{context}"
        try:
            ans = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception as e:
            ans = f"(LLM error, returning raw context) {e}\n\n{context}"
        return {"answer": ans, "sources": sources, "snippets": [s.text for s in snippets]}
