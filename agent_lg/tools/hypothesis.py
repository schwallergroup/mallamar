# Ground campaign hypotheses in local papers and observed experiments.

from __future__ import annotations
import hashlib, json, os, logging, pandas as pd, fitz
from pathlib import Path
from agent_lg.config import CONC_COLS

from agent_lg.services.profiling import profile_step, profiled

from agent_lg.services.validation import CampaignSafetyError
from agent_lg.services.literature import LocalCorpus
from agent_lg.services.storage import read_csv, file_revision, FileLock, atomic_write

logger = logging.getLogger(__name__)

# Limit extracted paper text to fit inside the model context window.
_PER_PAPER = 4000
_TOTAL = 16000

# Rows of observed data shown back to the agent each iteration.
_MAX_OBS_ROWS = 60

_SHORT = {"conc_MA_M": "MA", "conc_NaBrO3_M": "NaBrO3",
          "conc_H2SO4_M": "H2SO4", "conc_Ru_M": "Ru"}


# Literature corpus

_corpus_cache: str | None = None
_corpus_cache_key = None


@profiled("literature_load")
def load_corpus(dataset: str = "bz"):
    """Read the configured local papers, preserving the existing text limits."""
    global _corpus_cache, _corpus_cache_key
    corpus = LocalCorpus()
    papers = corpus.files()
    cache_key = (dataset, str(corpus.path),
                 tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in papers))
    if _corpus_cache is not None and _corpus_cache_key == cache_key:
        return _corpus_cache
    if corpus.path is not None and not papers:
        raise CampaignSafetyError(f"[literature] No pdf/txt/md papers in {corpus.path}; restore the configured collection")
    chunks = []
    for paper in papers:
        try:
            if paper.suffix == ".pdf":
                with fitz.open(paper) as document:
                    text = "\n".join(page.get_text() for page in document)
            else:
                text = paper.read_text(errors="ignore")
            if not text.strip():
                raise ValueError("Paper contains no extractable text")
            chunks.append(f"### PAPER: {paper.stem}\n{text.strip()[:_PER_PAPER]}")
        except (OSError, ValueError, RuntimeError) as error:
            raise CampaignSafetyError(f"[literature] Could not extract {paper.name}: {error}") from error
    _corpus_cache = "\n\n".join(chunks)[:_TOTAL] or "(no literature available — use your own chemistry knowledge)"
    _corpus_cache_key = cache_key
    return _corpus_cache


# Observed data table (the grounding that the ablation showed is load-bearing)

def observed_table(db_path: str, max_rows: int = _MAX_OBS_ROWS):
    """Compact (concentrations -> reward) table of what the robot has measured."""
    try:
        df = read_csv(db_path)
    except Exception as e:
        return f"(observed table unavailable: {e})"

    rcols = [c for c in ("reward_oscillation", "reward_signal") if c in df.columns]
    ccols = [c for c in CONC_COLS if c in df.columns]
    if not rcols or not ccols:
        return "(observed table unavailable: missing reward or concentration columns)"

    obs = df.dropna(subset=rcols)
    if obs.empty:
        return "(nothing measured yet)"
    obs = obs.tail(max_rows)

    head = "  ".join(_SHORT.get(c, c) for c in ccols) + "  | r_osc  r_sig"
    lines = [head]
    for _, r in obs.iterrows():
        cpart = " ".join(f"{float(r[c]):.3f}" for c in ccols)
        rpart = " ".join(f"{float(r[c]):+.2f}" for c in rcols)
        lines.append(f"{cpart} | {rpart}")
    n_hidden = len(df.dropna(subset=rcols)) - len(obs)
    if n_hidden > 0:
        lines.append(f"({n_hidden} earlier measured rows not shown)")
    return "\n".join(lines)


# Hypothesis persistence

class HypothesisStore:
    'File-backed current hypothesis + its revision history.'

    def __init__(self, path: str | Path | None = None):
        self.path = (Path(path) if path else self._default_path()).expanduser().resolve()
        self.text = ""
        self.history = []
        self._load()

    @staticmethod
    def _default_path():
        explicit = os.getenv("OCEAN_HYPOTHESIS_PATH")
        if explicit:
            return Path(explicit)
        db = os.getenv("OCEAN_CAMPAIGN_DB")
        if not db:
            raise ValueError("[HypothesisStore] Provide OCEAN_CAMPAIGN_DB or an explicit hypothesis path")
        pool = Path(db).expanduser().resolve()
        if os.getenv("MALLAMAR_ARTIFACTS_RECORD"):
            return pool.with_name("hypothesis.json") # keep the default recorded by older campaigns
        identifier = hashlib.sha256(str(pool).encode()).hexdigest()
        return pool.parent / ".mallamar" / "hypotheses" / (identifier + ".json")

    def _load(self):
        self.revision = file_revision(self.path)
        if self.revision is None:
            return
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(saved, dict) or not isinstance(saved.get("hypothesis", ""), str)
                or not isinstance(saved.get("history", []), list)
                or not all(isinstance(item, dict) and isinstance(item.get("hypothesis"), str)
                           for item in saved.get("history", []))):
            raise ValueError("[HypothesisStore] Invalid hypothesis history; original file preserved")
        self.text, self.history = saved.get("hypothesis", ""), saved.get("history", [])
        if file_revision(self.path) != self.revision:
            raise RuntimeError("[HypothesisStore] Hypothesis changed during loading; retry after its writer finishes")

    def _save(self, text, history):
        try:
            with FileLock(self.path):
                atomic_write(self.path, lambda temporary: temporary.write_text(
                    json.dumps({"hypothesis": text, "history": history}, indent=2), encoding="utf-8"),
                    expected_revision=self.revision, check_revision=True)
                self.revision = file_revision(self.path)
        except (OSError, ValueError, RuntimeError) as error:
            raise CampaignSafetyError(f"[hypothesis] Revision could not be saved: {error}") from error

    @staticmethod
    def _norm(text: str):
        """Whitespace-insensitive form, for deciding whether a revision changed anything."""
        return " ".join((text or "").split())

    def update(self, text: str, bo_iter: int | None = None):
        """Record a new/revised hypothesis. Returns a small status dict."""
        text = (text or "").strip()
        if not text:
            return {"stored": False, "reason": "empty hypothesis"}
        # Static mode keeps the first hypothesis; normal campaigns can revise it.
        if os.getenv("OCEAN_HYP_STATIC") and self.text:
            return {"stored": False, "reason": "static mode — hypothesis frozen"}

        # Identical wording does not add another revision.
        if self.text and self._norm(text) == self._norm(self.text):
            return {"stored": False,
                    "reason": "identical to the current hypothesis — nothing was revised. "
                              "Either revise it against the new observations (cite the new "
                              "points) or skip update_hypothesis entirely this iteration.",
                    "n_revisions": len(self.history)}

        # Attribute revisions to the current graph iteration.
        env_iter = os.getenv("OCEAN_BO_ITER")
        if env_iter is not None:
            try:
                bo_iter = int(env_iter)
            except ValueError:
                pass

        history = self.history + [{"bo_iter": bo_iter, "hypothesis": text}]
        self._save(text, history) # only report success after the revision is durable
        self.history, self.text = history, text
        return {"stored": True, "n_revisions": len(self.history),
                "hypothesis_len": len(text), "bo_iter": bo_iter}


_store: HypothesisStore | None = None


def store():
    """Cache the active path; re-read when another campaign or restored file is selected."""
    global _store
    try:
        path = HypothesisStore._default_path().expanduser().resolve()
        if _store is None or _store.path != path or _store.revision != file_revision(path):
            _store = HypothesisStore(path)
    except (OSError, ValueError, RuntimeError) as error:
        raise CampaignSafetyError(f"[hypothesis] Cannot load the campaign history: {error}") from error
    return _store


def _tool_update_hypothesis(hypothesis: str, bo_iter: int = None):
    """Persist the agent's (revised) working hypothesis."""
    res = store().update(hypothesis, bo_iter)
    if res.get("stored"):
        print(f"[hypothesis] revision {res['n_revisions']} stored "
              f"({res['hypothesis_len']} chars, bo_iter={res.get('bo_iter')}) "
              f"-> {store().path}")
    else:
        print(f"[hypothesis] NOT stored: {res.get('reason')}")
    return res
