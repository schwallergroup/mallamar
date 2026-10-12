# Retrieve literature context from online paper services.

import json, urllib.parse, urllib.request, logging
import xml.etree.ElementTree as ET
from agent_lg.services.validation import integer_input

logger = logging.getLogger(__name__)


_RETRIEVAL_QUERIES = [
    ("BZ Reaction Chemistry",
     "Belousov-Zhabotinsky reaction oscillation conditions chemical parameters period signal"),
    ("Multi-Objective Bayesian Optimization",
     "multi-objective Bayesian optimization expected hypervolume improvement acquisition function"),
    ("LLM for Chemistry BO",
     "language model molecular representation Bayesian optimization chemistry surrogate fine-tuning"),
]


def _tool_search_literature(query: str, max_results: int = 5):
    """Search arXiv for papers matching the query."""
    try:
        integer_input(max_results, 'max_results')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('[literature] Supply a nonempty query')
    except ValueError as error:
        return {'error': str(error)}
    q = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{q}&max_results={max_results}&sortBy=relevance")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            xml = r.read().decode()
        feed = ET.fromstring(xml)
    except Exception as e:
        return {"error": str(e)}
    papers = []
    for entry in feed.findall('{*}entry')[:max_results]:
        papers.append({
            'title': entry.findtext('{*}title', default='').strip().replace('\n', ' '),
            'url': entry.findtext('{*}id', default='').strip(),
            'summary': entry.findtext('{*}summary', default='').strip().replace('\n', ' ')[:2000],
            "source": "arxiv",
        })
    return {"query": query, "n_results": len(papers), "papers": papers}


def _tool_search_chemrxiv(query: str, max_results: int = 5):
    """Search ChemRxiv via the Cambridge Open Engage public API."""
    try:
        integer_input(max_results, 'max_results')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('[literature] Supply a nonempty query')
    except ValueError as error:
        return {'error': str(error)}
    q = urllib.parse.quote(query)
    url = (f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
           f"?term={q}&limit={max_results}&sort=relevant")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if not isinstance(data, dict) or not isinstance(data.get('itemHits', []), list):
            raise ValueError('[literature] ChemRxiv returned an invalid response')
        for hit in data.get('itemHits', [])[:max_results]:
            if not isinstance(hit, dict) or not isinstance(hit.get('item', {}), dict):
                raise ValueError('[literature] ChemRxiv returned an invalid paper')
            item = hit.get('item', {})
            if any(item.get(key) is not None and not isinstance(item[key], str) for key in ('title', 'abstract', 'doi')):
                raise ValueError('[literature] ChemRxiv paper fields must be text')
    except Exception as e:
        return {"error": str(e)}
    papers = []
    for hit in data.get("itemHits", [])[:max_results]:
        item = hit.get("item", {})
        doi = item.get("doi", "")
        papers.append({
            "title": (item.get("title") or "").replace("\n", " "),
            "url": f"https://doi.org/{doi}" if doi else "",
            "summary": (item.get("abstract") or "")[:2000].replace("\n", " "),
            "source": "chemrxiv",
        })
    return {"query": query, "n_results": len(papers), "papers": papers}

def _fetch_literature_context(n_per_query: int = 3):
    """Fetch paper summaries from arXiv + ChemRxiv for each retrieval topic."""
    sections = []

    # arXiv + ChemRxiv
    for topic, query in _RETRIEVAL_QUERIES:
        lines = [f"### {topic}"]
        for search_fn in (_tool_search_literature, _tool_search_chemrxiv):
            result = search_fn(query, max_results=n_per_query)
            if "error" in result:
                lines.append(f"({search_fn.__name__} failed: {result['error']})")
                continue
            for p in result.get("papers", []):
                src = p.get("source", "")
                lines.append(f"- [{src}] **{p['title']}**\n  {p['summary'][:2000]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
