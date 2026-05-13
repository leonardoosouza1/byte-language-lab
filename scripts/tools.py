#!/usr/bin/env python3
"""Tool router — brain spawns this subprocess for non-trivial fetches.

Subcommands:
  url <URL>           → fetch web page, return plain text (html2text)
  pdf <PATH or URL>   → extract text from PDF
  wiki <TOPIC>        → Wikipedia summary PT-BR (fallback EN)
  search <QUERY>      → DuckDuckGo HTML search top-3 results
  arxiv <QUERY>       → ArXiv paper abstracts
  github <REPO>       → GitHub README + meta

Output: plain text to stdout, errors to stderr, exit 0 success, 1 failure.
"""
import sys, urllib.request, urllib.parse, json, re, subprocess
from pathlib import Path

UA = "IARA-Byte-Brain/1.0 (research bot)"
TIMEOUT = 15

def html_to_text(html: str) -> str:
    """Quick HTML strip — no deps. Removes scripts/styles/tags."""
    # Drop script/style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block tags with newlines
    html = re.sub(r"<(?:p|div|br|h[1-6]|li)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode entities
    import html as html_mod
    html = html_mod.unescape(html)
    # Collapse whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n\n", html)
    return html.strip()

def cmd_url(target: str) -> str:
    """Fetch URL, return plain text."""
    req = urllib.request.Request(target, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read()
        if "pdf" in ctype.lower() or target.lower().endswith(".pdf"):
            return _extract_pdf_bytes(data)
        text = data.decode("utf-8", errors="replace")
        return html_to_text(text)[:8000]

def _extract_pdf_bytes(data: bytes) -> str:
    """Try multiple PDF backends."""
    for backend in [_pdf_pypdf, _pdf_pdftotext]:
        try:
            t = backend(data)
            if t and t.strip():
                return t[:8000]
        except Exception:
            continue
    return ""

def _pdf_pypdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return ""
    import io
    reader = PdfReader(io.BytesIO(data))
    text = []
    for page in reader.pages[:20]:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def _pdf_pdftotext(data: bytes) -> str:
    """Shell out to pdftotext (poppler-utils)."""
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data); path = f.name
    try:
        r = subprocess.run(["pdftotext", path, "-"], capture_output=True, timeout=30)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        pass
    finally:
        Path(path).unlink(missing_ok=True)
    return ""

def cmd_pdf(target: str) -> str:
    """Extract text from PDF (local path or URL)."""
    if target.startswith(("http://", "https://")):
        return cmd_url(target)
    p = Path(target)
    if not p.exists():
        raise FileNotFoundError(target)
    return _extract_pdf_bytes(p.read_bytes())

def cmd_wiki(topic: str, lang: str = "pt") -> str:
    """Wikipedia summary via REST API."""
    title = topic.strip()
    # Capitalize first letter to match Wikipedia convention
    title = title[0].upper() + title[1:] if title else title
    quoted = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quoted}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read())
            extract = d.get("extract", "")
            if extract:
                return extract[:2000]
    except Exception:
        pass
    # Fallback to EN if PT failed
    if lang != "en":
        return cmd_wiki(topic, "en")
    return ""

def cmd_search(query: str) -> str:
    """DuckDuckGo HTML search — top-3 results (title + snippet)."""
    q = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            html = r.read().decode("utf-8", errors="replace")
        # Parse top 3 results — DDG returns class "result"
        results = []
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]*>([^<]+)</a>(?:.*?<a[^>]*class="result__snippet"[^>]*>([^<]+))?',
                              html, flags=re.DOTALL):
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            snippet = re.sub(r"<[^>]+>", "", m.group(2) or "").strip()
            if title:
                results.append(f"• {title}\n  {snippet[:200]}")
            if len(results) >= 3: break
        return "\n\n".join(results) if results else ""
    except Exception:
        return ""

def cmd_arxiv(query: str) -> str:
    """ArXiv search — top-3 abstracts."""
    q = urllib.parse.quote_plus(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            xml = r.read().decode("utf-8")
        # Parse <entry><title> + <summary>
        results = []
        for m in re.finditer(r"<entry>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>",
                              xml, flags=re.DOTALL):
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            s = re.sub(r"\s+", " ", m.group(2)).strip()[:300]
            results.append(f"📄 {t}\n   {s}")
        return "\n\n".join(results) if results else ""
    except Exception:
        return ""

def cmd_github(repo: str) -> str:
    """Fetch GitHub repo README + meta (owner/repo or owner/repo)."""
    if "/" not in repo: return ""
    url = f"https://api.github.com/repos/{repo}/readme"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read().decode("utf-8", errors="replace")[:3000]
    except Exception:
        return ""

def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: tools.py <cmd> <arg>\n")
        sys.exit(1)
    cmd = sys.argv[1]
    arg = " ".join(sys.argv[2:])
    try:
        if cmd == "url":      out = cmd_url(arg)
        elif cmd == "pdf":    out = cmd_pdf(arg)
        elif cmd == "wiki":   out = cmd_wiki(arg)
        elif cmd == "search": out = cmd_search(arg)
        elif cmd == "arxiv":  out = cmd_arxiv(arg)
        elif cmd == "github": out = cmd_github(arg)
        else:
            sys.stderr.write(f"unknown cmd: {cmd}\n"); sys.exit(1)
        if not out.strip():
            sys.stderr.write("empty result\n"); sys.exit(2)
        sys.stdout.write(out)
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(3)

if __name__ == "__main__":
    main()
