"""FastAPI server for crawling articles using newspaper4k."""

from __future__ import annotations

import hashlib
import html as html_lib
import logging
import mimetypes
import os
import pathlib
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from newspaper import Article
from newspaper.configuration import Configuration
import newspaper

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
}
_REQUEST_TIMEOUT = 10

# JS cookie-challenge pattern used by sites like laodong.vn (Cloudrity protection).
# Example: document.cookie="D1N=abc123"+"; expires=...";window.location.reload(true);
_JS_COOKIE_CHALLENGE_RE = re.compile(
    r'document\.cookie\s*=\s*"([^=]+)=([a-f0-9A-F]+)"',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Media download / localisation
# ---------------------------------------------------------------------------

# Directory on disk where downloaded media is stored.
# Set MEDIA_STORAGE_PATH env var to override (e.g. /home/tss/scraper/medias).
_MEDIA_DIR = pathlib.Path(os.environ.get("MEDIA_STORAGE_PATH", "./medias")).resolve()

# Base CDN URL returned in responses after a file is downloaded.
# Set CDN_MEDIA_BASE_URL env var to override (e.g. https://sapbao.net/cdn-medias).
_CDN_BASE_URL = os.environ.get("CDN_MEDIA_BASE_URL", "https://sapbao.local/cdn-medias").rstrip("/")

# Maximum file size (bytes) to download for a single media asset (default 50 MB).
_MAX_MEDIA_SIZE = int(os.environ.get("MAX_MEDIA_SIZE_BYTES", str(50 * 1024 * 1024)))

# Extension overrides for common MIME types not covered by mimetypes stdlib.
_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg":   ".jpg",
    "image/png":    ".png",
    "image/gif":    ".gif",
    "image/webp":   ".webp",
    "image/avif":   ".avif",
    "image/svg+xml": ".svg",
    "video/mp4":    ".mp4",
    "video/webm":   ".webm",
    "video/ogg":    ".ogv",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "application/x-mpegURL": ".m3u8",
    "application/vnd.apple.mpegurl": ".m3u8",
}

# Image extensions that can be processed by imgproxy.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".svg"}
# Video extensions served directly without imgproxy.
_VIDEO_EXTS = {".mp4", ".webm", ".ogv", ".mov", ".avi", ".m3u8"}

# Regex to find markdown image references: ![alt](url)
_MD_IMAGE_RE = re.compile(r'(!\[[^\]]*\])\(([^)\s]+)\)')
_HTML_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Shared HTTP connection pool — reused across all requests and threads.
# _media_session is for media-only downloads (GET, no cookie mutation).
# ---------------------------------------------------------------------------
_http_adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=50,
    max_retries=Retry(total=0),  # explicit retry logic lives in callers
)
_media_session = requests.Session()
_media_session.headers.update(_HEADERS)
_media_session.mount("http://", _http_adapter)
_media_session.mount("https://", _http_adapter)

# ---------------------------------------------------------------------------
# Article-URL heuristics — compiled once at module load, not per request.
# ---------------------------------------------------------------------------
_SRC_NON_ARTICLE = re.compile(
    r"/(tag|tags|tu-khoa|label|labels"
    r"|topic|topics|chu-de|chuyen-de"
    r"|category|cat|danh-muc|chuyen-muc"
    r"|author|authors|tac-gia|user|profile"
    r"|search|tim-kiem|keyword"
    r"|page|trang"
    r"|epaper|e-paper|bao-in|archive|luu-tru"
    r"|gallery|video|photo|anh|infographic"
    r"|rss|feed|amp)"
    r"(/|$|\?|#)",
    re.IGNORECASE,
)
_SRC_ASSET_EXT = re.compile(
    r"\.(css|js|png|jpg|jpeg|gif|webp|svg|ico|pdf|zip|tar|gz|xml|json|rss|atom)(\?|$)",
    re.IGNORECASE,
)
_SRC_ARTICLE_ID = re.compile(r"\d{4,}")
_SRC_ARTICLE_SIGNAL = re.compile(
    r"(post\d+|\d{6,}\.html?|\d{6,}\.tpo|\d{5,}\.ldo|\d{4}/\d{2}/\d{2})",
    re.IGNORECASE,
)
_SRC_HEX_SUFFIX = re.compile(r"-[0-9a-f]{8,}(\.html?|\.tpo)?$", re.IGNORECASE)
_SRC_FILE_EXT = re.compile(r"\.(html?|tpo|aspx|ldo)$", re.IGNORECASE)


def _registered_domain(netloc: str) -> str:
    """Return the last two dot-separated labels (handles .co.uk etc. roughly)."""
    host = netloc.lower().lstrip("www.")
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _img_resolved_url(img_tag, base_url: str) -> str | None:
    """Best-effort image URL from an <img> tag (lazy-load attrs) made absolute."""
    def _pick_from_srcset(srcset_value: str) -> str | None:
        # srcset format: "url1 320w, url2 640w" → prefer the first valid URL token.
        for part in srcset_value.split(","):
            token = (part or "").strip().split(" ")[0].strip()
            if token and not token.lower().startswith("data:"):
                return token
        return None

    attrs_order = (
        "data-src",
        "data-original",
        "data-lazy-src",
        "data-url",
        "data-srcset",
        "srcset",
        "src",
    )
    raw: str | None = None
    for key in attrs_order:
        val = (img_tag.get(key) or "").strip()
        if not val or val.lower().startswith("data:"):
            continue
        if key in ("data-srcset", "srcset"):
            picked = _pick_from_srcset(val)
            if not picked:
                continue
            raw = picked
            break
        raw = val
        break
    # Fallback for <picture><source srcset>...<img ...></picture> patterns.
    if not raw:
        picture = img_tag.find_parent("picture")
        if picture:
            for source in picture.find_all("source"):
                srcset = (source.get("srcset") or source.get("data-srcset") or "").strip()
                if not srcset:
                    continue
                picked = _pick_from_srcset(srcset)
                if picked:
                    raw = picked
                    break
    if not raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not raw.startswith(("http://", "https://")):
        raw = urljoin(base_url, raw)
    return raw


def _guess_ext(url: str, content_type: str | None) -> str:
    """Return the best file extension for *url* based on URL path and content-type."""
    path = urlparse(url).path.split("?")[0]
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS | _VIDEO_EXTS:
        return ext
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in _MIME_TO_EXT:
            return _MIME_TO_EXT[ct]
        guessed = mimetypes.guess_extension(ct)
        if guessed:
            return guessed
    return ".bin"


def _download_media(media_url: str, ref_date: datetime | None = None) -> str | None:
    """Download *media_url* to local storage and return its CDN URL.

    Files are stored under ``_MEDIA_DIR/{year-month}/{day}/`` and
    deduplicated by a SHA-256 hash of the URL.  Returns ``None`` on error.

    *ref_date* is used for the storage sub-directory; when omitted the
    current date is used as a fallback.
    """
    try:
        url_hash = hashlib.sha256(media_url.encode()).hexdigest()[:20]
        now = ref_date or datetime.now()
        subdir = now.strftime("%Y-%m") + "/" + now.strftime("%d")
        dest_dir = _MEDIA_DIR / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Probe the file first; reuse the shared session for TCP connection pooling.
        resp = _media_session.get(
            media_url,
            timeout=_REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        content_length = int(resp.headers.get("content-length", 0) or 0)
        if content_length and content_length > _MAX_MEDIA_SIZE:
            log.warning(
                "Skipping large media (%d bytes > %d): %s",
                content_length, _MAX_MEDIA_SIZE, media_url,
            )
            return None

        ext = _guess_ext(media_url, content_type)
        filename = f"{url_hash}{ext}"
        filepath = dest_dir / filename
        relative = f"{subdir}/{filename}"

        # Skip download if already cached on disk.
        if not filepath.exists():
            downloaded = 0
            with open(filepath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > _MAX_MEDIA_SIZE:
                            fh.close()
                            filepath.unlink(missing_ok=True)
                            log.warning("Aborted oversized download: %s", media_url)
                            return None
                        fh.write(chunk)
            log.info("Downloaded media → %s", relative)

        return f"{_CDN_BASE_URL}/{relative}"

    except Exception as exc:
        log.warning("Failed to download media %s: %s", media_url, exc)
        return None


def _rewrite_markdown_images(
    text_markdown: str,
    ref_date: datetime | None = None,
    base_url: str = "",
    url_cache: dict | None = None,
) -> str:
    """Replace external image URLs in markdown ``![alt](url)`` with CDN URLs."""
    def _abs_url(u: str) -> str:
        u = u.strip()
        if u.startswith("//"):
            return "https:" + u
        if u.startswith(("http://", "https://")):
            return u
        if base_url:
            return urljoin(base_url, u)
        return u

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)          # e.g. "![alt text]"
        img_url = _abs_url(m.group(2))
        if not img_url.startswith(("http://", "https://")):
            return m.group(0)
        local_url = url_cache.get(img_url) if url_cache is not None else _download_media(img_url, ref_date)
        if local_url:
            return f"{prefix}({local_url})"
        return m.group(0)

    return _MD_IMAGE_RE.sub(_replace, text_markdown)


def _rewrite_html_image_sources(
    text_markdown: str,
    ref_date: datetime | None = None,
    base_url: str = "",
    url_cache: dict | None = None,
) -> str:
    """Replace external image URLs in raw HTML <img src="..."> blocks with CDN URLs."""
    def _abs_url(u: str) -> str:
        u = u.strip()
        if u.startswith("//"):
            return "https:" + u
        if u.startswith(("http://", "https://")):
            return u
        if base_url:
            return urljoin(base_url, u)
        return u

    def _replace(m: re.Match) -> str:
        prefix, img_url, quote = m.group(1), m.group(2), m.group(3)
        abs_url = _abs_url(img_url)
        if not abs_url.startswith(("http://", "https://")):
            return m.group(0)
        local_url = url_cache.get(abs_url) if url_cache is not None else _download_media(abs_url, ref_date)
        if local_url:
            return f"{prefix}{local_url}{quote}"
        return m.group(0)

    return _HTML_IMG_SRC_RE.sub(_replace, text_markdown)


def _fetch_html(url: str) -> tuple[str, str]:
    """Download HTML via requests with browser UA.

    Handles:
    - SSL errors: retries with ``verify=False``
    - JS cookie challenges (Cloudrity / laodong.vn-style): extracts cookie and retries

    Returns (html_text, final_url_after_redirects).
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    # Mount the shared adapter for TCP reuse; cookie state stays isolated per request.
    session.mount("http://", _http_adapter)
    session.mount("https://", _http_adapter)

    def _do_get(target: str, **kwargs) -> requests.Response:
        try:
            r = session.get(target, timeout=_REQUEST_TIMEOUT, allow_redirects=True, **kwargs)
        except requests.exceptions.SSLError:
            log.warning("SSL error for %s — retrying with verify=False", target)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=urllib3.exceptions.InsecureRequestWarning
                )
                r = session.get(
                    target, timeout=_REQUEST_TIMEOUT, allow_redirects=True, verify=False, **kwargs
                )
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r

    resp = _do_get(url)

    # Detect and solve JS cookie challenge (e.g. Cloudrity on laodong.vn)
    if 'window.location.reload' in resp.text:
        m = _JS_COOKIE_CHALLENGE_RE.search(resp.text)
        if m:
            key, val = m.group(1), m.group(2)
            log.info("JS cookie challenge for %s — setting %s=%s and retrying", url, key, val)
            session.cookies.set(key, val)
            resp = _do_get(str(resp.url))

    return resp.text, str(resp.url)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Article Crawler API",
    description="Crawl and extract article content from any URL using newspaper4k",
    version="1.0.0",
)


class ArticleResponse(BaseModel):
    url: str
    title: str
    authors: list[str]
    publish_date: Optional[str]
    text: str
    text_markdown: str
    top_image: Optional[str]
    images: list[str]
    movies: list[str]
    meta_keywords: list[str]
    tags: list[str]
    meta_description: Optional[str]
    meta_lang: Optional[str]
    source_url: Optional[str]
    article_type: str


class CrawlRequest(BaseModel):
    url: HttpUrl
    language: Optional[str] = None
    follow_meta_refresh: bool = False
    keep_article_html: bool = False
    download_media: bool = False


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/crawl", response_model=ArticleResponse, summary="Crawl article from URL")
def crawl_get(
    url: str = Query(..., description="URL of the article to crawl"),
    language: Optional[str] = Query(None, description="Language code, e.g. 'en', 'vi', 'zh'"),
    follow_meta_refresh: bool = Query(False, description="Follow meta refresh redirects"),
    download_media: bool = Query(False, description="Download media to local CDN and rewrite URLs"),
):
    """Crawl and extract article content from the given URL (GET)."""
    return _crawl(url, language=language, follow_meta_refresh=follow_meta_refresh, download_media=download_media)


@app.post("/crawl", response_model=ArticleResponse, summary="Crawl article from URL")
def crawl_post(body: CrawlRequest):
    """Crawl and extract article content from the given URL (POST)."""
    return _crawl(
        str(body.url),
        language=body.language,
        follow_meta_refresh=body.follow_meta_refresh,
        keep_article_html=body.keep_article_html,
        download_media=body.download_media,
    )


def _extract_movies_from_html(html: str, _soup: BeautifulSoup | None = None) -> list[str]:
    """Extract video URLs from <video src> and <source type="video/*"> tags.

    Newspaper does not parse HTML5 <video> elements — this supplements it.
    """
    soup = _soup if _soup is not None else BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    # <video src="..."> direct
    for tag in soup.find_all("video", src=True):
        u = tag["src"].strip()
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    # <source src="..." type="video/..."> inside <video>
    for tag in soup.find_all("source"):
        mime = tag.get("type", "")
        src = (tag.get("src") or "").strip()
        if src and src not in seen and ("video/" in mime or src.split("?")[0].endswith((".mp4", ".webm", ".ogg", ".m3u8", ".mov"))):
            seen.add(src)
            urls.append(src)
    return urls


def _extract_text_from_html(html: str, base_url: str = "", _soup: BeautifulSoup | None = None) -> tuple[str, str]:
    """Extract (text, text_markdown) from article HTML using BS4.

    Targets the primary article body selectors used by laodong.vn:
    - .chappeau  → lead paragraph
    - article .article-body / .detail__content / similar  → body text

    Returns plain text and a simple markdown representation.
    Falls back to empty strings if nothing found.

    *base_url* is the article page URL; used to absolutize relative / protocol-relative
    image URLs so markdown and downstream media download work.
    """
    soup = _soup if _soup is not None else BeautifulSoup(html, "html.parser")

    parts: list[str] = []

    # 1. Lead / chappeau
    lead = soup.find(class_="chappeau") or soup.find(class_="article__sapo") or soup.find(class_="sapo")
    if lead:
        parts.append(lead.get_text(" ", strip=True))

    # 2. Article body (order: specific layouts first, then common CMS blocks)
    body = (
        soup.find(class_="article__body")
        or soup.find(class_="article-body")
        or soup.find(class_="detail__content")
        or soup.find(class_="article__content")
        or soup.find(class_="edittor-content")  # VTC News (site uses this spelling)
        or soup.find(class_="editor-content")
        or soup.find(class_="fck_detail")
        or soup.find(class_="entry-content")
        or soup.find(class_="post-content")
        or soup.find(class_="content-wrapper")  # VTC outer prose (fallback if no edittor)
    )
    if body:
        # Remove sidebar / related / footer elements inside body
        for noise in body.find_all(class_=re.compile(r"related|sidebar|footer|widget|ad|social|comment", re.I)):
            noise.decompose()
        body_text = body.get_text(" ", strip=True)
        if body_text:
            parts.append(body_text)

    text = "\n\n".join(parts)
    # Markdown: lead italic when present; body blocks + images whenever body exists
    # (must not nest body under `if lead` — many sites have no sapo but rich HTML body)
    md_parts: list[str] = []
    if lead:
        md_parts.append(f"_{lead.get_text(' ', strip=True)}_")

    if body:
        body_md_parts: list[str] = []

        # Walk common content blocks in DOM order and preserve inline images.
        for el in body.find_all(
            ["p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "figure", "img"],
            recursive=True,
        ):
            name = el.name.lower() if el.name else ""

            if name == "figure":
                img = el.find("img")
                if img:
                    src = _img_resolved_url(img, base_url)
                    if src:
                        alt = html_lib.escape((img.get("alt") or "").strip(), quote=True)
                        caption_tag = el.find("figcaption")
                        caption_text = caption_tag.get_text(" ", strip=True) if caption_tag else ""
                        caption_html = (
                            f"<figcaption>{html_lib.escape(caption_text)}</figcaption>"
                            if caption_text else ""
                        )
                        body_md_parts.append(
                            f'<figure class="image"><img src="{html_lib.escape(src, quote=True)}" alt="{alt}" />{caption_html}</figure>'
                        )
                continue

            if name == "img":
                # Standalone image only; images inside <figure> are handled above.
                if el.find_parent("figure"):
                    continue
                src = _img_resolved_url(el, base_url)
                if src:
                    alt = (el.get("alt") or "").strip()
                    body_md_parts.append(f"![{alt}]({src})")
                continue

            # Avoid duplicate nested text (e.g. <li> within another <li>)
            if el.find_parent(["p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "figure"]):
                continue

            text_line = el.get_text(" ", strip=True)
            if text_line:
                body_md_parts.append(text_line)

        if body_md_parts:
            md_parts.append("\n\n".join(body_md_parts))
        else:
            body_md = body.get_text(" ", strip=True)
            if body_md:
                md_parts.append(body_md)
    elif not md_parts:
        md_parts = parts[:]

    text_markdown = "\n\n".join(md_parts)
    return text, text_markdown


def _crawl(
    url: str,
    language: Optional[str] = None,
    follow_meta_refresh: bool = False,
    keep_article_html: bool = False,
    download_media: bool = False,
) -> ArticleResponse:
    try:
        config = Configuration()
        if language:
            config.language = language
        config.follow_meta_refresh = follow_meta_refresh
        config.keep_article_html = keep_article_html

        html, final_url = _fetch_html(url)
        article = Article(final_url, config=config)
        article.download(input_html=html)
        article.parse()
        # Parse HTML once and share the soup object to avoid redundant parsing.
        _soup = BeautifulSoup(html, "html.parser")

        publish_date = (
            article.publish_date.isoformat() if article.publish_date else None
        )

        top_image = article.top_image or None
        # znews.vn: w1250 images have a watermark at footer — use w960 instead
        if top_image and "photo.znews.vn/w1250/" in top_image:
            top_image = top_image.replace("/w1250/", "/w960/", 1)
        # vov.vn: og_image style is too small — use large_watermark for better quality
        if top_image and "media.vov.vn" in top_image and "/styles/og_image/" in top_image:
            top_image = top_image.replace("/styles/og_image/", "/styles/large_watermark/", 1)

        # movies: newspaper misses HTML5 <video>/<source> elements — supplement via BS4
        movies = article.movies or []
        if not movies:
            movies = _extract_movies_from_html(html, _soup=_soup)

        # text / text_markdown: for video pages (or whenever newspaper returns poor text),
        # fall back to BS4 extraction of .chappeau + article body.
        text = article.text or ""
        text_markdown = article.text_markdown or ""
        bs_text, bs_md = _extract_text_from_html(html, final_url, _soup=_soup)
        if article.article_type == "video" or not text.strip():
            if bs_text:
                text = bs_text
                text_markdown = bs_md
        elif download_media and "![" in bs_md and "](" in bs_md:
            # newspaper's text_markdown usually has no inline images; BS4 body walk emits ![alt](url).
            text_markdown = bs_md
        elif not text_markdown.strip() and bs_md.strip():
            # newspaper often omits text_markdown entirely while HTML body matches our selectors
            text_markdown = bs_md
        images = list(article.images) if article.images else []

        # ── Media localisation ──────────────────────────────────────────────
        if download_media:
            # Use publish_date as storage date so files land in the right folder.
            # Fall back to current date when publish_date is unavailable.
            media_date = article.publish_date or datetime.now()

            # Collect every unique HTTP(S) URL to download in a single parallel pass.
            _to_download: set[str] = set()
            if top_image and top_image.startswith(("http://", "https://")):
                _to_download.add(top_image)
            for _mv in movies:
                if _mv.startswith(("http://", "https://")):
                    _to_download.add(_mv)
            for _img in images:
                if _img.startswith(("http://", "https://")):
                    _to_download.add(_img)
            if text_markdown:
                def _abs_md(u: str) -> str:
                    u = u.strip()
                    if u.startswith("//"): return "https:" + u
                    if u.startswith(("http://", "https://")): return u
                    return urljoin(final_url, u) if final_url else u
                for _m in _MD_IMAGE_RE.finditer(text_markdown):
                    _u = _abs_md(_m.group(2))
                    if _u.startswith(("http://", "https://")): _to_download.add(_u)
                for _m in _HTML_IMG_SRC_RE.finditer(text_markdown):
                    _u = _abs_md(_m.group(2))
                    if _u.startswith(("http://", "https://")): _to_download.add(_u)

            # Download all assets concurrently — each is independent I/O.
            url_cache: dict[str, str | None] = {}
            if _to_download:
                _workers = min(len(_to_download), int(os.environ.get("MEDIA_WORKERS", "8")))
                with ThreadPoolExecutor(max_workers=_workers) as _ex:
                    _futs = {_ex.submit(_download_media, _u, media_date): _u for _u in _to_download}
                    for _f in as_completed(_futs):
                        url_cache[_futs[_f]] = _f.result()

            # 1. top_image
            if top_image and url_cache.get(top_image):
                top_image = url_cache[top_image]

            # 2. movies (videos)
            movies = [
                (url_cache.get(mv) or mv) if mv.startswith(("http://", "https://")) else mv
                for mv in movies
            ]

            # 3. inline images in text_markdown  (![alt](url) patterns)
            if text_markdown:
                text_markdown = _rewrite_markdown_images(
                    text_markdown, media_date, base_url=final_url, url_cache=url_cache
                )
                text_markdown = _rewrite_html_image_sources(
                    text_markdown, media_date, base_url=final_url, url_cache=url_cache
                )

            # 4. images list (all images referenced in the article)
            images = [
                (url_cache.get(img) or img) if img.startswith(("http://", "https://")) else img
                for img in images
            ]

        return ArticleResponse(
            url=article.url,
            title=article.title or "",
            authors=article.authors or [],
            publish_date=publish_date,
            text=text,
            text_markdown=text_markdown,
            top_image=top_image,
            images=images,
            movies=movies,
            meta_keywords=article.meta_keywords or [],
            tags=list(article.tags) if article.tags else [],
            meta_description=article.meta_description or None,
            meta_lang=article.meta_lang or None,
            source_url=article.source_url or None,
            article_type=article.article_type,
        )
    except Exception as exc:
        log.exception("Failed to crawl %s", url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SourceResponse(BaseModel):
    url: str
    domain: str
    article_urls: list[str]
    total: int


@app.get("/source", response_model=SourceResponse, summary="Discover article URLs from a homepage")
def scrape_source(
    url: str = Query(..., description="Homepage or source URL to discover articles from"),
    language: Optional[str] = Query(None, description="Language code, e.g. 'vi', 'en'"),
    only_in_path: bool = Query(False, description="Only include article URLs within the same URL path"),
):
    """Discover article URLs from a news source homepage.

    Uses ``newspaper.build()`` to fetch the page (handles UA, redirects, sessions),
    then combines newspaper's own article detection with HTML ``<a>`` extraction on
    ``source.html``.  Both sources are run through a strict article-URL heuristic
    that keeps only genuine article links and discards category / tag / topic /
    author / pagination pages.

    Article-URL signals (all must pass):
    - Same registered domain as the source URL
    - Path contains ≥4 consecutive digits (date segment or numeric ID)
    - Path depth ≥ 2 levels (not a top-level category slug)
    - NOT matched by known non-article path segments (tag, topic, danh-muc, …)
    - NOT a static-asset extension
    """
    # ── Heuristic helpers (regexes and _registered_domain are module-level) ──

    def is_article_url(candidate: str, base_reg_domain: str) -> bool:
        try:
            p = urlparse(candidate)
        except Exception:
            return False
        if p.scheme not in ("http", "https"):
            return False
        if _registered_domain(p.netloc) != base_reg_domain:
            return False
        path = p.path.rstrip("/")
        if not path or path == "/":
            return False
        # Must be at least 2 path levels deep (e.g. /rubric/article, not /the-thao)
        if path.count("/") < 1 or len(path) < 10:
            return False
        if _SRC_ASSET_EXT.search(path):
            return False
        if _SRC_NON_ARTICLE.search(path):
            return False
        # ── Tier 1: strong explicit article signals (any depth) ──────────
        # post12345, 6-digit id.html, date path, hex suffix like slug-a1b2c3d4.html
        if _SRC_ARTICLE_SIGNAL.search(path) or _SRC_HEX_SUFFIX.search(path):
            return True
        # ── Tier 2: file-extension URL with long slug (depth ≥ 1) ────────
        # Covers sites like baophapluat.vn where articles live at /slug.html
        # (no numeric ID). Require path ≥ 30 chars to exclude short section pages
        # like /the-thao.html or /van-hoa.html.
        if _SRC_FILE_EXT.search(path) and len(path) >= 30:
            return True
        # ── Tier 3: depth ≥ 2 with 4+ digit ID ──────────────────────────
        if path.count("/") >= 2 and _SRC_ARTICLE_ID.search(path):
            return True
        return False

    # ── Build source via newspaper (handles HTTP, UA, redirects) ──────────
    try:
        config = Configuration()
        config.browser_user_agent = _BROWSER_UA
        config.headers = _HEADERS
        if language:
            config.language = language
        config.memoize_articles = False  # avoid stale cache across calls

        source = newspaper.build(
            url,
            config=config,
            only_homepage=True,
            only_in_path=only_in_path,
        )

        parsed_base = urlparse(url)
        base_reg = _registered_domain(parsed_base.netloc)

        seen: set[str] = set()
        article_urls: list[str] = []

        def _add(u: str) -> None:
            clean = u.split("#")[0].rstrip("/")
            if clean and clean not in seen and is_article_url(clean, base_reg):
                seen.add(clean)
                article_urls.append(clean)

        # 1. newspaper's own article detection (works for standard URL patterns)
        for a in source.articles:
            if a.url:
                _add(a.url)

        # 2. Parse <a> tags from the HTML already downloaded by newspaper
        #    (covers sites with non-standard extensions like .tpo, .htm, .ldo etc.)
        #    If newspaper returned minimal HTML (e.g. JS cookie challenge), fall back
        #    to _fetch_html which handles those challenges.
        html = getattr(source, "html", "") or ""
        if len(html) < 500:
            log.info("newspaper returned minimal HTML (%d bytes) for %s — using _fetch_html fallback", len(html), url)
            try:
                html, _ = _fetch_html(url)
            except Exception as e:
                log.warning("_fetch_html fallback failed for %s: %s", url, e)
                html = ""
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all("a", href=True):
                href = (tag.get("href") or "").strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                _add(urljoin(url, href))

        log.info(
            "scrape_source %s → %d article URLs (newspaper native=%d, html_parse=%d)",
            url,
            len(article_urls),
            len(source.articles),
            len(seen) - len(source.articles),
        )

        return SourceResponse(
            url=url,
            domain=parsed_base.netloc,
            article_urls=article_urls,
            total=len(article_urls),
        )
    except Exception as exc:
        log.exception("Failed to scrape source %s", url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


@app.get("/popular", summary="Danh sách các nguồn tin phổ biến")
def popular_urls():
    """Trả về danh sách URL các nguồn tin tức phổ biến được định sẵn trong thư viện."""
    return {"urls": newspaper.popular_urls()}


# ---------------------------------------------------------------------------
# Google Trends categories (Google Trends Daily RSS `cat` parameter values)
# ---------------------------------------------------------------------------
_TREND_CATEGORIES: dict[str, str] = {
    "all":           "",   # tất cả
    "business":      "b",  # kinh doanh
    "entertainment": "e",  # giải trí
    "health":        "m",  # sức khỏe
    "sports":        "s",  # thể thao
    "tech":          "t",  # công nghệ
}


def _fetch_trends(geo: str = "VN", hl: str = "vi", category: str = "") -> list[dict]:
    """Fetch and parse Google Trends Daily RSS, returning rich trend objects.

    Feedparser only captures the *last* ht:news_item per <item>; we parse
    the raw XML ourselves to collect all related articles per trend entry.
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    cat_code = _TREND_CATEGORIES.get(category, category)
    url = f"https://trends.google.com/trending/rss?geo={geo}&hl={hl}"
    if cat_code:
        url += f"&cat={cat_code}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            xml_bytes = resp.read()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Google Trends fetch failed: {exc}") from exc

    NS = "https://trends.google.com/trending/rss"
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []

    results = []
    for item in channel.findall("item"):
        title_el = item.find("title")
        pub_el   = item.find("pubDate")
        traffic_el = item.find(f"{{{NS}}}approx_traffic")
        picture_el = item.find(f"{{{NS}}}picture")

        # Collect all related news articles (up to 3 per trend)
        related = []
        for ni in item.findall(f"{{{NS}}}news_item"):
            ni_title = ni.find(f"{{{NS}}}news_item_title")
            ni_url   = ni.find(f"{{{NS}}}news_item_url")
            ni_src   = ni.find(f"{{{NS}}}news_item_source")
            ni_pic   = ni.find(f"{{{NS}}}news_item_picture")
            related.append({
                "title":  ni_title.text if ni_title is not None else None,
                "url":    ni_url.text   if ni_url   is not None else None,
                "source": ni_src.text   if ni_src   is not None else None,
                "image":  ni_pic.text   if ni_pic   is not None else None,
            })

        results.append({
            "keyword":      title_el.text  if title_el  is not None else None,
            "traffic":      traffic_el.text if traffic_el is not None else None,
            "published_at": pub_el.text    if pub_el    is not None else None,
            "image":        picture_el.text if picture_el is not None else None,
            "category":     category or "all",
            "related":      related,
        })

    return results


@app.get("/hot", summary="Các từ khóa đang hot trên Google Trends")
def hot_trends(
    geo: str = Query(default="VN", description="Mã quốc gia ISO (VN, US, JP…)"),
    hl: str = Query(default="vi", description="Ngôn ngữ hiển thị (vi, en…)"),
    category: str = Query(
        default="all",
        description="Danh mục: all | business | entertainment | health | sports | tech",
    ),
):
    """Trả về danh sách xu hướng tìm kiếm trên Google Trends, kèm bài liên quan.

    - Mỗi trend có: keyword, traffic, published_at, image, category, related (tối đa 3 bài)
    - Nếu `category=all`, tổng hợp tất cả danh mục và trả về danh sách đã dedup.
    """
    if category == "all":
        seen: set[str] = set()
        merged: list[dict] = []
        # Fetch all categories in parallel — each is an independent HTTP request.
        _cats = [c for c in _TREND_CATEGORIES if c != "all"]
        with ThreadPoolExecutor(max_workers=len(_cats)) as _ex:
            _futs = {_ex.submit(_fetch_trends, geo=geo, hl=hl, category=c): c for c in _cats}
            for _f in as_completed(_futs):
                for trend in _f.result():
                    key = (trend["keyword"] or "").lower()
                    if key not in seen:
                        seen.add(key)
                        merged.append(trend)
        # Sort by traffic descending (e.g. "500+" > "200+" > "100+")
        def _traffic_val(t: dict) -> int:
            tr = t.get("traffic") or "0"
            return int(tr.replace("+", "").replace(",", "") or 0)
        merged.sort(key=_traffic_val, reverse=True)
        return {"total": len(merged), "geo": geo, "trends": merged}
    else:
        if category not in _TREND_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category '{category}'. Valid: {list(_TREND_CATEGORIES.keys())}",
            )
        trends = _fetch_trends(geo=geo, hl=hl, category=category)
        return {"total": len(trends), "geo": geo, "category": category, "trends": trends}
