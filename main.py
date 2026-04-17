"""FastAPI server for crawling articles using newspaper4k."""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from markdownify import MarkdownConverter
from pydantic import BaseModel, HttpUrl

import newspaper
from newspaper import Article
from newspaper.configuration import Configuration

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# CSS selectors / attributes whose elements should be stripped from the
# article body before converting to markdown (non-exhaustive but covers
# most common boilerplate).
_REMOVE_SELECTORS = [
    "script", "style", "noscript", "iframe",
    "nav", "header", "footer",
    # Related / recommended content boxes
    '[type="RelatedOneNews"]',
    '[class*="related"]', '[id*="related"]',
    '[class*="recommend"]', '[id*="recommend"]',
    # Social sharing / follow widgets
    '[class*="social"]', '[class*="share"]',
    '[class*="follow"]', '[class*="subscribe"]',
    # Ads / promo
    '[class*=" ad "]', '[class*="advertisement"]',
    '[class*="promo"]', '[id*="promo"]',
    # Comments
    '[class*="comment"]', '[id*="comment"]',
    # Newsletter signup
    '[class*="newsletter"]', '[id*="newsletter"]',
]

# Ordered list of selectors used to locate the main article body container.
_ARTICLE_BODY_SELECTORS = [
    '[itemprop="articleBody"]',
    '[data-role="content"]',
    "article",
    '[class*="article-body"]',
    '[class*="article-content"]',
    '[id*="article-body"]',
    '[id*="article-content"]',
    '[class*="post-content"]',
    '[class*="entry-content"]',
    '[class*="detail-content"]',
]


def _build_markdown(raw_html: str) -> str:
    """Convert raw article page HTML to clean Markdown, including images.

    Strategy:
    1. Find the main article body using common selectors.
    2. Strip non-content boilerplate inside the container.
    3. Normalise lazy-loaded images (data-original / data-src → src).
    4. Promote figcaption text into the img alt attribute, then remove it.
    5. Run markdownify on the resulting HTML.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # 1. Locate article body container.
    container = None
    for selector in _ARTICLE_BODY_SELECTORS:
        container = soup.select_one(selector)
        if container:
            break
    if container is None:
        container = soup.find("body") or soup

    # Work on a copy so we don't mutate the original soup.
    from copy import deepcopy
    container = deepcopy(container)

    # 2. Remove boilerplate elements.
    for selector in _REMOVE_SELECTORS:
        for el in container.select(selector):
            el.decompose()

    # 3. Fix lazy-loaded images.
    for img in container.find_all("img"):
        lazy_src = img.get("data-original") or img.get("data-src") or img.get("data-lazy-src")
        if lazy_src:
            img["src"] = lazy_src

    # 4. Move figcaption text into img alt, then remove the figcaption element.
    for fig in container.find_all("figure"):
        img = fig.find("img")
        cap = fig.find("figcaption")
        if img and cap:
            caption_text = cap.get_text(" ", strip=True)
            if caption_text:
                img["alt"] = caption_text
                img["title"] = caption_text
        if cap:
            cap.decompose()

    # 5. Convert to Markdown.
    md = MarkdownConverter(heading_style="ATX", bullets="-").convert_soup(container)

    # Light post-processing: collapse 3+ consecutive blank lines to 2.
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

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


class CrawlRequest(BaseModel):
    url: HttpUrl
    language: Optional[str] = None
    follow_meta_refresh: bool = False
    keep_article_html: bool = False


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/crawl", response_model=ArticleResponse, summary="Crawl article from URL")
def crawl_get(
    url: str = Query(..., description="URL of the article to crawl"),
    language: Optional[str] = Query(None, description="Language code, e.g. 'en', 'vi', 'zh'"),
    follow_meta_refresh: bool = Query(False, description="Follow meta refresh redirects"),
):
    """Crawl and extract article content from the given URL (GET)."""
    return _crawl(url, language=language, follow_meta_refresh=follow_meta_refresh)


@app.post("/crawl", response_model=ArticleResponse, summary="Crawl article from URL")
def crawl_post(body: CrawlRequest):
    """Crawl and extract article content from the given URL (POST)."""
    return _crawl(
        str(body.url),
        language=body.language,
        follow_meta_refresh=body.follow_meta_refresh,
        keep_article_html=body.keep_article_html,
    )


def _crawl(
    url: str,
    language: Optional[str] = None,
    follow_meta_refresh: bool = False,
    keep_article_html: bool = False,
) -> ArticleResponse:
    try:
        config = Configuration()
        if language:
            config.language = language
        config.follow_meta_refresh = follow_meta_refresh
        config.keep_article_html = keep_article_html

        article = Article(url, config=config)
        article.download()
        article.parse()

        publish_date = (
            article.publish_date.isoformat() if article.publish_date else None
        )

        text_markdown = ""
        try:
            text_markdown = _build_markdown(article.html)
        except Exception:
            log.warning("Could not build markdown for %s", url, exc_info=True)

        return ArticleResponse(
            url=article.url,
            title=article.title or "",
            authors=article.authors or [],
            publish_date=publish_date,
            text=article.text or "",
            text_markdown=text_markdown,
            top_image=article.top_image or None,
            images=list(article.images) if article.images else [],
            movies=article.movies or [],
            meta_keywords=article.meta_keywords or [],
            tags=list(article.tags) if article.tags else [],
            meta_description=article.meta_description or None,
            meta_lang=article.meta_lang or None,
            source_url=article.source_url or None,
        )
    except Exception as exc:
        log.exception("Failed to crawl %s", url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
