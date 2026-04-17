"""FastAPI server for crawling articles using newspaper4k."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl

from newspaper import Article
from newspaper.configuration import Configuration

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

        return ArticleResponse(
            url=article.url,
            title=article.title or "",
            authors=article.authors or [],
            publish_date=publish_date,
            text=article.text or "",
            text_markdown=article.text_markdown,
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
