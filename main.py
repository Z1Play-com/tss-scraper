"""FastAPI server for crawling articles using newspaper4k."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl

from newspaper import Article
from newspaper.configuration import Configuration
import newspaper

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
            article_type=article.article_type,
        )
    except Exception as exc:
        log.exception("Failed to crawl %s", url)
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
        for cat_name in _TREND_CATEGORIES:
            if cat_name == "all":
                continue
            for trend in _fetch_trends(geo=geo, hl=hl, category=cat_name):
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
