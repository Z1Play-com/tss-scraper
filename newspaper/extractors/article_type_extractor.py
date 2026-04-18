"""Extractor that classifies an article into a content type.

Supported types (extensible):
  normal       — standard text/news article (default)
  video        — main content is a video
  podcast      — audio / podcast episode
  longform     — long-form journalism
  infographic  — data-visual / infographic
  photo        — photo gallery / photo story
  live         — live blog / live ticker

Detection pipeline (highest-priority first):
  1. Per-domain URL-pattern rules
  2. Generic URL-path patterns
  3. Schema.org JSON-LD ``@type``
  4. Open Graph ``og:type``
  5. Body / article element class+id signals
  6. Structural HTML signals (video/audio tags, image density)
  7. Word-count longform threshold
  8. Fallback: ``normal``
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from lxml.html import HtmlElement

import newspaper.parsers as parsers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONGFORM_MIN_WORDS = 1800

# Per-domain URL-pattern rules: {domain_suffix: [(pattern, type), ...]}
# Patterns are matched (re.search) against the full URL (case-insensitive).
# More-specific rules should be listed before less-specific ones.
_DOMAIN_URL_RULES: dict[str, list[tuple[str, str]]] = {
    "vnexpress.net": [
        (r"/video-", "video"),
        (r"/infographic-", "infographic"),
        (r"/podcast-", "podcast"),
        (r"/longform-", "longform"),
        (r"/photo-", "photo"),
    ],
    "tuoitre.vn": [
        (r"/video\b|/video-|/video\.htm", "video"),
        (r"/longform-|/long-form-", "longform"),
        (r"/infographic-", "infographic"),
        (r"/anh-|/photo-", "photo"),
    ],
    "thanhnien.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/photo-|/anh-", "photo"),
    ],
    "tienphong.vn": [
        (r"/video-", "video"),
        (r"/podcast-", "podcast"),
        (r"/longform-|/long-form-", "longform"),
        (r"/infographic-", "infographic"),
        (r"/photo-|/anh-", "photo"),
    ],
    "znews.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/anh-|/photo-", "photo"),
    ],
    "zingnews.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
    ],
    "dantri.com.vn": [
        (r"/video-", "video"),
        (r"/podcast-", "podcast"),
        (r"/longform-|/long-form-", "longform"),
        (r"/infographic-", "infographic"),
        (r"/anh-|/photo-", "photo"),
    ],
    "kenh14.vn": [
        (r"/video-|/video/", "video"),
        (r"/longform-", "longform"),
        (r"/infographic-", "infographic"),
    ],
    "vietnamplus.vn": [
        (r"/video-", "video"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/longform-", "longform"),
    ],
    "vtv.vn": [
        (r"/video-", "video"),
        (r"/podcast-", "podcast"),
    ],
    "laodong.vn": [
        (r"/video-|/clip-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
    ],
    "plo.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
    ],
    "24h.com.vn": [
        (r"/video-|/clip-", "video"),
        (r"/podcast-", "podcast"),
    ],
    "soha.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
    ],
    "cafef.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
    ],
    "cafebiz.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
    ],
    "ndh.vn": [
        (r"/video-", "video"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
    ],
    "baomoi.com": [
        (r"/video-|/clip-", "video"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/longform-", "longform"),
    ],
    "eva.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
    ],
    "nld.com.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/photo-|/anh-", "photo"),
    ],
    "nhandan.vn": [
        (r"/video-", "video"),
        (r"/longform-", "longform"),
        (r"/podcast-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/anh-|/photo-", "photo"),
    ],
    "vov.vn": [
        (r"/video-", "video"),
        (r"/podcast-|/audio-", "podcast"),
        (r"/infographic-", "infographic"),
        (r"/longform-", "longform"),
    ],
    "vnanet.vn": [
        (r"/video-", "video"),
        (r"/infographic-", "infographic"),
        (r"/photo-", "photo"),
    ],
    "baotintuc.vn": [
        (r"/video-", "video"),
        (r"/infographic-", "infographic"),
        (r"/photo-|/anh-", "photo"),
    ],
    "cnn.com": [
        (r"/video/", "video"),
        (r"/interactive/", "infographic"),
    ],
    "bbc.com": [
        (r"/av/", "video"),
        (r"/sounds/", "podcast"),
        (r"/news/resources/", "longform"),
        (r"/news/extra/", "longform"),
        (r"/news/in-pictures", "photo"),
    ],
    "bbc.co.uk": [
        (r"/av/", "video"),
        (r"/sounds/", "podcast"),
        (r"/news/resources/", "longform"),
        (r"/news/in-pictures", "photo"),
    ],
    "theguardian.com": [
        (r"/video/", "video"),
        (r"/podcast/", "podcast"),
        (r"/interactive/", "infographic"),
        (r"/ng-interactive/", "infographic"),
        (r"/gallery/", "photo"),
    ],
    "nytimes.com": [
        (r"/video/", "video"),
        (r"/interactive/", "infographic"),
        (r"/slideshow/", "photo"),
    ],
    "reuters.com": [
        (r"/video/", "video"),
        (r"/graphics/", "infographic"),
        (r"/pictures/", "photo"),
        (r"/investigates/", "longform"),
    ],
}

# Generic URL-path patterns applied when no domain-specific rule matches
_GENERIC_URL_PATTERNS: list[tuple[str, str]] = [
    (r"/video/|/videos?-|[/_]video[/_-]", "video"),
    (r"/podcast/|/podcasts?-|[/_]podcast[/_-]|/audio/", "podcast"),
    (r"/longform/|/long-form/", "longform"),
    (r"/infographic/|/infographics?-|[/_]infographic[/_-]", "infographic"),
    (r"/photo-story/|/photo-gallery/|/gallery/|/in-pictures", "photo"),
    (r"/live/|/live-blog/|/liveblog/", "live"),
]

# Schema.org @type → article_type
_SCHEMA_TYPE_MAP: dict[str, str] = {
    "VideoObject": "video",
    "VideoNewsArticle": "video",
    "PodcastEpisode": "podcast",
    "PodcastSeries": "podcast",
    "AudioObject": "podcast",
    "Infographic": "infographic",
    "ImageGallery": "photo",
    "MediaGallery": "photo",
    "ImageObject": "photo",
    "LiveBlogPosting": "live",
    "AnalysisNewsArticle": "longform",
    "BackgroundNewsArticle": "longform",
    "NewsArticle": "normal",
    "ReportageNewsArticle": "normal",
    "ReviewNewsArticle": "normal",
    "OpinionNewsArticle": "normal",
    "Article": "normal",
    "BlogPosting": "normal",
}

# og:type → article_type
_OG_TYPE_MAP: dict[str, str] = {
    "video": "video",
    "video.movie": "video",
    "video.episode": "video",
    "video.other": "video",
    "music.song": "podcast",
    "music.album": "podcast",
    "article": "normal",
    "website": "normal",
}

# class/id text signals on structural elements (body, main, article, div#content…)
_CLASS_ID_SIGNALS: list[tuple[str, str]] = [
    (r"\bvideo\b", "video"),
    (r"\bpodcast\b", "podcast"),
    (r"\binfographic\b", "infographic"),
    (r"\blongform\b|\blong-form\b|\blong_form\b", "longform"),
    (r"\bgallery\b|\bphoto-story\b|\bphotojournal\b|\bphotostory\b", "photo"),
    (r"\blive-blog\b|\bliveblog\b|\blive_blog\b", "live"),
]


def _domain_matches(hostname: str, suffix: str) -> bool:
    """Return True if *hostname* is or ends with *suffix*."""
    return hostname == suffix or hostname.endswith("." + suffix)


class ArticleTypeExtractor:
    """Detect article content type from URL, HTML, and text signals."""

    def detect(
        self,
        url: str,
        doc: HtmlElement | None,
        text: str = "",
    ) -> str:
        """Return the article type string for the given article.

        Args:
            url: Full article URL.
            doc: Parsed lxml HtmlElement of the page.
            text: Extracted plain text (used for word-count longform check).

        Returns:
            One of: ``normal``, ``video``, ``podcast``, ``longform``,
            ``infographic``, ``photo``, ``live``.
        """
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().lstrip("www.")

        # 1. Per-domain URL rules
        for domain_suffix, rules in _DOMAIN_URL_RULES.items():
            if _domain_matches(hostname, domain_suffix):
                for pattern, atype in rules:
                    if re.search(pattern, url, re.IGNORECASE):
                        return atype
                break  # domain matched but no URL rule hit → continue pipeline

        # 2. Generic URL-path patterns
        path_and_url = parsed.path + "?" + (parsed.query or "")
        for pattern, atype in _GENERIC_URL_PATTERNS:
            if re.search(pattern, path_and_url, re.IGNORECASE):
                return atype

        if doc is None:
            return "normal"

        # 3. Schema.org JSON-LD
        result = self._from_jsonld(doc)
        if result:
            return result

        # 4. Open Graph og:type
        result = self._from_og_type(doc)
        if result:
            return result

        # 5. CSS class/id signals on structural elements
        result = self._from_class_id(doc)
        if result:
            return result

        # 6. Structural HTML signals
        result = self._from_html_structure(doc)
        if result:
            return result

        # 7. Longform by word count
        if text and len(text.split()) >= LONGFORM_MIN_WORDS:
            return "longform"

        return "normal"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _from_jsonld(self, doc: HtmlElement) -> str:
        """Extract article type from schema.org JSON-LD blocks."""
        scripts = parsers.get_tags(doc, "script", attribs={"type": "application/ld+json"})
        for script in scripts:
            raw = parsers.get_text(script)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            # data may be a single object or a list
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                continue
            for obj in data:
                schema_type = obj.get("@type", "")
                if isinstance(schema_type, list):
                    # Check all types in array; prefer non-normal matches
                    for t in schema_type:
                        mapped = _SCHEMA_TYPE_MAP.get(t, "")
                        if mapped and mapped != "normal":
                            return mapped
                    # Fall back to first mapped type
                    for t in schema_type:
                        mapped = _SCHEMA_TYPE_MAP.get(t, "")
                        if mapped:
                            return mapped
                elif isinstance(schema_type, str):
                    mapped = _SCHEMA_TYPE_MAP.get(schema_type, "")
                    if mapped:
                        return mapped
                # Recurse into @graph
                graph = obj.get("@graph", [])
                if isinstance(graph, list):
                    for node in graph:
                        if not isinstance(node, dict):
                            continue
                        t = node.get("@type", "")
                        mapped = _SCHEMA_TYPE_MAP.get(t, "")
                        if mapped:
                            return mapped
        return ""

    def _from_og_type(self, doc: HtmlElement) -> str:
        """Extract article type from og:type meta tag."""
        metas = parsers.get_tags(doc, "meta", attribs={"property": "og:type"})
        for meta in metas:
            content = (meta.get("content") or "").strip().lower()
            mapped = _OG_TYPE_MAP.get(content, "")
            if mapped and mapped != "normal":
                return mapped
        return ""

    def _from_class_id(self, doc: HtmlElement) -> str:
        """Check class/id attributes on body, main, article, and section tags."""
        check_tags = ["body", "main", "article", "section", "div"]
        for tag in check_tags:
            for el in parsers.get_tags(doc, tag)[:5]:  # check only first few
                attrs = " ".join([
                    el.get("class", ""),
                    el.get("id", ""),
                    el.get("data-type", ""),
                    el.get("data-article-type", ""),
                    el.get("data-content-type", ""),
                ])
                for pattern, atype in _CLASS_ID_SIGNALS:
                    if re.search(pattern, attrs, re.IGNORECASE):
                        return atype
        return ""

    def _from_html_structure(self, doc: HtmlElement) -> str:
        """Infer type from presence of video/audio tags and image density."""
        # Prominent video element → video
        videos = parsers.get_tags(doc, "video")
        if videos:
            # Confirm it's a main video (has src/source child or poster)
            for v in videos:
                if v.get("src") or v.get("poster") or parsers.get_tags(v, "source"):
                    return "video"

        # Audio element → podcast
        audios = parsers.get_tags(doc, "audio")
        if audios:
            for a in audios:
                if a.get("src") or parsers.get_tags(a, "source"):
                    return "podcast"

        return ""
