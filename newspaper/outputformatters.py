# Much of the code here was forked from https://github.com/codelucas/newspaper
# Copyright (c) Lucas Ou-Yang (codelucas)

"""Module provinding the OutputFormatter class, which converts the article top node
to plain text, removing most boilerplate and other unwanted elements.
"""

import logging
import re
from copy import deepcopy
from statistics import mean, stdev
from typing import Any

from lxml.html import HtmlElement
from lxml_html_clean import Cleaner

from newspaper import parsers, settings
from newspaper.configuration import Configuration

log = logging.getLogger(__name__)

WHITESPACE_CHARS = "\n\r\t " + "\u00a0" + "\ufeff"
MAX_PARAGRAPH_BEFORE_TITLE = 200

# ---------------------------------------------------------------------------
# Markdown conversion helpers
# ---------------------------------------------------------------------------

# Elements to strip before markdown conversion (boilerplate, ads, related
# articles, social widgets, etc.).  Ordered from most-specific to least.
_MD_REMOVE_SELECTORS = [
    "script", "style", "noscript", "iframe",
    "nav", "header", "footer",
    '[type="RelatedOneNews"]',          # tuoitre.vn related box
    '.article-relate',
    'table.article',                    # znews.vn inline related-article tables
    '[class*="inner-article"]',
    '[class*="related"]', '[id*="related"]',
    '[class*="recommend"]', '[id*="recommend"]',
    '[class*="read-more"]', '[class*="readmore"]',
    '[class*="news-relation"]',         # techz.vn related box
    '[id="original_link"]',             # techz.vn source link box
    '[class*="notebox"]',
    '[class*="evtBox"]',               # eva.vn related/promo boxes
    '.banner-ads',                      # vnexpress.net ad banners
    '.width-detail-photo',              # vnexpress.net author footer
    '[class*="social"]', '[class*="share"]',
    '[class*="follow"]', '[class*="subscribe"]',
    '[class*=" ad "]', '[class*="advertisement"]',
    '[class*="promo"]', '[id*="promo"]',
    '[class*="comment"]', '[id*="comment"]',
    '[class*="newsletter"]', '[id*="newsletter"]',
    # tienphong.vn
    '.breadcrumb', '.breadcrumb-detail',
    '.img-ggnews',                      # Google News follow button
    '.article__header',                 # repeated title + author + date block
    '.article__social', '.audio-social',  # share/audio toolbar
    '.article-footer',                  # author name + hashtag block at bottom
    '.rennab',                          # ad placeholder divs
    '[class*="article__tag"]', '[class*="tag-list"]',
]

# Ordered list of selectors to locate the main article body container.
# More specific selectors first; generic "article" tag is last resort.
_MD_BODY_SELECTORS = [
    '[itemprop="articleBody"]',
    '[data-role="content"]',
    '[class*="article-body"]',
    '[class*="article-content"]',
    '[id*="article-body"]',
    '[id*="article-content"]',
    '[class*="post-content"]',
    '[class*="entry-content"]',
    '[class*="detail-content"]',
    '[class*="entry-body"]',
    ".fck_detail",          # vnexpress.net article body
    "[class*='article__body']",  # tienphong.vn / sites using BEM naming
    "[class*='cms-body']",
    "[class*='zce-content']",
    "article",
]


def build_markdown(raw_html: str, top_node_html: str | None = None) -> str:
    """Convert a raw article page HTML string to clean Markdown with images,
    videos and inline captions.

    Strategy:
    1. Locate the article body container using common CSS selectors.
    2. Strip non-content boilerplate (ads, related articles, social widgets…).
    3. Pre-process site-specific media elements before conversion:
       - ``figure.video`` → ``<figure>`` with ``<img>`` (thumbnail) and
         ``<figcaption>`` (video caption / title).
       - ``table.picture`` → ``<figure>`` with ``<img>`` and ``<figcaption>``.
       - Lazy-loaded ``<img>`` → resolve ``data-src`` / ``data-original``.
    4. Convert to Markdown using a custom ``MarkdownConverter`` subclass that
       renders ``<figure>`` as ``![alt](url)\\n*caption*``.

    Args:
        raw_html: The full HTML of the downloaded page (``article.html``).

    Returns:
        A Markdown string, or an empty string if conversion fails.
    """
    try:
        from bs4 import BeautifulSoup, Tag
        from markdownify import MarkdownConverter
    except ImportError:
        log.warning(
            "markdownify and/or beautifulsoup4 are required for Markdown output. "
            "Install them with: pip install markdownify"
        )
        return ""

    # ------------------------------------------------------------------
    # Custom converter: render <figure> as image + italic caption below.
    # ------------------------------------------------------------------
    class _ArticleConverter(MarkdownConverter):
        def convert_figure(self, el, text, **kwargs):
            img = el.find("img")
            cap = el.find("figcaption")
            link = el.find("a")
            if img is None:
                return text or ""
            src = img.get("src", "")
            alt = img.get("alt", "").replace("\n", " ").strip()
            caption = cap.get_text(" ", strip=True) if cap else ""
            caption = caption.replace("\n", " ").strip()
            display_alt = caption or alt
            href = link.get("href", "") if link else ""
            if href:
                md = f"\n[![{display_alt}]({src})]({href})\n"
            else:
                md = f"\n![{display_alt}]({src})\n"
            if caption:
                md += f"*{caption}*\n"
            return md

    # 1. Find article body container.
    # Prefer newspaper's already-computed top_node when it contains enough
    # content (>= 500 text chars).  If newspaper mis-identified the top_node
    # (e.g. picked a photo caption div), fall back to CSS selector search.
    _MIN_TOP_NODE_TEXT = 500
    container = None
    if top_node_html:
        _tn_soup = BeautifulSoup(top_node_html, "lxml")
        _tn_body = _tn_soup.find("body")
        _tn_root = next(
            (c for c in (_tn_body.children if _tn_body else []) if hasattr(c, "name") and c.name),
            _tn_body,
        )
        if _tn_root and len(_tn_root.get_text()) >= _MIN_TOP_NODE_TEXT:
            container = _tn_root

    if container is None:
        soup = BeautifulSoup(raw_html, "lxml")
        for selector in _MD_BODY_SELECTORS:
            container = soup.select_one(selector)
            if container:
                break
        if container is None:
            container = soup.find("body") or soup

    container = deepcopy(container)

    # 2. Remove boilerplate.
    for selector in _MD_REMOVE_SELECTORS:
        for el in container.select(selector):
            el.decompose()

    # 3a. Normalise vnexpress-style photo galleries:
    #     Each slide is a <div class="item_slide_show"> with:
    #     - <div class="block_thumb_slide_show" data-src="..."> for the image
    #     - <p class="Normal"> for the caption
    #     Duplicate slides (same URL) are skipped.
    #
    #     Also handle <div data-component="true"> which are JS-rendered gallery
    #     components: the real image URL is in data-component-back and caption
    #     in data-component-caption (HTML-encoded JSON string).
    _seen_slide_urls: set = set()

    # Convert data-component gallery elements first (before item_slide_show),
    # so their placeholder <img src> doesn't cause phantom images in step 4.
    for comp in container.find_all("div", attrs={"data-component": "true"}):
        img_url = comp.get("data-component-back", "").strip()
        if not img_url:
            comp.decompose()
            continue
        # data-component-caption is a JSON-encoded HTML string; decode it.
        cap_raw = comp.get("data-component-caption", "")
        try:
            import json as _json
            cap_raw = _json.loads(cap_raw)
        except Exception:
            pass
        caption = BeautifulSoup(cap_raw, "lxml").get_text(" ", strip=True)
        if img_url not in _seen_slide_urls:
            _seen_slide_urls.add(img_url)
            cap_html = f"<figcaption>{caption}</figcaption>" if caption else ""
            new_fig = BeautifulSoup(
                f'<figure><img src="{img_url}" alt="{caption}">{cap_html}</figure>', "lxml"
            ).find("figure")
            comp.replace_with(new_fig)
        else:
            comp.decompose()

    for slide in container.find_all("div", class_="item_slide_show"):
        bt = slide.find("div", class_="block_thumb_slide_show")
        img_url = bt.get("data-src", "") if bt else ""
        # skip if no image or already seen
        if not img_url or img_url in _seen_slide_urls:
            slide.decompose()
            continue
        _seen_slide_urls.add(img_url)
        cap_el = slide.find("p", class_="Normal")
        caption = cap_el.get_text(" ", strip=True) if cap_el else ""
        cap_html = f"<figcaption>{caption}</figcaption>" if caption else ""
        new_fig = BeautifulSoup(
            f'<figure><img src="{img_url}" alt="{caption}">{cap_html}</figure>', "lxml"
        ).find("figure")
        slide.replace_with(new_fig)
    # Remove alternate gallery representations that duplicate the slides above.
    for el in container.select(".gallery-detail-photo, .gallery_block, .medium-insert-embed"):
        el.decompose()

    # 3b. Normalise video figures → <figure><img><figcaption>
    #     (e.g. znews.vn <figure class="video cms-video" data-video-src="...">)
    for fig in container.find_all("figure", class_=re.compile(r"\bvideo\b")):
        video_url = (
            fig.get("data-video-src")
            or fig.get("source-url")
            or ""
        )
        # Thumbnail from inner div background-image or <video poster>
        thumb_url = ""
        inner_div = fig.find("div", style=True)
        if inner_div:
            m = re.search(r"background-image:\s*url\(['\"]?([^'\")\s]+)['\"]?\)", inner_div.get("style", ""))
            if m:
                thumb_url = m.group(1)
        if not thumb_url:
            video_tag = fig.find("video")
            if video_tag:
                thumb_url = video_tag.get("poster", "")

        # Caption from existing figcaption: prefer the <strong> title link
        # over the full description text that some sites embed there.
        figcap = fig.find("figcaption")
        caption_text = ""
        if figcap:
            strong = figcap.find("strong")
            if strong:
                caption_text = strong.get_text(" ", strip=True)
            else:
                caption_text = figcap.get_text(" ", strip=True)

        # Build replacement <figure>
        img_html = f'<img src="{thumb_url}" alt="▶ Video">' if thumb_url else ""
        cap_html = f"<figcaption>{caption_text}</figcaption>" if caption_text else ""
        link_html = f'<a href="{video_url}">{img_html or "▶ Video"}</a>'
        new_fig = BeautifulSoup(f"<figure>{link_html}{cap_html}</figure>", "lxml").find("figure")
        fig.replace_with(new_fig)

    # 3b. Normalise picture tables → <figure><img><figcaption>
    #     (e.g. znews.vn <table class="picture">)
    for tbl in container.find_all("table", class_=re.compile(r"\bpicture\b")):
        img = tbl.find("img")
        cap_td = tbl.find("td", class_=re.compile(r"\bcaption\b|\bpCaption\b"))
        if not img:
            tbl.decompose()
            continue
        real_src = (
            img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy-src")
            or img.get("src", "")
        )
        if real_src.startswith("data:"):
            real_src = ""
        if not real_src:
            tbl.decompose()
            continue
        caption = cap_td.get_text(" ", strip=True) if cap_td else img.get("title") or img.get("alt", "")
        cap_html = f"<figcaption>{caption}</figcaption>" if caption else ""
        new_fig = BeautifulSoup(
            f'<figure><img src="{real_src}" alt="{caption}">{cap_html}</figure>', "lxml"
        ).find("figure")
        tbl.replace_with(new_fig)

    # 4. Fix lazy-loaded images (general case).
    for img in container.find_all("img"):
        lazy_src = (
            img.get("data-original")
            or img.get("data-src")
            or img.get("data-lazy-src")
        )
        if lazy_src and not lazy_src.startswith("data:"):
            img["src"] = lazy_src
        elif img.get("src", "").startswith("data:") and not lazy_src:
            img.decompose()

    # 5. Normalise plain figures (tuoitre.vn etc.):
    #    fix lazy img src, keep figcaption for convert_figure above.
    for fig in container.find_all("figure"):
        img = fig.find("img")
        if img:
            lazy_src = (
                img.get("data-original")
                or img.get("data-src")
                or img.get("data-lazy-src")
            )
            if lazy_src and not lazy_src.startswith("data:"):
                img["src"] = lazy_src

    # 6. Convert to Markdown.
    md = _ArticleConverter(heading_style="ATX", bullets="-").convert_soup(container)

    # Collapse 3+ consecutive blank lines to 2.
    md = re.sub(r"\n{3,}", "\n\n", md)

    # Strip Vietnamese newspaper abbreviation prefixes at the start of a
    # paragraph.  Two forms:
    #   "TPO - ", "HNM - ", "NDO - ", "ANTD - ", "GD&TĐ - "
    #   "(NLĐO)- ", "(TTXVN)- ", "(SGGP)- "
    _UPPER = r"A-ZĐÁÀẢÃẠĂẮẶẰẲẴÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    md = re.sub(
        rf"(?m)^\(([{_UPPER}][{_UPPER}0-9&/]{{1,9}})\)\s*[-–]\s*",
        "",
        md,
    )
    md = re.sub(
        rf"(?m)^([{_UPPER}][{_UPPER}0-9&/]{{1,7}})\s*[-–]\s+",
        "",
        md,
    )

    return md.strip()


class OutputFormatter:
    """Class that converts the article top node into text, cleaning up
    debris tags, replacing <br> with newlines, etc.

    if `config.clean_article_html` is True, then the article's html is
    cleaned as well. Only `settings.CLEAN_ARTICLE_TAGS` are allowed to
    remain in the html.
    """

    def __init__(self, config=None):
        self.config = config or Configuration()

    def get_formatted(self, top_node: HtmlElement, article_title: str | None = None) -> tuple[str, str]:
        """Returns the body text of an article, and also the cleaned html body
        article of the article.

        Arguments:
            top_node {HtmlElement} -- The top node element of the article
            article_title {str} -- The title of the article, if available, to
                be removed from the text (and max 1 paragraph before it)

        Returns:
            Tuple[str, str] -- The body text of the article, and the cleaned
            html body of the article
        """
        html, text = "", ""
        if top_node is None:
            return (text, html)

        node_cleaned = deepcopy(top_node)

        self._remove_negativescores_nodes(node_cleaned)

        if not self.config.clean_article_html:
            # We deliver the HTML untouched (only the negative nodes are removed)
            html = parsers.node_to_string(node_cleaned)

        self._remove_advertisement_nodes(node_cleaned)

        self._remove_unlikely_nodes(node_cleaned)

        self._remove_empty_tags(node_cleaned)

        # removes some same level tags that might
        # contain non-content like menus, gallery,  etc.
        # this can misfire on some sites
        self._remove_trailing_media_div(node_cleaned)

        if self.config.clean_article_html:
            html = self._create_clean_html(node_cleaned)

        text = self._convert_to_text(node_cleaned, article_title)

        return (text, html)

    def _convert_to_text(self, top_node: HtmlElement, article_title: str | None = None) -> str:
        article_cleaner = Cleaner(
            javascript=True,
            style=True,
            remove_unknown_tags=False,
            meta=True,
            embedded=True,
            frames=True,
            allow_tags=settings.BLOCK_LEVEL_TAGS + ["br"],
        )

        cleaned_node = article_cleaner.clean_html(top_node)
        # TODO: do not remove newlines in <pre> tags

        txts = [re.sub(r"[\s\t\xa0\uFEFF]+", " ", value, flags=re.UNICODE) for value in cleaned_node.itertext()]
        txts = [x.strip(" \t") for x in txts if x.strip(WHITESPACE_CHARS)]
        if article_title and len(txts) > 1:
            # Remove the title and the first paragraph before it
            # (if it's not too long)
            def normalize_string(s: str) -> str:
                # remove punctuation, double spaces and lowers the case
                s = re.sub(r"[^\w\s]", "", s)
                s = re.sub(r"\s+", " ", s)
                s = s.lower()
                return s

            if normalize_string(txts[0]) == normalize_string(article_title):
                txts = txts[1:]
            elif len(txts[0]) < MAX_PARAGRAPH_BEFORE_TITLE and normalize_string(txts[1]) == normalize_string(
                article_title
            ):
                txts = txts[2:]

        return "\n\n".join(txts)

    def _create_clean_html(self, top_node: HtmlElement):
        article_cleaner = Cleaner(
            javascript=True,
            style=True,
            remove_unknown_tags=False,
            meta=True,
            embedded=True,
            allow_tags=settings.CLEAN_ARTICLE_TAGS,
        )

        cleaned_node = article_cleaner.clean_html(top_node)
        return parsers.node_to_string(cleaned_node)

    def _add_newline_to_br(self, top_node: HtmlElement):
        """Replace all br tags in 'element' with a newline character"""
        br_tags = top_node.xpath(".//br")
        for br in br_tags:
            br.tail = "\n" + br.tail if br.tail else "\n"

    def _remove_negativescores_nodes(self, top_node: HtmlElement):
        """If there are elements inside our top node that have a
        negative gravity score, let's give em the boot.
        """
        gravity_items = top_node.xpath(".//*[@gravityScore]")
        for item in gravity_items:
            score = item.attrib.get("gravityScore", "0")
            score = float(score)
            if score < 1:
                item.getparent().remove(item)

    def _remove_empty_tags(self, top_node: HtmlElement):
        """It's common in top_node to have tags that are filled with data
        in their properties but do not have any displayable text.
        """
        all_nodes = parsers.get_tags(top_node)
        all_nodes.reverse()
        for el in all_nodes:
            tag = el.tag
            if tag == "br":
                continue
            if len(parsers.get_elements_by_tagslist(el, ["object", "embed"])) > 0:
                continue

            txt = parsers.get_text(el)
            txt = re.sub(r"[\s\t]+", "", txt)

            if not txt:
                parsers.remove(el)

    def _get_top_level_nodes(self, top_node: HtmlElement):
        """Returns a list of nodes that are of the top level"""
        top_level_nodes = top_node.getchildren()
        if top_node.tag == "body" and len(top_level_nodes) == 1:
            top_level_nodes = top_level_nodes[0].getchildren()

        return top_level_nodes

    def _remove_trailing_media_div(self, top_node: HtmlElement):
        """Punish the *last top level* node in the top_node if it's
        DOM depth is too deep or has a a lot of links. Many media non-content
        links are eliminated: "related", "loading gallery", etc. It skips
        removal if last top level node's class is one of NON_MEDIA_CLASSES.
        """
        NON_MEDIA_CLASSES = ("zn-body__read-all",)

        top_level_nodes = self._get_top_level_nodes(top_node)

        if len(top_level_nodes) < 3:
            return

        last_node = top_level_nodes[-1]

        last_node_class = parsers.get_attribute(last_node, "class")
        if last_node_class in NON_MEDIA_CLASSES:
            return
        if last_node.tag != "p" and len(parsers.get_tags(last_node, "p")) > 0:
            if parsers.get_node_gravity_score(last_node) > 15:
                return

        if parsers.get_node_depth(last_node) >= 2:
            parsers.remove(last_node)
        elif parsers.is_highlink_density(last_node, self.config.language):
            parsers.remove(last_node)

    def _top_nodes_stats(self, top_node: HtmlElement):
        """Returns a list of top nodes and stats about them"""
        top_nodes = self._get_top_level_nodes(top_node)
        node_stats: dict[str, dict[str, Any]] = {}
        for el in top_nodes:
            node_stats[el.tag] = node_stats.setdefault(el.tag, {"count": 0, "gravity": [], "depth": []})
            node_stats[el.tag]["count"] += 1
            node_stats[el.tag]["gravity"].append(parsers.get_node_gravity_score(el))
            node_stats[el.tag]["depth"].append(parsers.get_node_depth(el))

        node_stats = {
            k: {
                "count": v["count"],
                "gravity_mean": mean(v["gravity"]),
                "gravity_std": stdev(v["gravity"]) if len(v["gravity"]) > 1 else 0,
                "depth": mean(v["depth"]),
                "depth_std": stdev(v["depth"]) if len(v["depth"]) > 1 else 0,
            }
            for k, v in node_stats.items()
        }

        return node_stats

    def _remove_unlikely_nodes(self, top_node: HtmlElement):
        """Remove unlikely top level nodes from the top node
        based on statistical analysis based on depth and gravity score
        """
        stats = self._top_nodes_stats(top_node)
        top_nodes = self._get_top_level_nodes(top_node)

        # has p and divs. Analyse if divs are not boilerplate or ads
        if "p" in stats and "div" in stats:
            for node in top_nodes:
                if node.tag != "div":
                    continue
                gravity = parsers.get_node_gravity_score(node)
                depth = parsers.get_node_depth(node)

                if (
                    depth > round(stats["div"]["depth"] + stats["div"]["depth_std"])
                    or depth > round(stats["p"]["depth"] + stats["p"]["depth_std"])
                    or gravity < stats["p"]["gravity_mean"] - 2 * stats["p"]["gravity_std"]
                    or gravity < stats["div"]["gravity_mean"] - 2 * stats["div"]["gravity_std"]
                ):
                    parsers.remove(node)

    def _remove_advertisement_nodes(self, top_node: HtmlElement):
        """Remove nodes that may contain advertisement content."""
        divs = top_node.xpath(".//div")
        stats = self._top_nodes_stats(top_node)

        for el in divs:
            # Does it contain p tags?
            if len(parsers.get_tags(el, "p")):
                if parsers.is_highlink_density(el, self.config.language):
                    gravity = parsers.get_node_gravity_score(el)
                    if len(stats):
                        limit = max([stats[x]["gravity_mean"] - 2 * stats[x]["gravity_std"] for x in stats])
                    else:
                        limit = 15  # no gravity scores, then remove all

                    if gravity < limit:
                        parsers.remove(el)

                continue

            if parsers.is_highlink_density(el, self.config.language):
                parsers.remove(el)
                continue
            attrs = el.get("class", "") + " " + el.get("id", "")
            if re.search(settings.ADVERTISEMENT_ATTR_VALUES, attrs, re.IGNORECASE):
                parsers.remove(el)
                continue
