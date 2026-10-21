"""Render the finished Markdown article as a portable, accessible reading page."""
import html, re
from pathlib import Path
from markdown_it import MarkdownIt
from html.parser import HTMLParser

CSS = """
:root{color-scheme:light;--paper:#fcfaf6;--ink:#25221f;--muted:#746b62;--line:#dfd8cd;--accent:#bd4c23}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.75 Georgia,'Times New Roman',serif}
.shell{max-width:940px;margin:auto;padding:0 44px}header{font:12px/1.5 system-ui,sans-serif;letter-spacing:.13em;text-transform:uppercase;padding:30px 0;border-bottom:1px solid var(--line)}
header .right{float:right}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}header a{color:var(--muted)}
article{padding:54px 0 72px}h1,h2,h3{font-family:system-ui,sans-serif;letter-spacing:-.035em;line-height:1.14}h1{font-size:clamp(36px,5vw,60px);font-weight:750;margin:0 0 24px}h2{font-size:30px;margin:58px 0 20px}h3{font-size:23px;margin:35px 0 16px}
article>p:first-of-type{font-size:23px;line-height:1.5;color:#5b5148;margin-bottom:36px}p{margin:20px 0}strong{font-weight:700}img{display:block;width:100%;height:auto;margin:32px auto 12px}
.table-wrap{overflow-x:auto;margin:28px -8px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font:13px/1.5 system-ui,sans-serif;font-variant-numeric:tabular-nums}th{text-align:left;font-weight:650;color:#5b5148;white-space:nowrap}td,th{padding:12px 9px;border-bottom:1px solid #e9e3d9}tr:last-child td{border:0}thead{background:#f1ece3}td:not(:first-child){white-space:nowrap}
blockquote{border-left:3px solid var(--accent);padding:1px 24px;margin:28px 0;color:#5b5148;background:#f5efe5}
code{font: .8em ui-monospace,Consolas,monospace;background:#f0ebe2;padding:.1em .3em;border-radius:3px}pre{overflow:auto;padding:22px;background:#2b2723;color:#f4ece0;line-height:1.6;border-radius:6px}pre code{background:none;padding:0}li{margin:8px 0}hr{border:0;border-top:1px solid var(--line);margin:44px 0}
footer{font:13px/1.6 system-ui,sans-serif;border-top:1px solid var(--line);padding:26px 0 40px;color:var(--muted)}.skip{position:absolute;left:-10000px}.skip:focus{left:20px;top:10px;background:white;padding:10px}
@media(max-width:640px){.shell{padding:0 22px}header .right{float:none;display:block;margin-top:10px}article{padding-top:32px}body{font-size:17px}h2{font-size:26px;margin-top:42px}article>p:first-of-type{font-size:21px}td,th{padding:10px 8px}blockquote{padding-left:16px}}
@media print{body{font-size:11pt}header .right{display:none}article{padding:20px 0}h1{font-size:30pt}h2{break-after:avoid}img,table{break-inside:avoid}.shell{max-width:none;padding:0}a{color:inherit}}
"""

class AssetAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="img":
            assert attrs.get("alt"),"Image missing alternative text"
        for key in ("src","href"):
            value=attrs.get(key,"")
            if value and not value.startswith(("https:","http:","#","mailto:")):
                self.assets.append(value)

def main():
    folder=Path("blog")
    md=(folder/"hybrid-search-reranking.md").read_text()
    title=next(line[2:] for line in md.splitlines() if line.startswith("# "))
    body=MarkdownIt("commonmark",{"html":False}).enable("table").render(md)
    body=re.sub(r"(<table>.*?</table>)",r'<div class="table-wrap">\1</div>',body,flags=re.S)
    body=body.replace("<h2>","<h2 tabindex='-1'>")
    document=(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        "<meta name='description' content='A measured BEIR comparison of hybrid search, cross-encoders, GPT-6 Luna, Jev, Decisions API, and selective reranking.'>"
        f"<style>{CSS}</style></head><body><a class='skip' href='#article'>Skip to article</a>"
        "<div class='shell'><header>Research notes / Search systems"
        "<span class='right'><a href='hybrid-search-reranking.md'>Markdown version</a> · October 21, 2026</span></header>"
        f"<main><article id='article'>{body}</article></main>"
        "<footer>Methods, measurements, and reproducible results · October 2026</footer>"
        "</div></body></html>"
    )
    audit=AssetAudit()
    audit.feed(document)
    for asset in audit.assets:
        assert (folder/asset).is_file(),f"Missing local asset: {asset}"
    (folder/"index.html").write_text(document)
    print(f"Rendered article; validated {len(audit.assets)} local links and images.")

if __name__=="__main__":
    main()
