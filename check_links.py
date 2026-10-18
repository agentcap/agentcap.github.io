from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
root = Path(__file__).parent
class Page(HTMLParser):
 def __init__(self, text):
  super().__init__(); self.refs=[]; self.ids=set(); self.feed(text)
 def handle_starttag(self, tag, attrs):
  a=dict(attrs)
  if 'id' in a: self.ids.add(a['id'])
  for key in ('src','href'):
   if key in a: self.refs.append(a[key])
errors=[]
for f in root.rglob('*.html'):
 page=Page(f.read_text(encoding='utf-8-sig'))
 for ref in page.refs:
  u=urlsplit(ref)
  if u.scheme or u.netloc: continue
  target=(f.parent/unquote(u.path)).resolve() if u.path else f
  if target.is_dir(): target=target/'index.html'
  if not target.exists(): errors.append(f'{f.name}: missing {ref}')
  elif u.fragment and target.suffix=='.html' and u.fragment not in Page(target.read_text(encoding='utf-8-sig')).ids: errors.append(f'{f.name}: missing anchor {ref}')
print('\n'.join(errors) if errors else 'All local assets, page links, and fragment targets passed.')
raise SystemExit(bool(errors))
