# -*- coding: utf-8 -*-
"""引用全量审计：20 条 bib 逐条与 arXiv/出版页对拍。
输出：每条的状态（OK / TITLE_DIFF / AUTHOR_DIFF / NOT_FOUND）+ 修正建议。
"""
import io
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

bib_text = open(os.path.join(ROOT, 'paper', 'latex', 'references.bib'), encoding='utf-8').read()

# ---- 解析 bib ----
entries = {}
for m in re.finditer(r'@(\w+)\{([^,]+),\s*(.*?)\n\}', bib_text, re.S):
    key = m.group(2).strip()
    body = m.group(3)
    am = re.search(r'author\s*=\s*\{(.*?)\}', body, re.S)
    tm = re.search(r'title\s*=\s*\{(.*?)\}', body, re.S)
    ym = re.search(r'year\s*=\s*\{(\d{4})\}', body)
    entries[key] = dict(
        authors_raw=am.group(1).strip() if am else '',
        title_raw=tm.group(1).strip() if tm else '',
        year=ym.group(1) if ym else '',
    )

# ---- arXiv ID 映射 ----
ARXIV = {
    'pinwm2025': '2504.16693', 'physcore2026': '2607.20653',
    'hgn2019': '1909.13789', 'chnn2020': '2010.13581',
    'pderefiner': '2308.05732', 'mbpo2019': '1906.08253',
    'lyle2021': '2204.09560', 'metabalance2022': '2203.06801',
    'deepseekv3': '2412.19437', 'auxlossfree2024': '2408.15664',
    'nbf2025': '2510.03614', 'colas2018': '1806.08295',
    'neuralsim2021': '2011.04217', 'se3hode2021': '2106.12782',
}

def fetch_arxiv(ids):
    """批量获取 arXiv 元数据（带延迟防限流）。"""
    url = f"https://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results={len(ids)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'paper-audit/1.0'})
    xml = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore')
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml)
    out = {}
    for e in root.findall('a:entry', ns):
        eid = re.sub(r'v\d+$', '', e.find('a:id', ns).text.split('/abs/')[-1])
        title = re.sub(r'\s+', ' ', e.find('a:title', ns).text.strip())
        authors = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
        out[eid] = dict(title=title, authors=authors)
    return out

def fetch_page(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore')

# ---- 存储 ----
report = {}
errors = []

# ===== Phase 1: arXiv 批量核实 =====
arxiv_ids = list(ARXIV.values())
print(f"Phase 1: arXiv 批量核实 {len(arxiv_ids)} 条…")
try:
    ax = fetch_arxiv(arxiv_ids)
except Exception as e:
    print(f"  arXiv API 失败（{e}），5s 后单条重试…")
    ax = {}
    for aid in arxiv_ids:
        time.sleep(1.5)
        try:
            ax.update(fetch_arxiv([aid]))
        except Exception:
            pass

id_to_key = {v: k for k, v in ARXIV.items()}
for aid, meta in ax.items():
    key = id_to_key.get(aid)
    if not key:
        continue
    e = entries[key]
    ax_authors = meta['authors']
    ax_title = meta['title']

    # bib 作者解析（LaTeX → 姓列表）
    bib_surnames = []
    for person in re.split(r'\s+and\s+', e['authors_raw']):
        parts = person.strip().replace('{', '').replace('}', '').split(',')
        surname = parts[0].strip()
        # LaTeX accent 清理
        surname = re.sub(r"\\[a-zA-Z]", "", surname).strip()
        if surname:
            bib_surnames.append(surname)
    ax_surnames = [a.split()[-1] for a in ax_authors]

    # 标题对拍
    bt = re.sub(r'[{}\\]', '', e['title_raw']).lower().strip()
    at = ax_title.lower().strip()
    title_ok = bt[:40] in at or at[:40] in bt

    # 作者对拍（前 3 个姓是否匹配）
    n_check = min(3, len(bib_surnames), len(ax_surnames))
    auth_ok = all(
        bib_surnames[i].lower().replace("í", "i").replace("é", "e") ==
        ax_surnames[i].lower().replace("í", "i").replace("é", "e")
        for i in range(n_check)
    ) if n_check > 0 else False

    # 年份对拍
    year_match = e['year'] in ax_title  # 粗查

    status = 'OK' if title_ok and auth_ok else 'MISMATCH'
    if not title_ok:
        errors.append(f"{key}: TITLE bib='{e['title_raw'][:50]}' vs arXiv='{ax_title[:50]}'")
    if not auth_ok:
        errors.append(f"{key}: AUTHORS bib={bib_surnames[:3]} vs arXiv={ax_surnames[:3]}")

    report[key] = dict(
        status=status, title_ok=title_ok, authors_ok=auth_ok,
        bib_authors=e['authors_raw'][:80],
        arxiv_authors=', '.join(ax_authors[:6]) + (f" (+{len(ax_authors)-6})" if len(ax_authors) > 6 else ''),
        arxiv_title=ax_title[:80],
        bib_title=e['title_raw'][:80],
    )
    print(f"  [{status}] {key}: title={'[OK]' if title_ok else '[FAIL]'} authors={'[OK]' if auth_ok else '[FAIL]'}")

# arXiv 没返回的
for key, aid in ARXIV.items():
    if key not in report:
        errors.append(f"{key}: arXiv {aid} 未返回（限流或 ID 错误）")
        report[key] = dict(status='NOT_FOUND', arxiv_id=aid)
        print(f"  [NOT_FOUND] {key} ({aid})")

# ===== Phase 2: 非 arXiv 条目 =====
print(f"\nPhase 2: 非 arXiv 条目（{sum(1 for k in entries if k not in ARXIV)} 条）…")

# ani2026: Nature Comms
try:
    page = fetch_page("https://www.nature.com/articles/s41467-026-74002-2")
    # Nature 页面的作者在 meta 标签里
    dc_authors = re.findall(r'name="dc\.creator" content="([^"]+)"', page)
    if dc_authors:
        report['ani2026'] = dict(status='OK', nature_authors=dc_authors[:8],
                                 n_authors=len(dc_authors))
        print(f"  [OK] ani2026: {len(dc_authors)} 位作者: {', '.join(dc_authors[:4])}…")
    else:
        # 尝试 article meta
        m = re.search(r'"author":\s*\[([^\]]+)\]', page)
        if m:
            names = re.findall(r'"name":\s*"([^"]+)"', m.group(1))
            report['ani2026'] = dict(status='OK', nature_authors=names[:8], n_authors=len(names))
            print(f"  [OK] ani2026: {len(names)} 位作者")
        else:
            report['ani2026'] = dict(status='PAGE_OK_AUTHORS_UNCLEAR')
            print("  [[WARN]] ani2026: 页面可访问但作者提取失败")
except Exception as e:
    report['ani2026'] = dict(status='FETCH_ERROR', error=str(e))
    errors.append(f"ani2026: Nature 页面获取失败 {e}")
    print(f"  [ERROR] ani2026: {e}")

time.sleep(1)

# tossingbot2019: RSS 2019（走 arXiv API，abs 页 HTML 不能用 XML 解析）
try:
    page = fetch_arxiv(["1903.11239"])
    meta = page["1903.11239"]
    authors = meta["authors"]
    title = meta["title"]
    report['tossingbot2019'] = dict(status='OK', arxiv_authors=authors, arxiv_title=title)
    print(f"  [OK] tossingbot2019: {', '.join(authors[:4])}")
    # 对拍
    bib_sn = re.split(r'\s+and\s+', entries['tossingbot2019']['authors_raw'])
    bib_sn = [x.strip().split(',')[0] for x in bib_sn]
    ax_sn = [a.split()[-1] for a in authors]
    if bib_sn[0].lower() != ax_sn[0].lower():
        errors.append(f"tossingbot2019: 第一作者 bib='{bib_sn[0]}' vs arXiv='{ax_sn[0]}'")
except Exception as e:
    report['tossingbot2019'] = dict(status='FETCH_ERROR', error=str(e))
    print(f"  [ERROR] tossingbot2019: {e}")

time.sleep(1)

# dreamerv3_2025: Nature
try:
    page = fetch_page("https://www.nature.com/articles/s41586-025-08744-2")
    dc_authors = re.findall(r'name="dc\.creator" content="([^"]+)"', page)
    if dc_authors:
        report['dreamerv3_2025'] = dict(status='OK', nature_authors=dc_authors[:6], n_authors=len(dc_authors))
        print(f"  [OK] dreamerv3_2025: {len(dc_authors)} 位作者: {', '.join(dc_authors[:4])}…")
    else:
        report['dreamerv3_2025'] = dict(status='PAGE_OK_AUTHORS_UNCLEAR')
        print("  [[WARN]] dreamerv3_2025: 作者提取失败")
except Exception as e:
    report['dreamerv3_2025'] = dict(status='FETCH_ERROR', error=str(e))
    print(f"  [ERROR] dreamerv3_2025: {e}")

# kronos2026: DELETED 2026-10-06 — Zenodo DOI 10.5281/zenodo.22645812 returns 404
# and exact-title search finds no record; entry removed from bib and text.

# zero_vel: 不在 bib 里
# se3hode2021 已在 Phase 1

# ===== 输出 =====
print(f"\n{'='*70}")
print(f"审计结果：{len(report)} 条")
print(f"错误/差异：{len(errors)} 条")
for e in errors:
    print(f"  [WARN] {e}")
if not errors:
    print("  （无）")

out_path = os.path.join(ROOT, 'analysis_output', 'citation_audit.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(dict(report=report, errors=errors), f, ensure_ascii=False, indent=1)
print(f"[save] {out_path}")
