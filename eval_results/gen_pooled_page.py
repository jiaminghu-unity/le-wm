#!/usr/bin/env python3
"""pooled2seed.json -> pooled2seed.html (双训练种子池化大汇总, cem/icem, T1-T5+总体, avg±SD)."""
import json, html

S = "/tmp/claude-1001/-/ee4386a4-8d1c-4244-b6fa-3843a95c13a0/scratchpad"
D = json.load(open(f"{S}/pooled2seed.json"))

TASKS = [
    ("pusht", "Push-T", "3072 + 3073"),
    ("reacher", "Reacher", "3072 + 3073"),
    ("cube", "Cube", "3072 + 3073"),
    ("tworoom", "Two-Room", "3072 + 3073"),
    ("pointmaze", "PointMaze", "3072 + 3074(DINO-WM 为 3072 + 3073,该模型无 3074 种子)"),
]
ARMS = ["LeWM 基线", "SCALE", "q-head 辅助", "DINO-WM"]
SOLS = [("cem", "CEM"), ("icem", "iCEM")]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def cell(v, sd, best):
    b = " b" if best else ""
    return f'<td class="num{b}">{v:.1f}<span class="sd">±{sd:.1f}</span></td>'


def task_table(key):
    rows = []
    for sol, sol_lbl in SOLS:
        # best per column within this solver
        vals = {a: D[f"{key}|{a}"][sol] for a in ARMS if f"{key}|{a}" in D}
        best_t = [max(v["t"][i] for v in vals.values()) for i in range(5)]
        best_o = max(v["o"] for v in vals.values())
        for i, a in enumerate(ARMS):
            v = vals.get(a)
            if v is None:
                continue
            first = f'<th class="sol" rowspan="{len(vals)}">{sol_lbl}</th>' if i == 0 else ""
            note = v.get("note")
            arm = html.escape(a) + (' <span class="flag">*</span>' if note else "")
            tds = "".join(cell(v["t"][j], v["ts"][j], v["t"][j] == best_t[j]) for j in range(5))
            tds += cell(v["o"], v["os"], v["o"] == best_o).replace('class="num', 'class="num ov')
            rows.append(f"<tr>{first}<td class='arm'>{arm}</td>{tds}</tr>")
    head = "".join(f"<th>{t}</th>" for t in TIERS) + '<th class="ovh">总体</th>'
    return (f'<div class="tw"><table><thead><tr><th></th><th>方法</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


sections = []
for key, lbl, seeds in TASKS:
    sections.append(
        f'<section><h2>{lbl}</h2><p class="meta">训练种子 {seeds} · 每格 n = 12 条独立测量'
        f'(2 训练种子 × 6 评测种子),每条为该层 100 回合成功率;总体列按每条种子的 500 回合池化。</p>'
        f"{task_table(key)}</section>"
    )

page = f"""<title>双种子池化总表</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#faf9f5;--panel:#ffffff;--ink:#20211d;--muted:#6f6d63;--rule:#e3e0d4;--accent:#2f6f5e;--best:#eaf3ee;--besttx:#1d5443}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#191a17;--panel:#20221e;--ink:#e8e6dd;--muted:#98958a;--rule:#34362f;--accent:#7dbfa9;--best:#243a31;--besttx:#a9dcc7;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#191a17;--panel:#20221e;--ink:#e8e6dd;--muted:#98958a;--rule:#34362f;--accent:#7dbfa9;--best:#243a31;--besttx:#a9dcc7;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;padding:0 16px;margin:0}}
main{{max-width:960px;margin:0 auto;padding-block:36px 64px;display:flex;flex-direction:column;gap:40px}}
h1{{font-family:'Source Serif 4',Georgia,serif;font-weight:700;font-size:30px;margin:0;text-wrap:balance}}
h2{{font-family:'Source Serif 4',Georgia,serif;font-weight:600;font-size:21px;margin:0 0 4px;color:var(--accent)}}
.sub{{color:var(--muted);font-size:13.5px;line-height:1.65;max-width:68ch;margin:10px 0 0}}
.meta{{color:var(--muted);font-size:12px;margin:0 0 10px}}
.tw{{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px}}
table{{border-collapse:collapse;width:100%;min-width:640px;font-size:13px}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule)}}
thead th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}}
tbody tr:last-child td,tbody tr:last-child th{{border-bottom:none}}
.sol{{text-align:left;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--accent);vertical-align:top;border-right:1px solid var(--rule)}}
.arm{{text-align:left;white-space:nowrap}}
.num{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
.num.b{{background:var(--best);color:var(--besttx);font-weight:500}}
.ov{{border-left:1px solid var(--rule);font-weight:500}}
.ovh{{border-left:1px solid var(--rule)}}
.sd{{font-size:9.5px;color:var(--muted);margin-left:2px}}
.num.b .sd{{color:var(--besttx);opacity:.75}}
.flag{{color:var(--accent);font-weight:600}}
.foot{{color:var(--muted);font-size:12px;line-height:1.7;border-top:1px solid var(--rule);padding-top:14px}}
</style>
<main>
<header>
<h1>双训练种子池化总表(CEM / iCEM)</h1>
<p class="sub">五个任务 × 四种方法 × CEM/iCEM,每格为两个训练种子 × 六个评测种子(s101–106)共 12 条独立成功率的均值 ± 样本标准差(ddof=1)。T1–T5 为规划预算档;总体为各评测种子 500 回合池化后再取均值。绿色为该列(同求解器内)最高。数据截至 2026-09-26。</p>
</header>
{''.join(sections)}
<p class="foot"><span class="flag">*</span> PointMaze 的 DINO-WM 无 3074 训练种子(该模型仅训过 3072/3073),此行改用 3072 + 3073 池化,其余 PointMaze 行均为 3072 + 3074。副本种子 cfg 映射:3073 = <code>{{stem}}r73</code>,3074 = <code>{{stem}}r74</code>;来源目录 final_eval_*(sd_mirror 全量镜像),不含 final_eval_l1 / FILTERED / broken_render / 旧 _mppi_ 协议文件。</p>
</main>
"""
open(f"{S}/pooled2seed.html", "w").write(page)
print("ok", len(page))
