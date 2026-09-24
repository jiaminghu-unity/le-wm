#!/usr/bin/env python3
"""消融表全景页生成器 v2:所有 SR 格分 T1-T5(总 SR 加粗 + 五档小字)。"""
import json, os, re
S=os.path.dirname(os.path.abspath(__file__))
def load(n):
    p=f'{S}/{n}'
    return json.load(open(p)) if os.path.exists(p) else {}
a2sr=load('a2_sr.json'); a45sr=load('a45_sr.json'); a3sr=load('a3_sr.json')

def tiers(t):
    return '<div class="tiers">'+' '.join('–' if x is None else f'{x:.0f}' for x in t)+'</div>'
def one(v,state='r'):
    """v = {"o":..,"t":[..]} 或 None"""
    if v: return f'<td class="have"><b>{v["o"]}</b>{tiers(v["t"])}</td>'
    return {'h':'<td class="have"></td>','r':'<td class="run"></td>','q':'<td></td>'}[state]
def pair(v,state='r'):
    """v = {"cem":..,"icem":..} -> 两行"""
    if v:
        c,i=v['cem'],v['icem']
        return (f'<td class="have"><span class="sol">c</span> <b>{c["o"]}</b>{tiers(c["t"])}'
                f'<span class="sol">i</span> <b>{i["o"]}</b>{tiers(i["t"])}</td>')
    return {'h':'<td class="have"></td>','r':'<td class="run"></td>','q':'<td></td>'}[state]
def row(th,c): return f'<tr><th>{th}</th>{"".join(c)}</tr>'
def table(head,body): return f'<div class="tblw"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'

# A1(计时,无分档)
a1_head='<tr><th>臂</th><td>训练 it/s</td><td>训练总时长(10 epoch,cube)</td><td>规划 ms/plan</td><td>s/episode</td><td>规划相对基线</td></tr>'
a1=''.join('<tr><th>%s</th>'%r[0]+''.join(f'<td>{c}</td>' for c in r[1:])+'</tr>' for r in [
 ('LeWM 基线','4.35','8.2 h','485–509','0.8–1.0','1.00×'),
 ('LeWM + SCALE','3.85(+13%)','9.2 h','485–507','0.8–1.6','1.00×'),
 ('LeWM + q-head 辅助','4.20(+4%)','8.5 h','480–519','0.9–2.9','1.00×'),
 ('DINO-WM','8.95(编码器冻结)','4.0 h','24,070–24,295','32–35','≈48×')])

# A2
Ts=[8,32,64,128,256,512]
CFG={'pusht':('c1','c3_l01'),'reacher':('r1','r2_l015'),'cube':('k1','k2'),
     'tworoom':('t1','t2_l01'),'pointmaze':('p1','p2_l01')}
PAPER={'pusht':64,'reacher':32,'cube':32,'tworoom':64,'pointmaze':256}
a2_head='<tr><th>任务 · 臂</th>'+''.join(f'<td>T={t}</td>' for t in Ts)+'</tr>'
a2=''
for task in ['pusht','reacher','cube','tworoom','pointmaze']:
    for arm,cfg in zip(['基线','SCALE'],CFG[task]):
        cells=[one(a2sr.get(f'{task}|{cfg}|{t}')) for t in Ts]
        name=f'{task} · {arm}'+(f'(论文 T={PAPER[task]})' if arm=='基线' else '')
        a2+=row(name,cells)
n_a2=len(a2sr)

# A3
a3_head='<tr><th>任务 · SCALE 的 q</th><td>q 维数</td><td>cem</td><td>icem</td><td>Δ vs 基线(cem/icem,总)</td></tr>'
A3ROWS=[('pusht · 论文 q:无速度子集','6','pusht|paper'),
 ('pusht · 全集:native(含速度)','8','pusht|full'),
 ('reacher · 论文 q:仅关节','4','reacher|paper'),
 ('reacher · 全集:native(关节+指尖+速度)','8','reacher|full'),
 ('cube · 论文 q:end-effector','9','cube|paper'),
 ('cube · 全集:full-config','22','cube|full'),
 ('pointmaze · 论文 q:位置','2','pointmaze|paper'),
 ('pointmaze · 全集:native(位置+速度)','4','pointmaze|full')]
a3=''
for th,dim,key in A3ROWS:
    v=a3sr.get(key)
    if v:
        a3+=row(th,[f'<td>{dim}</td>',one(v['cem']),one(v['icem']),f'<td class="have">{v["delta"]}</td>'])
    else:
        a3+=row(th,[f'<td>{dim}</td>','<td class="run"></td>','<td class="run"></td>','<td class="run"></td>'])
a3+='<tr><th>tworoom · 论文 q = 全集(数据无速度列)</th><td colspan="4" style="color:var(--muted)">不适用——论文所用 q 已是可得全集</td></tr>'

# A4/A5
def wrow(kind,task,dagger,grid):
    def get(w):
        key=f'base|{task}|0' if w=='0' else f'{kind}|{task}|{w}'
        return a45sr.get(key)
    return row(f'{task}(†{dagger})',[pair(get(w)) for w in grid])
G4=['0','0.03','0.1','0.15','0.3','1.0']; G5=['0','0.1','0.3','0.4','1.0']
a4_head='<tr><th>任务(论文值 †)</th>'+''.join(f'<td>{w if w!="0" else "0(基线)"}</td>' for w in G4)+'</tr>'
a4=''.join([wrow('obj','pusht','0.1',G4),wrow('obj','reacher','0.15',G4),wrow('obj','cube','0.1',G4),
            wrow('obj','tworoom','0.1',G4),wrow('obj','pointmaze','0.1',G4)])
a5_head='<tr><th>任务(论文值 †)</th>'+''.join(f'<td>{w if w!="0" else "0(基线)"}</td>' for w in G5)+'</tr>'
a5=''.join([wrow('aux','pusht','0.3',G5),wrow('aux','reacher','0.4',G5),wrow('aux','cube','0.1',G5),
            wrow('aux','tworoom','0.1',G5),wrow('aux','pointmaze','0.1',G5)])

html=f'''<title>消融表全景</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{{--bg:#F9F9F7;--ink:#191C20;--muted:#5D6570;--accent:#1D6E5E;--card:#FFF;--line:#E2E4E0;--code:#EFF1ED;--have:#DCEFE7;--run:#FBF3D8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181A;--ink:#E5E7E4;--muted:#98A19C;--accent:#5DBBA6;--card:#1C2022;--line:#2B302E;--code:#21262A;--have:#1E3A31;--run:#3B331B}}}}
:root[data-theme="dark"]{{--bg:#15181A;--ink:#E5E7E4;--muted:#98A19C;--accent:#5DBBA6;--card:#1C2022;--line:#2B302E;--code:#21262A;--have:#1E3A31;--run:#3B331B}}
body{{background:var(--bg);color:var(--ink);font:15px/1.7 "Noto Sans SC",-apple-system,sans-serif;margin:0}}
.wrap{{max-width:1080px;margin:0 auto;padding:38px 22px 80px}}
h1{{font-size:24px;margin:4px 0}}
.sub{{color:var(--muted);font-size:13.5px;margin:0 0 10px}}
h2{{font-size:18px;margin:34px 0 4px;padding-top:14px;border-top:1px solid var(--line)}}
h2 .tag{{color:var(--accent);font-family:"JetBrains Mono",monospace;font-size:13px;margin-right:8px}}
p{{margin:6px 0;max-width:76ch}} .mini{{color:var(--muted);font-size:13px}}
.tblw{{overflow-x:auto;margin:10px 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;width:fit-content;max-width:100%}}
table{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid var(--line);padding:4px 9px;white-space:nowrap;vertical-align:top}}
th{{background:var(--code);font-weight:500;font-family:"JetBrains Mono",monospace;font-size:11.5px;text-align:left}}
td{{min-width:68px;text-align:center}}
td.have{{background:var(--have)}} td.run{{background:var(--run)}}
b{{font-size:12.5px}}
.tiers{{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--muted);letter-spacing:.4px;line-height:1.5}}
.sol{{font-family:"JetBrains Mono",monospace;font-size:9.5px;color:var(--accent);margin-right:2px}}
.legend{{display:flex;gap:14px;margin:10px 0;font-size:13px;color:var(--muted);flex-wrap:wrap}}
.sw{{display:inline-block;width:14px;height:14px;border:1px solid var(--line);border-radius:3px;vertical-align:-2px;margin-right:5px}}
</style>
<div class="wrap">
<h1>SCALE 论文消融表 · 全景</h1>
<p class="sub">seed 3072 · 每格:<b>总 SR</b> + 小字 T1 T2 T3 T4 T5 分档;A4/A5 里 <span class="sol">c</span>=cem、<span class="sol">i</span>=icem · mppi 格 = 5 个 held-out 种子(102–106)× 100 回合,cem/icem 格 = 6 种子 × 100 回合 · 2026-09-23</p>
<div class="legend"><span><span class="sw" style="background:var(--have)"></span>已有数值</span><span><span class="sw" style="background:var(--run)"></span>训练/评测在跑</span><span><span class="sw"></span>排队中</span></div>

<h2><span class="tag">A1</span>计算成本:训练与规划(已完成)</h2>
<p class="mini">训练 = 同一张 A100 上 300 步实测 it/s(cube 配方,12,796 步/epoch × 10 外推);规划 = 全部评测 CSV 聚合(每格 ≥3,600 回合),cem/icem/mppi 一致。结论:SCALE 以 +13% 训练成本换 SR 提升、推理零开销;DINO-WM 训练便宜(冻结编码器)但规划贵 48 倍。</p>
{table(a1_head,a1)}

<h2><span class="tag">A2</span>mppi 温度选择(已完成,60/60)</h2>
<p class="mini">每任务 × {{基线, SCALE}} × T∈{{8,32,64,128,256,512}};论文所用 T 在行名标注。结论:pointmaze/pusht/cube 平台宽且两臂峰位一致;tworoom SCALE 峰在 T=128;reacher 两臂峰均在 64–128(论文 T=32 非峰值,主表 cem/icem 结果不受影响)。全表 60 格中,每个任务在每个已测温度下 SCALE 均压基线——增益与温度选择无关。</p>
{table(a2_head,a2)}

<h2><span class="tag">A3</span>SCALE 的 q 选择:论文所用 q vs 全集 q</h2>
<p class="mini">严格两点对比:论文 SCALE 实际所用 q vs 该任务可得的全集/native q(仅在两者不同的任务上);Δ = 相对基线总 SR。结论(已完成,8/8 行):四个任务的论文 q 与全集 q 全部持平于种子噪声内(总 SR 差 ≤1.2pt,pointmaze icem +2.6 为最大单格差,cube icem 反向)——SCALE 对 q 选择不敏感,全集(含速度)在 L_obj 中无毒;tworoom 论文 q 即全集,不适用。</p>
{table(a3_head,a3)}

<h2><span class="tag">A4</span>L_obj 权重敏感性(已完成)</h2>
<p class="mini">全部五任务 × 统一网格 λ∈{{0, 0.03, 0.1, 0.15, 0.3, 1.0}};论文取值 † 标注。结论(50/50 格):tworoom 单调上升至 1.0(92.8/95.5,较基线 +15/+12.6,全场最大增益)、pusht 单调缓升、cube 与 reacher 全平台(reacher 0.03–1.0 五格都在 71–72/77–81);唯 pointmaze 在 1.0 劣化且单训练方差大(±6pt,慎读)。论文 0.1 处处安全但普遍偏保守。</p>
{table(a4_head,a4)}

<h2><span class="tag">A5</span>q-head 辅助权重敏感性(已完成)</h2>
<p class="mini">与 A4 同构:λ∈{{0, 0.1, 0.3, 0.4, 1.0}};论文取值 † 标注。</p>
{table(a5_head,a5)}

<p class="mini">五张消融表全部完成(2026-09-24);正式论文表由本页导出(去掉状态着色)。底账:final_eval*/、final_eval_mppi_t/、eval/timecost_3072.txt。</p>
</div>'''
open(f'{S}/ablation_tables.html','w').write(html)
tags=re.findall(r'tag">(A\d)</span>', html)
assert tags==['A1','A2','A3','A4','A5'], tags
print("OK",tags,"a2:",n_a2,"a45:",len(a45sr),"a3:",len(a3sr))
