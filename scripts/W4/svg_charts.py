# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
svg_charts.py - 仪表盘内联 SVG 图渲染
说明：服务端生成静态 SVG，数据点带 data-tip 属性，前端统一用一个悬浮层读数；
      不使用任何外部图表库，交付文件保持离线单文件。
"""
from __future__ import annotations

INK, SUB, GRID, BAND = "#1b1f24", "#5b6470", "#e3e7ec", "#f7f9fb"
BLUE, RED, GREEN, ORANGE = "#1f5fbf", "#c0392b", "#1e8449", "#c77b1a"
PALETTE = [BLUE, RED, GREEN, ORANGE, "#6c5ce7", "#0e8f8f"]


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def num(v, nd: int = 0) -> str:
    return f"{v:,.{nd}f}"


def pct(v, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}%"


def tip(html: str) -> str:
    return f'data-tip="{esc(html)}"'


def _nice_top(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


class Canvas:
    def __init__(self, w: int, h: int, label: str = ""):
        self.w, self.h, self.label = w, h, label
        self.parts: list[str] = []

    def add(self, s: str) -> "Canvas":
        self.parts.append(s)
        return self

    def rect(self, x, y, w, h, fill, **kw):
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.2f}" '
                        f'height="{max(h, 0):.2f}" fill="{fill}"{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None, **kw):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)
        return self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{stroke}" stroke-width="{width}"{d}{extra}/>')

    def text(self, x, y, s, size=11, fill=SUB, anchor="start", weight=None, transform=None):
        wgt = f' font-weight="{weight}"' if weight else ""
        tr = f' transform="{transform}"' if transform else ""
        return self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                        f'text-anchor="{anchor}"{wgt}{tr}>{esc(s)}</text>')

    def poly(self, pts, stroke=BLUE, width=2.4, dash=None, fill="none"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return self.add(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" '
                        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def circle(self, x, y, r=3, fill=BLUE, stroke=None, width=1.6, **kw):
        s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)
        return self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{fill}"{s}{extra}/>')

    def svg(self) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" preserveAspectRatio="xMidYMid meet" '
                f'role="img" aria-label="{esc(self.label)}">' + "".join(self.parts) + "</svg>")


def _lum(rgb):
    def f(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_on(hexcol, alpha):
    """底色 = hexcol 按 alpha 叠在白底上；返回 (最佳对比度, 主字色, 次字色)。"""
    h = hexcol.lstrip("#")
    base = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    rgb = tuple(255 * (1 - alpha) + c * alpha for c in base)
    lum = _lum(rgb)
    white = 1.05 / (lum + 0.05)
    ink = (lum + 0.05) / 0.0635
    if white > ink:
        return white, "#ffffff", "#eef3fa"
    return ink, INK, SUB


def text_on(hexcol, alpha):
    """文字配色 (主字, 次字)：深底用白字，浅底用深灰字。"""
    return contrast_on(hexcol, alpha)[1:]


def readable_alpha(hexcol, alpha, floor=4.5, step=0.04, tries=4):
    """对比度不够时微调填充深浅（±0.04 一档，最多 4 档），保证字看得清。"""
    if contrast_on(hexcol, alpha)[0] >= floor:
        return alpha
    for d in range(1, tries + 1):
        for cand in (alpha + step * d, alpha - step * d):
            cand = round(cand, 3)
            if 0 <= cand <= 1 and contrast_on(hexcol, cand)[0] >= floor:
                return cand
    return alpha


# ---------------------------------------------------------------- 柱状图
def chart_bars(items, *, w=560, h=300, color=BLUE, ylab="", val_fmt=lambda v: num(v),
               y_from_zero=True, x_labels=None, bar_color=None) -> str:
    """items: [(label, value, tip?)]"""
    l, r, t, b = 56, 16, 26, 42
    top = _nice_top(max(v for _, v, *_ in items) * 1.15)
    c = Canvas(w, h, ylab or "柱状图")
    cw, ch = w - l - r, h - t - b
    for i in range(5):
        y = t + ch * i / 4
        c.line(l, y, l + cw, y, GRID if i else "#cdd5de")
        c.text(l - 8, y + 4, num(top * (4 - i) / 4), anchor="end")
    if ylab:
        c.text(14, t - 8, ylab, fill=SUB)
    step = cw / len(items)
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        tp = it[2] if len(it) > 2 else f"{label}：{val_fmt(value)}"
        bh = ch * value / top
        x = l + step * i + step * 0.18
        fill = bar_color(i) if bar_color else color
        c.rect(x, t + ch - bh, step * 0.64, bh, fill, rx=2, **{"data-tip": esc(tp)})
        c.text(x + step * 0.32, t + ch - bh - 6, val_fmt(value), anchor="middle", fill=INK)
        c.text(x + step * 0.32, t + ch + 16, label, anchor="middle")
    c.line(l, t + ch, l + cw, t + ch, "#cdd5de")
    return c.svg()


# ---------------------------------------------------------------- 横条图
def chart_hbars(items, *, w=560, h=430, color=BLUE, val_fmt=lambda v: f"{v:.3f}",
                xlab="") -> str:
    """items: [(label, value, tip?)]，按给定顺序自上而下"""
    l, r, t, b = 132, 66, 22, 34
    top = max(v for _, v, *_ in items) * 1.06
    c = Canvas(w, h, xlab or "横条图")
    cw, ch = w - l - r, h - t - b
    step = ch / len(items)
    for i in range(5):
        x = l + cw * i / 4
        c.line(x, t, x, t + ch, GRID)
        c.text(x, t + ch + 16, f"{top * i / 4:.2f}", anchor="middle")
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        tp = it[2] if len(it) > 2 else f"{label}：{val_fmt(value)}"
        y = t + step * i + step * 0.16
        c.rect(l, y, cw * value / top, step * 0.68, color, rx=2, **{"data-tip": esc(tp)})
        c.text(l - 8, y + step * 0.5, label, anchor="end", fill=INK)
        c.text(l + cw * value / top + 6, y + step * 0.52, val_fmt(value), fill=SUB)
    if xlab:
        c.text(l + cw / 2, t - 6, xlab, anchor="middle")
    return c.svg()


# ---------------------------------------------------------------- 直方图
def chart_hist(bins, *, w=300, h=200, color=BLUE, x_fmt=lambda v: num(v),
               ylab="笔数", unit="笔", tip_prefix="", small=True) -> str:
    l, r, t, b = 44, 8, 14, 32
    c = Canvas(w, h, "分布直方图")
    cw, ch = w - l - r, h - t - b
    top = _nice_top(max(cn for _, _, cn in bins) * 1.05)
    total = sum(cn for _, _, cn in bins) or 1
    x0, x1 = bins[0][0], bins[-1][1]
    for i in range(3):
        y = t + ch * i / 2
        c.line(l, y, l + cw, y, GRID)
        c.text(l - 6, y + 4, num(top * (2 - i) / 2), anchor="end", size=10)
    for lo, hi, cn in bins:
        if cn <= 0:
            continue
        bx = l + cw * (lo - x0) / (x1 - x0)
        bw = max(cw * (hi - lo) / (x1 - x0), 0.7)
        bh = ch * cn / top
        c.rect(bx, t + ch - bh, bw, bh, color,
               **{"data-tip": esc(f"{tip_prefix}{x_fmt(lo)} ~ {x_fmt(hi)}：{num(cn)} {unit}"
                                  f"（{cn / total * 100:.1f}%）")})
    c.line(l, t + ch, l + cw, t + ch, "#cdd5de")
    for i in range(3):
        x = l + cw * i / 2
        c.text(x, t + ch + 15, x_fmt(x0 + (x1 - x0) * i / 2), anchor="middle", size=10)
    return c.svg()


def chart_minis(panels, *, cols=3) -> str:
    out = [f'<div class="minis" style="--cols:{cols}">']
    for p in panels:
        out.append('<div class="mini"><div class="mt">' + esc(p["title"]) + "</div>"
                   + p["svg"] + "</div>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- 箱线图
def chart_box(panels, *, cols=2, cell_w=330, cell_h=210) -> str:
    """panels: [{'title','unit','log':bool,'groups':[('正常',stats),('违约',stats)]}]"""
    import math
    out = [f'<div class="minis" style="--cols:{cols}">']
    for p in panels:
        log = p.get("log", False)
        vals = []
        for _, s in p["groups"]:
            vals += [s["min"], s["max"]]
        lo, hi = min(vals), max(vals)
        if log:
            lo, hi = math.log10(max(lo, 1)), math.log10(max(hi, 1))
        pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - pad, hi + pad

        def y_of(v):
            if log:
                v = math.log10(max(v, 1))
            return t + ch * (hi - v) / (hi - lo)

        w, h = cell_w, cell_h
        l, r, t, b = 54, 12, 16, 30
        c = Canvas(w, h, p["title"] + " 箱线图")
        cw, ch = w - l - r, h - t - b
        unit = p.get("unit", "")

        def fmtv(v):
            return f"{num(v)}" if abs(v) >= 100 else f"{v:g}"

        for i in range(4):
            v = hi - (hi - lo) * i / 3
            real = 10 ** v if log else v
            y = t + ch * i / 3
            c.line(l, y, l + cw, y, GRID)
            c.text(l - 6, y + 4, fmtv(real), anchor="end", size=10)
        for gi, (name, s) in enumerate(p["groups"]):
            cx = l + cw * (gi + 0.5) / len(p["groups"])
            bw = cw / len(p["groups"]) * 0.36
            col = BLUE if gi == 0 else RED
            c.line(cx, y_of(s["min"]), cx, y_of(s["max"]), col, 1.2)
            c.line(cx - bw * 0.3, y_of(s["min"]), cx + bw * 0.3, y_of(s["min"]), col, 1.2)
            c.line(cx - bw * 0.3, y_of(s["max"]), cx + bw * 0.3, y_of(s["max"]), col, 1.2)
            box_tip = (f"{name}｜中位数 {fmtv(s['med'])}{unit}｜四分位 "
                       f"{fmtv(s['q1'])}–{fmtv(s['q3'])}{unit}｜均值 {fmtv(s['mean'])}{unit}"
                       f"｜样本 {num(s['n'])} 笔")
            c.rect(cx - bw / 2, y_of(s["q3"]), bw, max(y_of(s["q1"]) - y_of(s["q3"]), 1),
                   col, rx=2, opacity=0.85, **{"data-tip": esc(box_tip)})
            c.line(cx - bw / 2, y_of(s["med"]), cx + bw / 2, y_of(s["med"]), "#ffffff", 2)
            c.text(cx, t + ch + 18, name, anchor="middle")
        c.line(l, t + ch, l + cw, t + ch, "#cdd5de")
        if p.get("note"):
            c.text(l - 46, t - 4, p["note"], size=10)
        out.append('<div class="mini"><div class="mt">' + esc(p["title"]) + "</div>"
                   + c.svg() + "</div>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- 热力图
def chart_heatmap(labels, matrix, *, cell=62, cell_h=None, vmin=None, vmax=None,
                  fmt=lambda v: f"{v:.2f}", cell_tip=None, row_labels=None,
                  color_pos="#c0392b", color_neg="#1f5fbf") -> str:
    n, m = len(matrix), len(matrix[0])
    cell_h = cell if cell_h is None else cell_h
    cols = row_labels or labels
    l, t = 108, 62
    w, h = l + cell * m + 22, t + cell_h * n + 16
    vmin = min(min(r) for r in matrix) if vmin is None else vmin
    vmax = max(max(r) for r in matrix) if vmax is None else vmax
    span = max(abs(vmin), abs(vmax)) or 1
    c = Canvas(w, h, "相关性热力图")
    for j, name in enumerate(labels):
        c.text(l + cell * (j + 0.5), t - 10, name, anchor="middle", size=10)
    for i, name in enumerate(cols):
        c.text(l - 8, t + cell_h * (i + 0.5) + 4, name, anchor="end", size=10, fill=INK)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            k = v / span
            base = color_pos if k >= 0 else color_neg
            opacity = readable_alpha(base, round(min(abs(k) ** 0.75 + 0.06, 1), 3))
            html = (cell_tip(i, j, v) if cell_tip else f"{cols[i]} × {labels[j]}：{fmt(v)}")
            c.rect(l + cell * j, t + cell_h * i, cell - 1.5, cell_h - 1.5, base, opacity=round(opacity, 3),
                   rx=2, **{"data-tip": esc(html)})
            c.text(l + cell * (j + 0.5), t + cell_h * (i + 0.5) + 4, fmt(v), anchor="middle",
                   size=10, fill=text_on(base, round(opacity, 3))[0])
    return c.svg()


# ---------------------------------------------------------------- 折线图
def chart_lines(series, *, w=560, h=300, xlab="", ylab="", x_fmt=lambda v: f"{v:g}",
                y_fmt=lambda v: f"{v:.3f}", y_ticks=5, legend=True, markers=True,
                marker_every=1, hit=True, x_is_index=False, y_pad=0.0006) -> str:
    """series: [{'name','color','pts':[(x,y)],'dash':None,'tip':fn(x,y)->str}]"""
    l, r, t, b = 62, 16, 30 if legend else 22, 40
    xs = [p[0] for s in series for p in s["pts"]]
    ys = [p[1] for s in series for p in s["pts"]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys) - y_pad, max(ys) + y_pad
    c = Canvas(w, h, ylab or "折线图")
    cw, ch = w - l - r, h - t - b
    X = lambda v: l + cw * (v - x0) / ((x1 - x0) or 1)
    Y = lambda v: t + ch * (y1 - v) / ((y1 - y0) or 1)
    for i in range(y_ticks):
        v = y1 - (y1 - y0) * i / (y_ticks - 1)
        y = t + ch * i / (y_ticks - 1)
        c.line(l, y, l + cw, y, GRID)
        c.text(l - 8, y + 4, y_fmt(v), anchor="end", size=10)
    ticks = 5
    for i in range(ticks + 1):
        v = x0 + (x1 - x0) * i / ticks
        c.text(X(v), t + ch + 17, x_fmt(v), anchor="middle", size=10)
    if legend:
        x = l
        for s in series:
            c.rect(x, t - 16, 14, 3, s.get("color", BLUE))
            c.text(x + 19, t - 11, s["name"], size=11)
            x += 19 + 9 * len(s["name"]) + 22
    for s in series:
        col = s.get("color", BLUE)
        pts = [(X(px), Y(py)) for px, py in s["pts"]]
        c.poly(pts, stroke=col, width=2.2, dash=s.get("dash"))
        step = max(1, int(marker_every))
        for i, ((px, py), (rx, ry)) in enumerate(zip(pts, s["pts"])):
            tp = s["tip"](rx, ry) if s.get("tip") else f"{s['name']}：{y_fmt(ry)}"
            if markers and (i % step == 0 or i == len(pts) - 1):
                c.circle(px, py, 3.4, "#ffffff", stroke=col, width=2,
                         **{"data-tip": esc(tp)})
            elif hit:
                c.circle(px, py, 8, "rgba(0,0,0,0)",
                         **{"data-tip": esc(tp), "pointer-events": "all"})
    c.line(l, t + ch, l + cw, t + ch, "#cdd5de")
    if xlab:
        c.text(l + cw / 2, h - 8, xlab, anchor="middle")
    if ylab:
        c.text(13, t + ch / 2, ylab, anchor="middle",
               transform=f"rotate(-90 13 {t + ch / 2:.1f})")
    return c.svg()


# ---------------------------------------------------------------- 混淆矩阵
def chart_confusion(cm, *, w=560, h=250) -> str:
    cells = [("真违约 · 被拒（TP）", cm["tp"], GREEN), ("正常人 · 被拒（FP）", cm["fp"], ORANGE),
             ("真违约 · 放行（FN）", cm["fn"], RED), ("正常人 · 放行（TN）", cm["tn"], BLUE)]
    l, t, cw, chh = 96, 52, (w - 96 - 20) / 2, (h - 52 - 20) / 2
    c = Canvas(w, h, "混淆矩阵")
    c.text(l - 8, t + chh / 2 + 4, "模型：拒贷", anchor="end", fill=INK)
    c.text(l - 8, t + chh + chh / 2 + 4, "模型：放行", anchor="end", fill=INK)
    c.text(l + cw / 2, t - 30, "实际违约", anchor="middle", fill=INK)
    c.text(l + cw + cw / 2, t - 30, "实际正常", anchor="middle", fill=INK)
    total = sum(v for _, v, _ in cells)
    for k, (name, v, col) in enumerate(cells):
        i, j = divmod(k, 2)
        x, y = l + cw * j, t + chh * i
        share = v / total
        alpha = round(min(0.16 + share * 2.6, 0.95), 3)
        main, sub = text_on(col, alpha)          # 字色按底色对比度自动选，深底用白字
        c.rect(x, y, cw - 8, chh - 8, col, rx=3, opacity=alpha,
               **{"data-tip": esc(f"{name}：{num(v)} 人（占测试集 {share * 100:.1f}%）")})
        c.text(x + 14, y + 26, num(v), size=17, fill=main, weight="600")
        c.text(x + 14, y + 46, name, size=11, fill=sub)
    return c.svg()
