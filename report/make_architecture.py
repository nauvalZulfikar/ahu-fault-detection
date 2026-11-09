"""Draw the Q3 architecture diagram (report/figures/architecture.png)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "figures" / "architecture.png"
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

def zone(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec="#888", lw=1, alpha=0.35))
    ax.text(x + 0.12, y + h - 0.25, title, fontsize=10, weight="bold", color="#333")

def box(x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3)
    return (x, y, w, h)

def arrow(a, b, label="", side="r", dy=0):
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "r":   s, e = (ax0 + aw, ay0 + ah / 2 + dy), (bx0, by0 + bh / 2 + dy)
    elif side == "d": s, e = (ax0 + aw / 2, ay0), (bx0 + bw / 2, by0 + bh)
    else:             s, e = (ax0 + aw / 2, ay0 + ah), (bx0 + bw / 2, by0)
    ax.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.1))
    if label:
        ax.text((s[0] + e[0]) / 2, (s[1] + e[1]) / 2 + 0.12, label, ha="center", fontsize=7.3, color="#555")

zone(0.1, 0.3, 2.5, 5.6, "Building (x25)", "#d9e8f5")
zone(2.9, 0.3, 6.3, 5.6, "Cloud", "#e3f0dc")
zone(9.5, 0.3, 2.4, 5.6, "Client", "#f5e6d3")

def line(points, color="#444", ls="-"):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.2, ls=ls)
    ax.annotate("", points[-1], points[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls))

bms    = box(0.35, 4.3, 2.0, 0.8, "BMS SQL Server\n(5-min trends)")
agent  = box(0.35, 2.6, 2.0, 1.2, "Edge agent\nincremental read\npoint mapping\n14-day buffer")
ingest = box(3.1, 4.4, 1.8, 0.8, "Ingest API\n(dedupe, auth)")
raw    = box(3.1, 2.9, 1.8, 0.9, "Raw zone\n(append-only\nParquet)")
mon    = box(3.1, 0.8, 1.8, 1.1, "Monitoring\nfreshness, DQ,\ndrift, precision")
dq     = box(5.2, 4.4, 1.8, 0.8, "DQ checks (clean.py)\nflags per AHU-hour")
cur    = box(5.2, 2.9, 1.8, 0.9, "Curated store\n+ labels\n(Postgres/Timescale)")
infer  = box(5.2, 0.8, 1.8, 1.1, "Hourly inference\nfeatures, scores,\nalarm state")
train  = box(7.3, 4.4, 1.7, 0.8, "Weekly retrain\n+ LOMO eval")
reg    = box(7.3, 2.9, 1.7, 0.9, "Model registry\nshadow -> live")
rules  = box(7.3, 0.8, 1.7, 1.1, "Alert rules\ndedupe, building\ngrouping, severity")
dig    = box(9.7, 4.4, 2.0, 0.8, "Weekly digest\n+ dashboard")
cmms   = box(9.7, 2.6, 2.0, 1.2, "CMMS\n(REST API)\nwork orders")
tech   = box(9.7, 0.8, 2.0, 1.0, "Technicians\nconfirm / no fault")

arrow(bms, agent, side="d")
line([(2.35, 3.2), (3.1, 4.8)]); ax.text(2.45, 4.25, "HTTPS\nout only", fontsize=7.3, color="#555")
arrow(ingest, raw, side="d")
line([(4.9, 3.6), (5.2, 4.6)])                       # raw -> DQ
arrow(dq, cur, side="d")
arrow(cur, infer, side="d")
line([(7.0, 3.6), (7.3, 4.6)])                       # curated -> training
arrow(train, reg, side="d")
line([(7.3, 3.0), (7.0, 1.7)]); ax.text(7.36, 2.2, "bundle", fontsize=7.3, color="#555")
arrow(infer, rules, side="r")
line([(5.2, 1.35), (4.9, 1.35)])                     # inference -> monitoring
line([(9.0, 1.6), (9.7, 3.0)])                       # rules -> CMMS
line([(9.0, 1.8), (9.7, 4.6)])                       # rules -> digest
arrow(cmms, tech, side="d")
# feedback: CMMS outcomes back to curated labels, through the gap between rows
line([(9.7, 3.6), (9.4, 3.6), (9.4, 4.1), (6.7, 4.1), (6.7, 3.8)], color="#c0392b", ls="--")
ax.text(7.35, 4.16, "outcomes = new labels", fontsize=7.5, color="#c0392b")
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print(OUT)
