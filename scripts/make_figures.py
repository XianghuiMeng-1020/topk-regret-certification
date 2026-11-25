"""Render the README figures as standalone SVG from the real-data experiment outputs.

Only the standard library is used. Every plotted number is read from a committed
JSON file; figures/FIGURE_DATA.json records each value with its source file
and the file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "figures"

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e4e7eb"
STRONG = "#9aa5b1"
COUPLED = "#0b7285"
ACCENT = "#c2410c"
FONT = "font-family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel: str, used: dict) -> dict:
    path = EXP / rel
    used[f"experiments/{rel}"] = sha(path)
    return json.loads(path.read_text())


def svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' role='img' aria-label='{title}'>\n"
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>\n" + "\n".join(body) + "\n</svg>\n"
    )


def text(x, y, s, size=13, color=INK, anchor="start", weight="400"):
    return f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"


def fig_tau_curve(data: dict) -> tuple[str, dict]:
    taus = ["0.05", "0.1", "0.25", "0.5", "1.0"]
    W, H = 760, 370
    body = [text(24, 30, "Certification rate vs. tolerance τ (union-bound box event)", 16, weight="600"),
            text(24, 50, "Share of 540 cells per truth certified, averaged over truths. 81 truths.", 12, MUTED)]
    values = {}
    floor = min(data[t][k][s] for t in taus for k in ("tlc", "github") for s in ("strong_union", "box_union"))
    lo = int(floor * 20) / 20
    ticks = [round(lo + i * 0.05, 2) for i in range(int(round((1 - lo) / 0.05)) + 1)]
    panels = [("tlc", "NYC TLC · 45 truths", 40), ("github", "GitHub · 36 truths", 410)]
    for key, label, x0 in panels:
        pw, ph, y0 = 310, 200, 90
        body.append(text(x0, y0 - 14, label, 13, weight="600"))
        if all(data[t][key]["strong_union"] == data[t][key]["box_union"] for t in taus):
            body.append(text(x0 + pw, y0 - 14, "both lines at 1.00", 11, MUTED, "end"))
        for g in ticks:
            y = y0 + ph - (g - lo) / (1 - lo) * ph
            body.append(f"<line x1='{x0 + 36}' x2='{x0 + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>")
            body.append(text(x0 + 30, y + 4, f"{g:.2f}", 11, MUTED, "end"))
        for series, color, dash in (("strong_union", STRONG, "6 4"), ("box_union", COUPLED, "")):
            pts = []
            for i, t in enumerate(taus):
                v = data[t][key][series]
                values[f"{key}.{series}.tau={t}"] = v
                x = x0 + 50 + i * (pw - 70) / (len(taus) - 1)
                y = y0 + ph - (v - lo) / (1 - lo) * ph
                pts.append((x, y))
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            body.append(f"<polyline points='{d}' fill='none' stroke='{color}' stroke-width='2.5' stroke-dasharray='{dash}'/>")
            for x, y in pts:
                body.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>")
        for i, t in enumerate(taus):
            x = x0 + 50 + i * (pw - 70) / (len(taus) - 1)
            body.append(text(x, y0 + ph + 18, t, 11, MUTED, "middle"))
        body.append(text(x0 + pw / 2 + 18, y0 + ph + 38, "tolerance τ", 12, MUTED, "middle"))
    ly = H - 16
    body.append(f"<line x1='40' x2='66' y1='{ly - 4}' y2='{ly - 4}' stroke='{COUPLED}' stroke-width='2.5'/>")
    body.append(text(72, ly, "coupled certificate (one competitor)", 12))
    body.append(f"<line x1='330' x2='356' y1='{ly - 4}' y2='{ly - 4}' stroke='{STRONG}' stroke-width='2.5' stroke-dasharray='6 4'/>")
    body.append(text(362, ly, "strong separate-competitor baseline", 12))
    return svg(W, H, body, "Certification rate versus tau"), values


def fig_decomposition(box_boot: dict, os_boot: dict) -> tuple[str, dict]:
    rows = [
        ("Box event: coupled − strong", box_boot["truth_cluster"]["delta_union"]["point_from_full_sample"],
         box_boot["truth_cluster"]["delta_union"]["lo"], box_boot["truth_cluster"]["delta_union"]["hi"]),
        ("OS rank-1 box: coupled − strong", os_boot["point"]["rank1_coupling"],
         os_boot["truth_cluster"]["rank1_coupling"]["lo"], os_boot["truth_cluster"]["rank1_coupling"]["hi"]),
        ("OS-aware envelope: coupled − strong", os_boot["point"]["os_coupling"],
         os_boot["truth_cluster"]["os_coupling"]["lo"], os_boot["truth_cluster"]["os_coupling"]["hi"]),
        ("Coupled envelope − coupled rank-1", os_boot["point"]["coupled_refinement"],
         os_boot["truth_cluster"]["coupled_refinement"]["lo"], os_boot["truth_cluster"]["coupled_refinement"]["hi"]),
    ]
    W, H = 800, 300
    x0, x1 = 330, 700
    vmax = 0.07
    sx = lambda v: x0 + v / vmax * (x1 - x0)  # noqa: E731
    body = [text(24, 30, "Where the coupling gain comes from (τ = 0.10, equal-weight source macro)", 16, weight="600"),
            text(24, 50, "Points from the ledgers; bars are 95% truth-cluster bootstrap intervals (B = 1999).", 12, MUTED)]
    for g in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07):
        x = sx(g)
        body.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='70' y2='{H - 40}' stroke='{GRID if g else MUTED}'/>")
        body.append(text(x, H - 22, f"{g * 100:.0f} pp", 11, MUTED, "middle"))
    values = {}
    for i, (label, p, lo, hi) in enumerate(rows):
        y = 100 + i * 46
        values[label] = {"point": p, "lo": lo, "hi": hi}
        color = COUPLED if i < 2 else ACCENT
        body.append(text(24, y + 4, label, 12.5))
        body.append(f"<line x1='{sx(lo):.1f}' x2='{sx(hi):.1f}' y1='{y}' y2='{y}' stroke='{color}' stroke-width='6' stroke-linecap='round' opacity='0.35'/>")
        body.append(f"<circle cx='{sx(p):.1f}' cy='{y}' r='6' fill='{color}'/>")
        note = "   95% CI includes 0" if lo <= 0.0 else ""
        body.append(text(sx(hi) + 10, y + 4, f"{p * 100:+.2f} pp{note}", 12, color, weight="600"))
    return svg(W, H, body, "Decomposition of coupling gain"), values


def fig_time_forward(summary: dict) -> tuple[str, dict]:
    a = summary["A_union"]
    W, H = 760, 330
    da = a["D_accept"]["equal_weight_macro"]
    dr = a["D_regret"]["equal_weight_macro"]
    body = [text(24, 30, "Time-forward thresholds: calibrate on earlier days, apply to later days", 16, weight="600"),
            text(24, 50, f"Stratum-equal macro over 54 strata per source. Coupled − strong acceptance {da * 100:+.2f} pp; "
                         f"strong − coupled accepted regret {dr:.2e}.", 12, MUTED)]
    values = {"D_accept.macro": da, "D_regret.macro": dr}
    panels = [("frac", "Acceptance rate", 40, 0.0, 0.75, lambda v: f"{v:.3f}"),
              ("mean", "Mean normalized regret of accepted cells", 410, 0.0, 2.2e-5, lambda v: f"{v:.1e}")]
    base_y, span = 262, 160
    for key, label, px, lo, hi, fmt in panels:
        body.append(text(px, 84, label, 13, weight="600"))
        groups = [("gharchive", "GitHub"), ("tlc_yellow", "TLC"), (None, "Macro")]
        for gi, (src, gname) in enumerate(groups):
            gx = px + 22 + gi * 108
            for si, (side, color) in enumerate((("l", STRONG), ("r", COUPLED))):
                block = a[f"{key}_{side}"]
                v = block["equal_weight_macro"] if src is None else block[src]["mean"]
                values[f"{key}_{side}.{src or 'macro'}"] = v
                h = max((v - lo) / (hi - lo) * span, 1.0)
                x = gx + si * 40
                body.append(f"<rect x='{x}' y='{base_y - h:.1f}' width='34' height='{h:.1f}' rx='3' fill='{color}'/>")
                body.append(text(x + 17, base_y - h - 6, fmt(v), 10, color if side == "r" else MUTED, "middle", "600" if side == "r" else "400"))
            body.append(text(gx + 37, base_y + 18, gname, 12, MUTED, "middle"))
        body.append(f"<line x1='{px + 10}' x2='{px + 340}' y1='{base_y}' y2='{base_y}' stroke='{MUTED}'/>")
    body.append(f"<rect x='40' y='{H - 22}' width='12' height='12' fill='{STRONG}'/>")
    body.append(text(58, H - 12, "strong baseline", 12))
    body.append(f"<rect x='190' y='{H - 22}' width='12' height='12' fill='{COUPLED}'/>")
    body.append(text(208, H - 12, "coupled certificate", 12))
    return svg(W, H, body, "Time-forward utility"), values


def main() -> int:
    OUT.mkdir(exist_ok=True)
    used: dict[str, str] = {}
    figs = {}
    s1, v1 = fig_tau_curve(load("real_box_evaluation/source_results.json", used))
    s2, v2 = fig_decomposition(load("real_box_evaluation/bootstrap.json", used), load("real_order_statistic_evaluation/bootstrap.json", used))
    s3, v3 = fig_time_forward(load("time_forward_utility/primary_summary.json", used))
    for name, content, values in (("certification_rate_vs_tau.svg", s1, v1),
                                  ("coupling_gain_decomposition.svg", s2, v2),
                                  ("time_forward_utility.svg", s3, v3)):
        (OUT / name).write_text(content, encoding="utf-8")
        figs[name] = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "values": values}
    (OUT / "FIGURE_DATA.json").write_text(json.dumps({"sources": used, "figures": figs}, indent=2) + "\n")
    print(json.dumps({k: v["sha256"][:12] for k, v in figs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
