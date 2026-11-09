"""Write an HTML report of a CAT run: the properties and a 3D view of every molecule.

The report is a single, self-contained HTML file: it needs neither an internet connection
nor any other file, and can be opened with any web browser.
It is written by :func:`CAT.base.prep` if :attr:`optional.report` is set.

Index
-----
.. currentmodule:: CAT.report
.. autosummary::
    write_report

API
---
.. autofunction:: write_report

"""

from __future__ import annotations

import os
import html
import contextlib
import importlib
import json
import math
import datetime
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scm.plams import Molecule

from .__version__ import __version__
from ._compat import atoms_as_array
from .utils import get_formula
from .workflows import MOL

__all__ = ['write_report']

#: The (first-level) columns that are not shown in the tables
HIDDEN_COLUMNS = frozenset({'mol', 'hdf5 index', 'settings'})

#: The colors of the elements (those of Jmol); other elements are pink
COLORS = {
    'H': '#FFFFFF', 'Li': '#CC80FF', 'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8',
    'O': '#FF0D0D', 'F': '#90E050', 'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6',
    'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'K': '#8F40D4',
    'Ca': '#3DFF00', 'Ti': '#BFC2C7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
    'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3',
    'Se': '#FFA100', 'Br': '#A62929', 'Rb': '#702EB0', 'Ag': '#C0C0C0', 'Cd': '#FFD98F',
    'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094',
    'Cs': '#57178F', 'Au': '#FFD123', 'Hg': '#B8B8D0', 'Pb': '#575961', 'Bi': '#9E4FB5',
}


def _fmt(value: Any, max_len: int = 80) -> str:
    """Format a single table cell, shortening strings longer than **max_len**."""
    if value is None:
        return ''
    elif isinstance(value, (bool, np.bool_)):
        return 'yes' if value else 'no'
    elif isinstance(value, (float, np.floating)):
        return '' if math.isnan(value) else f'{value:.6g}'
    ret = str(value)
    return ret if len(ret) <= max_len else f'{ret[:max_len - 3]}...'


def _column_name(key: Any) -> str:
    """Return the header of a column: ``"index"`` or ``"index: sub index"``."""
    if isinstance(key, tuple):
        top, sub = (str(i) for i in key[:2]) if len(key) > 1 else (str(key[0]), '')
        return top if not sub else f'{top}: {sub}'
    return str(key)


def _shown_columns(df: pd.DataFrame) -> list[Any]:
    """Return the columns of **df** that are shown in the table."""
    ret = []
    for key in df.columns:
        top = key[0] if isinstance(key, tuple) else key
        if top in HIDDEN_COLUMNS or str(top).startswith('job_settings'):
            continue
        ret.append(key)
    return ret


def _structure(mol: Molecule, name: str) -> dict[str, Any]:
    """Return the JSON-compatible structure of **mol** for the 3D viewer."""
    mol.set_atoms_id(start=0)
    try:
        bonds = [[b.atom1.id, b.atom2.id] for b in mol.bonds]
    finally:
        mol.unset_atoms_id()
    return {
        'name': name,
        'symbols': [at.symbol for at in mol],
        'xyz': np.round(atoms_as_array(mol), 3).ravel().tolist(),
        'bonds': bonds,
    }


def _table(df: pd.DataFrame, kind: str,
           structures: dict[str, dict[str, Any]]) -> str:
    """Return an HTML table of **df**, adding the structure of each molecule to **structures**."""
    index_names = [str(i) for i in df.index.names if i is not None] or ['index']
    columns = _shown_columns(df)
    # Ligands and cores already have a formula column
    add_formula = 'formula' not in df.columns.get_level_values(0)
    head = ''.join(f'<th>{html.escape(i)}</th>' for i in index_names)
    head += '<th>formula</th><th>atoms</th>' if add_formula else '<th>atoms</th>'
    head += ''.join(f'<th>{html.escape(_column_name(i))}</th>' for i in columns)

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        idx_tup = idx if isinstance(idx, tuple) else (idx,)
        name = ' | '.join(_fmt(j, max_len=30) for j in idx_tup)
        mol = row[MOL] if MOL in df.columns else None
        cells = ''.join(f'<td>{html.escape(_fmt(j))}</td>' for j in idx_tup)
        if isinstance(mol, Molecule) and len(mol):
            key = f'{kind}-{i}'
            structures[key] = _structure(mol, f'{kind}: {name}')
            if add_formula:
                cells += f'<td>{html.escape(get_formula(mol))}</td>'
            cells += f'<td>{len(mol)}</td>'
            button = (f'<button type="button" class="view" data-key="{key}" '
                      f'aria-label="View {html.escape(name)}">View</button>')
        else:
            cells += '<td></td><td></td>' if add_formula else '<td></td>'
            button = ''
        cells += ''.join(f'<td>{html.escape(_fmt(row[j]))}</td>' for j in columns)
        rows.append(f'<tr><td class="action">{button}</td>{cells}</tr>')

    return (f'<div class="table-wrap"><table><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _versions() -> str:
    """Return the versions of CAT and, if installed, Nano-CAT and Data-CAT."""
    ret = [f'CAT {__version__}']
    for module, name in [('nanoCAT', 'Nano-CAT'), ('dataCAT', 'Data-CAT')]:
        with contextlib.suppress(Exception):
            ret.append(f'{name} {importlib.import_module(module).__version__}')
    return ', '.join(ret)


def write_report(
    filename: str | os.PathLike[str],
    qd_df: pd.DataFrame | None = None,
    core_df: pd.DataFrame | None = None,
    ligand_df: pd.DataFrame | None = None,
    *,
    title: str = 'CAT report',
    elapsed: float | None = None,
) -> None:
    """Write an HTML report of the quantum dots, cores and ligands of a CAT run.

    The report contains a table of each type of molecule, with all their properties
    (*e.g.* the results of the optional workflows), and a 3D view of every molecule.

    Parameters
    ----------
    filename : :term:`path-like object`
        The name of the HTML file.
    qd_df, core_df, ligand_df : :class:`pandas.DataFrame`, optional
        The quantum dots, cores and ligands, as returned by :func:`CAT.base.prep`.
    title : :class:`str`
        The title of the report.
    elapsed : :class:`float`, optional
        The duration of the run in seconds.

    """
    structures: dict[str, dict[str, Any]] = {}
    sections = []
    counts = []
    items: Iterable[tuple[str, str, str, pd.DataFrame | None]] = [
        ('qd', 'Quantum dots', 'quantum dot', qd_df),
        ('ligand', 'Ligands', 'ligand', ligand_df),
        ('core', 'Cores', 'core', core_df),
    ]
    for kind, label, singular, df in items:
        if df is None:
            continue
        noun = singular if len(df) == 1 else label.lower()
        counts.append(f'<div class="card"><b>{len(df)}</b><span>{noun}</span></div>')
        if not len(df):
            continue
        sections.append(f'<section><h2>{label}</h2>{_table(df, kind, structures)}</section>')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    meta = f'{html.escape(_versions())} &middot; {now}'
    if elapsed is not None:
        meta += f' &middot; {elapsed:.1f} s'

    data = json.dumps({'structures': structures, 'colors': COLORS}, separators=(',', ':'))
    data = data.replace('</', '<\\/')
    page = _TEMPLATE.format(
        title=html.escape(title), meta=meta, counts=''.join(counts),
        sections=''.join(sections) or '<p>No molecules.</p>', data=data,
    )
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(page)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #ffffff; --fg: #1d2330; --muted: #5b6475; --line: #dde1e8; --head: #f3f5f8;
  --accent: #2f6fdf; --viewer: #f7f8fa;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14171c; --fg: #e4e7ec; --muted: #9aa3b2; --line: #2c323c; --head: #1c2027;
    --accent: #6d9cf0; --viewer: #0e1014;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
header h1 {{ margin: 0 0 4px; font-size: 22px; }}
header p {{ margin: 0 0 16px; color: var(--muted); }}
.cards {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.card {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; }}
.card b {{ font-size: 20px; margin-right: 6px; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }}
@media (min-width: 1100px) {{
  .layout {{ grid-template-columns: minmax(0, 1fr) 460px; }}
  .viewer {{ position: sticky; top: 16px; align-self: start; }}
}}
.viewer {{ border: 1px solid var(--line); border-radius: 8px; background: var(--viewer); }}
.viewer h2 {{ font-size: 14px; margin: 0; padding: 8px 12px; border-bottom: 1px solid var(--line);
  overflow-wrap: anywhere; }}
.viewer canvas {{ display: block; width: 100%; height: 420px; touch-action: none; cursor: grab; }}
.viewer p {{ margin: 0; padding: 6px 12px; color: var(--muted); font-size: 12px; }}
section h2 {{ font-size: 17px; margin: 8px 0; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px;
  margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 5px 8px; border-bottom: 1px solid var(--line); text-align: left;
  white-space: nowrap; }}
th {{ background: var(--head); position: sticky; top: 0; font-weight: 600; }}
tr.selected td {{ background: color-mix(in srgb, var(--accent) 14%, transparent); }}
button.view {{ border: 1px solid var(--accent); color: var(--accent); background: transparent;
  border-radius: 6px; padding: 2px 10px; cursor: pointer; font: inherit; }}
button.view:hover, button.view:focus-visible {{ background: var(--accent); color: var(--bg); }}
</style>
</head>
<body>
<header>
<h1>{title}</h1>
<p>{meta}</p>
</header>
<div class="cards">{counts}</div>
<div class="layout">
<main>{sections}</main>
<aside class="viewer" aria-label="3D view">
<h2 id="viewer-title">Select a molecule</h2>
<canvas id="viewer" aria-label="3D view of the selected molecule"></canvas>
<p>Drag to rotate, scroll or pinch to zoom, double-click to reset.</p>
</aside>
</div>
<script type="application/json" id="cat-data">{data}</script>
<script>
(function () {{
  "use strict";
  var DATA = JSON.parse(document.getElementById("cat-data").textContent);
  var canvas = document.getElementById("viewer");
  var title = document.getElementById("viewer-title");
  var ctx = canvas.getContext("2d");
  var RADII = {{H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05, Cl: 1.02,
    Br: 1.20, I: 1.39, Se: 1.20, Te: 1.38, Cd: 1.44, Zn: 1.22, Pb: 1.46, Cs: 2.44, In: 1.42}};
  var mol = null, rot = [1, 0, 0, 0, 1, 0, 0, 0, 1], zoom = 1, drag = null, pinch = null;

  function mul(a, b) {{
    var r = [];
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++)
      r.push(a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j]);
    return r;
  }}
  function rotation(dx, dy) {{
    var cx = Math.cos(dy), sx = Math.sin(dy), cy = Math.cos(dx), sy = Math.sin(dx);
    var ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy], rx = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
    return mul(rx, ry);
  }}
  function load(key) {{
    var s = DATA.structures[key], n = s.symbols.length, c = [0, 0, 0], i, k;
    for (i = 0; i < n; i++) for (k = 0; k < 3; k++) c[k] += s.xyz[3 * i + k] / n;
    var xyz = [], extent = 1;
    for (i = 0; i < n; i++) {{
      var p = [s.xyz[3 * i] - c[0], s.xyz[3 * i + 1] - c[1], s.xyz[3 * i + 2] - c[2]];
      extent = Math.max(extent, Math.hypot(p[0], p[1], p[2]) + 1.5);
      xyz.push(p);
    }}
    mol = {{symbols: s.symbols, xyz: xyz, bonds: s.bonds, extent: extent}};
    rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]; zoom = 1;
    title.textContent = s.name;
    draw();
  }}
  function draw() {{
    var dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!mol) return;
    var scale = zoom * Math.min(w, h) / (2 * mol.extent), pts = [], items = [], i;
    for (i = 0; i < mol.xyz.length; i++) {{
      var p = mol.xyz[i];
      var x = rot[0] * p[0] + rot[1] * p[1] + rot[2] * p[2];
      var y = rot[3] * p[0] + rot[4] * p[1] + rot[5] * p[2];
      var z = rot[6] * p[0] + rot[7] * p[1] + rot[8] * p[2];
      pts.push([w / 2 + x * scale, h / 2 - y * scale, z]);
      items.push([z, 0, i]);
    }}
    for (i = 0; i < mol.bonds.length; i++) {{
      var b = mol.bonds[i];
      items.push([(pts[b[0]][2] + pts[b[1]][2]) / 2 - 0.01, 1, i]);
    }}
    items.sort(function (a, b) {{ return a[0] - b[0]; }});
    ctx.lineCap = "round";
    for (i = 0; i < items.length; i++) {{
      var it = items[i];
      if (it[1] === 1) {{
        var bd = mol.bonds[it[2]], p1 = pts[bd[0]], p2 = pts[bd[1]];
        var mx = (p1[0] + p2[0]) / 2, my = (p1[1] + p2[1]) / 2;
        ctx.lineWidth = Math.max(1.5, 0.22 * scale);
        ctx.strokeStyle = color(mol.symbols[bd[0]]);
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(mx, my); ctx.stroke();
        ctx.strokeStyle = color(mol.symbols[bd[1]]);
        ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
      }} else {{
        var sym = mol.symbols[it[2]], pt = pts[it[2]];
        var r = Math.max(2, (0.18 + 0.32 * (RADII[sym] || 1.3)) * scale);
        var g = ctx.createRadialGradient(pt[0] - r / 3, pt[1] - r / 3, r / 6, pt[0], pt[1], r);
        g.addColorStop(0, "#ffffff"); g.addColorStop(0.35, color(sym));
        g.addColorStop(1, shade(color(sym)));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(pt[0], pt[1], r, 0, 2 * Math.PI); ctx.fill();
      }}
    }}
  }}
  function color(sym) {{ return DATA.colors[sym] || "#FF1493"; }}
  function shade(hex) {{
    var v = parseInt(hex.slice(1), 16);
    var r = (v >> 16) * 0.45, g = ((v >> 8) & 255) * 0.45, b = (v & 255) * 0.45;
    return "rgb(" + (r | 0) + "," + (g | 0) + "," + (b | 0) + ")";
  }}
  canvas.addEventListener("pointerdown", function (e) {{
    drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId);
  }});
  canvas.addEventListener("pointermove", function (e) {{
    if (!drag || pinch) return;
    rot = mul(rotation((e.clientX - drag[0]) / 120, (e.clientY - drag[1]) / 120), rot);
    drag = [e.clientX, e.clientY]; draw();
  }});
  canvas.addEventListener("pointerup", function () {{ drag = null; }});
  canvas.addEventListener("wheel", function (e) {{
    e.preventDefault(); zoom *= Math.exp(-e.deltaY / 500); draw();
  }}, {{passive: false}});
  canvas.addEventListener("touchmove", function (e) {{
    if (e.touches.length !== 2) {{ pinch = null; return; }}
    var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                       e.touches[0].clientY - e.touches[1].clientY);
    if (pinch) {{ zoom *= d / pinch; draw(); }}
    pinch = d;
  }});
  canvas.addEventListener("touchend", function () {{ pinch = null; }});
  canvas.addEventListener("dblclick", function () {{
    rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]; zoom = 1; draw();
  }});
  window.addEventListener("resize", draw);
  document.addEventListener("click", function (e) {{
    var btn = e.target.closest("button.view");
    if (!btn) return;
    document.querySelectorAll("tr.selected").forEach(function (tr) {{
      tr.classList.remove("selected");
    }});
    btn.closest("tr").classList.add("selected");
    load(btn.dataset.key);
    if (window.innerWidth < 1100) canvas.scrollIntoView({{behavior: "smooth", block: "center"}});
  }});
  var first = document.querySelector("button.view");
  if (first) {{ first.closest("tr").classList.add("selected"); load(first.dataset.key); }}
}})();
</script>
</body>
</html>
"""
