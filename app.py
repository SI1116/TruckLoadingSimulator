# app.py
# 依存: pip install streamlit numpy plotly streamlit-plotly-events

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="紙袋×パレット×荷台 3D積載シミュレーター（忠実寸法＋上限制御＋非重なり保証）", layout="wide")

EPS = 1e-6

# ========= ユーティリティ =========
def mm3_to_m3(v_mm3: float) -> float:
    return v_mm3 / 1e9

def grid_fit_with_gap(cap_len, cap_wid, item_len, item_wid, gap):
    """パレット同士の隙間(gap)込みのグリッド充填（列数・行数）"""
    if item_len <= 0 or item_wid <= 0:
        return 0, 0
    cols = int((cap_len + gap) // (item_len + gap))
    rows = int((cap_wid + gap) // (item_wid + gap))
    return max(0, cols), max(0, rows)

def parse_row_pattern(text: str):
    """行パターン文字列を解析（例: '3-2-3' → [3,2,3]）"""
    try:
        parts = [p.strip() for p in text.split('-') if p.strip() != '']
        nums = [int(p) for p in parts]
        if all(n > 0 for n in nums):
            return nums
    except:
        pass
    return None

def compute_bara_z_layers(h_eff, bag_H):
    """有効高さ h_eff 内で積める ばら積みの段数（Z方向）"""
    return 0 if bag_H <= 0 else max(0, int(h_eff // bag_H))

# ---- AABB（軸平行バウンディングボックス）衝突判定 ----
def aabb_overlap_strict(o1, s1, o2, s2, eps=EPS):
    """True なら“体積を持って”重なる（接触だけはOK）"""
    (x1,y1,z1) = o1; (L1,W1,H1) = s1
    (x2,y2,z2) = o2; (L2,W2,H2) = s2
    def ov(a0,a1,b0,b1):
        return (a0 < b1 - eps) and (b0 < a1 - eps)  # 境界接触はOK
    return ov(x1, x1+L1, x2, x2+L2) and ov(y1, y1+W1, y2, y2+W2) and ov(z1, z1+H1, z2, z2+H2)

# ---- Plotly: 直方体メッシュ（面） ----
def cuboid_mesh(origin, size, color, opacity=1.0, name=None, showlegend=False):
    """直方体を Mesh3d で描画（忠実寸法。縮小・拡大なし）"""
    x0, y0, z0 = origin
    L, W, H = size
    x = np.array([x0, x0+L, x0+L, x0,   x0,   x0+L, x0+L, x0])
    y = np.array([y0, y0,   y0+W, y0+W, y0,   y0,   y0+W, y0+W])
    z = np.array([z0, z0,   z0,   z0,   z0+H, z0+H, z0+H, z0+H])
    I = [
        0,1,2, 0,2,3,
        4,5,6, 4,6,7,
        0,1,5, 0,5,4,
        1,2,6, 1,6,5,
        2,3,7, 2,7,6,
        3,0,4, 3,4,7
    ]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=I[0::3], j=I[1::3], k=I[2::3],
        color=color, opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.9, specular=0.0, fresnel=0.0, roughness=1.0),
        hoverinfo="skip",
        name=name or "", showlegend=showlegend
    )

# ---- 輪郭（複数箱を1トレースに統合） ----
def edges_group_trace(boxes, color="#000000", width=2, name=None, showlegend=False):
    if not boxes:
        return None
    xs, ys, zs = [], [], []
    def add(a,b):
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    for (o, s) in boxes:
        x0,y0,z0 = o; L,W,H = s
        P = [
            (x0,     y0,     z0),
            (x0+L,   y0,     z0),
            (x0+L,   y0+W,   z0),
            (x0,     y0+W,   z0),
            (x0,     y0,     z0+H),
            (x0+L,   y0,     z0+H),
            (x0+L,   y0+W,   z0+H),
            (x0,     y0+W,   z0+H),
        ]
        E = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        for a,b in E:
            add(P[a], P[b])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        showlegend=showlegend, name=name if showlegend else None,
        hoverinfo="skip"
    )

# ---- 複数直方体を Mesh3d 1本に統合（高速） ----
def cuboid_mesh_group(boxes, color, opacity=1.0, name=None, showlegend=False):
    if not boxes:
        return None
    x, y, z, i, j, k = [], [], [], [], [], []
    base = 0
    for (origin, size) in boxes:
        x0, y0, z0 = origin
        L, W, H = size
        xv = [x0, x0+L, x0+L, x0,   x0,   x0+L, x0+L, x0]
        yv = [y0, y0,   y0+W, y0+W, y0,   y0,   y0+W, y0+W]
        zv = [z0, z0,   z0,   z0,   z0+H, z0+H, z0+H, z0+H]
        idx = [
            0,1,2, 0,2,3,
            4,5,6, 4,6,7,
            0,1,5, 0,5,4,
            1,2,6, 1,6,5,
            2,3,7, 2,7,6,
            3,0,4, 3,4,7
        ]
        x.extend(xv); y.extend(yv); z.extend(zv)
        i.extend([base + idx[n] for n in range(0, len(idx), 3)])
        j.extend([base + idx[n] for n in range(1, len(idx), 3)])
        k.extend([base + idx[n] for n in range(2, len(idx), 3)])
        base += 8
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.9, specular=0.0, fresnel=0.0, roughness=1.0),
        hoverinfo="skip",
        name=name or "", showlegend=showlegend
    )

# === 表示用：荷台中心(0,0,0)に平行移動（スケールは変えない） ===
def center_origin(origin, truck_L, truck_W, truck_H_draw):
    cx, cy, cz = truck_L/2.0, truck_W/2.0, truck_H_draw/2.0
    x, y, z = origin
    return (x - cx, y - cy, z - cz)

# =========（新規）等間隔配置のヘルパ =========
def max_cols_feasible(Lp, Lb, gap_min):
    if Lb <= 0: return 0
    return int((Lp + gap_min) // (Lb + gap_min))

def compute_row_x_positions_equal_gap_edges(Lp, Lb, N, gap_min):
    if N <= 0: return []
    if N == 1:
        return [(Lp - Lb)/2.0]
    if (Lp - N*Lb) < (N-1)*gap_min - EPS:
        return []
    gap = (Lp - N*Lb) / (N-1)
    xs = [0.0]
    for i in range(1, N):
        xs.append(xs[-1] + Lb + gap)
    xs[-1] = Lp - Lb
    return xs

def layout_rows_y_equal_gap_edges(Wp, row_heights, gap_min):
    R = len(row_heights)
    if R == 0: return []
    if R == 1:
        return [(Wp - row_heights[0]) / 2.0]
    if Wp - sum(row_heights) < (R-1)*gap_min - EPS:
        return []
    free = Wp - sum(row_heights)
    g = free / (R-1)
    ys = [0.0]
    for i in range(1, R):
        ys.append(ys[-1] + row_heights[i-1] + g)
    ys[-1] = Wp - row_heights[-1]
    return ys

# ========= パレット上：一般パターン =========
def palletize_generic(Lp, Wp, Hp, Lb, Wb, Hb, gap_xy, rows_pattern, layers, alternate_row_rotation, h_eff):
    max_layers = 0 if Hb <= 0 else max(0, int((h_eff - Hp + EPS) // Hb))
    layers = max(0, min(layers, max_layers))
    bag_boxes = []; z0 = Hp; per_layer_count = 0
    if layers == 0:
        return bag_boxes, layers, 0
    for l in range(layers):
        y = 0.0; r = 0
        while True:
            rotate = (alternate_row_rotation and (r % 2 == 1))
            bag_L = (Wb if rotate else Lb)
            bag_W = (Lb if rotate else Wb)
            row_n = rows_pattern[r % len(rows_pattern)]
            n_fit = int((Lp + gap_xy) // (bag_L + gap_xy))
            place_n = min(row_n, max(0, n_fit))
            if place_n <= 0: break
            x = 0.0
            for _ in range(place_n):
                bag_boxes.append(((x, y, z0 + l*Hb), (bag_L, bag_W, Hb)))
                per_layer_count += 1
                x += bag_L + gap_xy
            y_next = y + bag_W + gap_xy
            if y_next - gap_xy + bag_W > Wp + EPS:
                break
            y = y_next
            r += 1
    return bag_boxes, layers, (per_layer_count // max(1, layers))

# ========= 8俵パターン =========
def palletize_8hyo_even_positions(Lp, Wp, Hp, Lb, Wb, Hb, gap_xy, layers, h_eff):
    max_layers = 0 if Hb <= 0 else max(0, int((h_eff - Hp + EPS) // Hb))
    layers = max(0, min(layers, max_layers))
    boxes = []; z0 = Hp
    SIZE_TATE = (Wb, Lb, Hb); SIZE_YOKO = (Lb, Wb, Hb)
    def center_start(total, needed): return max(0.0, (total - needed) / 2.0)
    if layers == 0:
        return boxes, 0, 0
    for l in range(layers):
        z = z0 + l*Hb
        odd = ((l+1) % 2 == 1)
        if odd:
            x_pitch = Wb + gap_xy; y_pitch = Lb + gap_xy
            cols = min(4, int((Lp + gap_xy) // x_pitch))
            rows = min(2, int((Wp + gap_xy) // y_pitch))
            need_x = cols*x_pitch - (gap_xy if cols>0 else 0.0)
            x0 = center_start(Lp, need_x)
            ys = [0.0, Wp - Lb] if rows >= 2 else ([center_start(Wp, Lb)] if rows == 1 else [])
            for y in ys:
                for c in range(cols):
                    x = x0 + c*x_pitch
                    if x + Wb <= Lp + EPS and y + Lb <= Wp + EPS:
                        boxes.append(((x, y, z), SIZE_TATE))
        else:
            x_left = 0.0; x_right = Lp - Lb
            y_top = 0.0; y_mid = max(0.0, (Wp - Wb)/2.0); y_bot = Wp - Wb
            for y in [y_top, y_mid, y_bot]:
                if y + Wb <= Wp + EPS:
                    if x_left + Lb <= Lp + EPS:  boxes.append(((x_left,  y, z), SIZE_YOKO))
                    if x_right + Lb <= Lp + EPS: boxes.append(((x_right, y, z), SIZE_YOKO))
            need_y_center = 2*Lb + gap_xy
            y_c1 = center_start(Wp, need_y_center); y_c2 = y_c1 + Lb + gap_xy
            x_center = center_start(Lp, Wb)
            if x_center + Wb <= Lp + EPS:
                if y_c1 + Lb <= Wp + EPS: boxes.append(((x_center, y_c1, z), SIZE_TATE))
                if y_c2 + Lb <= Wp + EPS: boxes.append(((x_center, y_c2, z), SIZE_TATE))
    return boxes, layers, 8

# =========（新規）5俵ばい =========
def palletize_5hyo_edges_center(Lp, Wp, Hp, Lb, Wb, Hb, gap_xy, layers, alternate_row_rotation, h_eff):
    max_layers = 0 if Hb <= 0 else max(0, int((h_eff - Hp + EPS) // Hb))
    layers = max(0, min(layers, max_layers))
    if layers == 0:
        return [], 0, 0

    z0 = Hp
    boxes_all = []
    per_layer_count_total = 0
    base_pattern = [2, 3]

    for l in range(layers):
        rows = []
        r = 0
        while True:
            rotate = (alternate_row_rotation and (r % 2 == 1))
            bag_L = (Wb if rotate else Lb)
            bag_W = (Lb if rotate else Wb)
            desired = base_pattern[r % 2]
            n_fit_edge = max_cols_feasible(Lp, bag_L, gap_xy)
            place_n = min(desired, n_fit_edge)
            if place_n <= 0:
                break
            rows.append((place_n, bag_L, bag_W))
            total_min_h = sum(h for (_, _, h) in rows)
            R = len(rows)
            need_h = total_min_h + max(0, R - 1) * gap_xy
            if need_h > Wp + EPS:
                rows.pop()
                break
            r += 1

        if not rows:
            break

        row_heights = [h for (_, _, h) in rows]
        ys = layout_rows_y_equal_gap_edges(Wp, row_heights, gap_xy)
        if not ys:
            place_n, bag_L2, bag_W2 = rows[0]
            ys = [(Wp - bag_W2) / 2.0]
            rows = [(place_n, bag_L2, bag_W2)]

        layer_boxes = []
        for (y, (n, bL, bW)) in zip(ys, rows):
            xs = compute_row_x_positions_equal_gap_edges(Lp, bL, n, gap_xy)
            if not xs:
                continue
            for x in xs:
                layer_boxes.append(((x, y, z0 + l * Hb), (bL, bW, Hb)))

        if ((l + 1) % 2) == 0:
            rotated_layer = []
            for (o, s) in layer_boxes:
                x, y, z = o
                bL, bW, bH = s
                x2 = Lp - x - bL
                y2 = Wp - y - bW
                rotated_layer.append(((x2, y2, z), (bL, bW, bH)))
            layer_boxes = rotated_layer

        boxes_all.extend(layer_boxes)
        per_layer_count_total += len(layer_boxes)

    per_layer = per_layer_count_total // max(1, layers)
    return boxes_all, layers, per_layer

# =========（新規）MaxRects ばら積みパッカー =========
# Rect: (x, y, w, h)
def _rects_intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx + EPS or bx + bw <= ax + EPS or
                ay + ah <= by + EPS or by + bh <= ay + EPS)

def _split_free_by_used(free_rect, used):
    """free_rect から used を差し引き、残り（最大で4分割）を返す"""
    fx, fy, fw, fh = free_rect
    ux, uy, uw, uh = used
    out = []
    # 上
    if uy > fy + EPS and uy < fy + fh - EPS:
        out.append((fx, fy, fw, uy - fy))
        fy = uy
        fh = (fy + fh) - uy
    # 下（上でfy更新済み）
    if uy + uh > fy + EPS and uy + uh < fy + fh - EPS:
        out.append((fx, uy + uh, fw, fy + fh - (uy + uh)))
        fh = (uy + uh) - fy
    # 左
    if ux > fx + EPS and ux < fx + fw - EPS:
        out.append((fx, fy, ux - fx, fh))
        fx = ux
        fw = (fx + fw) - ux
    # 右
    if ux + uw > fx + EPS and ux + uw < fx + fw - EPS:
        out.append((ux + uw, fy, fx + fw - (ux + uw), fh))
        fw = (ux + uw) - fx
    # 中央の残りは使えない（ちょうど used が占有）
    return [r for r in out if r[2] > EPS and r[3] > EPS]

def _prune_contained(rects):
    kept = []
    for i, a in enumerate(rects):
        ax, ay, aw, ah = a
        contained = False
        for j, b in enumerate(rects):
            if i == j: continue
            bx, by, bw, bh = b
            if (ax + EPS) >= bx and (ay + EPS) >= by and \
               (ax + aw) <= (bx + bw + EPS) and (ay + ah) <= (by + bh + EPS):
                contained = True
                break
        if not contained:
            kept.append(a)
    return kept

def _place_one(free_rects, w, h):
    """Best Short Side Fit"""
    best = None
    for fr in free_rects:
        fx, fy, fw, fh = fr
        # そのまま
        if w <= fw + EPS and h <= fh + EPS:
            s_short = min(fw - w, fh - h)
            s_long  = max(fw - w, fh - h)
            cand = (s_short, s_long, fx, fy, w, h)
            if (best is None) or (cand < best):
                best = cand
        # 回転
        if h <= fw + EPS and w <= fh + EPS:
            s_short = min(fw - h, fh - w)
            s_long  = max(fw - h, fh - w)
            cand = (s_short, s_long, fx, fy, h, w)
            if (best is None) or (cand < best):
                best = cand
    if best is None:
        return None
    _, _, px, py, pw, ph = best
    placed = (px, py, pw, ph)
    # free_rects を更新
    new_free = []
    for fr in free_rects:
        if _rects_intersect(fr, placed):
            new_free.extend(_split_free_by_used(fr, placed))
        else:
            new_free.append(fr)
    new_free = _prune_contained(new_free)
    return placed, new_free

def pack_bara_maxrects(truck_rect, pallet_rects, bag_L, bag_W, gap_xy, H, z_layers, quota=None):
    """
    truck_rect: (x0,y0,L,W)  … 荷台の有効領域
    pallet_rects: [(x,y,w,h), ...] … パレットの占有（床面）  ※ギャップも考慮して障害物化
    ばら積み：MaxRectsで格子に縛られず詰める
    """
    if z_layers <= 0 or bag_L <= 0 or bag_W <= 0:
        return []

    # クリアランスの考え方：
    # ・紙袋は (L+gap, W+gap) の膨らんだ矩形として扱う → 相互間隔 >= gap を保証
    # ・壁/パレットとの間隔 >= gap を保証するため、
    #   初期free領域を四方 gap/2 縮め、パレット障害物は四方 gap/2 ずつ膨らませて差し引く
    grow = gap_xy / 2.0
    x0, y0, LW, WW = truck_rect
    init_free = [(x0 + grow, y0 + grow, max(0.0, LW - 2*grow), max(0.0, WW - 2*grow))]

    # パレット障害物の差し引き（gap/2 膨張）
    free_rects = init_free
    for (px, py, pw, ph) in pallet_rects:
        obs = (px - grow, py - grow, pw + 2*grow, ph + 2*grow)
        new_list = []
        for fr in free_rects:
            if _rects_intersect(fr, obs):
                new_list.extend(_split_free_by_used(fr, obs))
            else:
                new_list.append(fr)
        free_rects = _prune_contained(new_list)

    placed2d = []
    inflated_w1, inflated_h1 = (bag_L + gap_xy, bag_W + gap_xy)

    remain = float('inf') if quota is None else int(max(0, quota))
    while remain > 0:
        res = _place_one(free_rects, inflated_w1, inflated_h1)  # 両向き自動
        if res is None:
            break
        used, free_rects = res
        placed2d.append(used)
        remain -= 1

    if not placed2d:
        return []

    # 2D→3D（z_layers 複製）。inflated → 実袋サイズへ戻す
    out = []
    for (ux, uy, uw, uh) in placed2d:
        # 実際の袋サイズ・原点（inflated内で左上に詰めているため、そのままでOK）
        if abs(uw - inflated_w1) < 1e-6 and abs(uh - inflated_h1) < 1e-6:
            Lb, Wb = bag_L, bag_W
        else:
            # 回転して置かれたケース
            Lb, Wb = bag_W, bag_L
        for iz in range(z_layers):
            out.append(((ux, uy, iz * H), (Lb, Wb, H)))
    return out

# =========（新規）カメラ状態の永続化（床ロール禁止） =========
def _apply_persisted_camera(fig, default_camera, uirev_key="keep-cam-v2"):
    cam = st.session_state.get("cam3d", default_camera)
    cam["up"] = dict(x=0, y=0, z=1)
    fig.update_layout(scene_camera=cam)
    fig.update_layout(uirevision=uirev_key)

def _extract_camera_from_relayout(relayout_dict):
    if not relayout_dict:
        return None
    cam = st.session_state.get("cam3d", dict(
        eye=dict(x=1.6, y=1.6, z=1.6),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic")
    ))
    def set_nested(d, keys, val):
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    updated = False
    for k, v in relayout_dict.items():
        if not k.startswith("scene.camera"):
            continue
        keys = k.split(".")
        if keys and keys[0] == "scene":
            keys = keys[1:]
        if len(keys) >= 2 and keys[0] == "camera" and keys[1] == "up":
            continue
        set_nested(cam, keys, v)
        updated = True
    cam["up"] = dict(x=0, y=0, z=1)
    return cam if updated else None

# ========= 入力UI =========
st.sidebar.header("① 紙袋の仕様（mm & kg）")
bag_L = st.sidebar.number_input("紙袋 長さ L (mm)", min_value=100, value=787, step=10)
bag_W = st.sidebar.number_input("紙袋 幅   W (mm)", min_value=100, value=465, step=10)
bag_H = st.sidebar.number_input("紙袋 高さ H (mm)", min_value=50,  value=150, step=5)
bag_weight = st.sidebar.number_input("紙袋 1袋あたり重量 (kg)", min_value=0.0, value=20.0, step=0.5)

st.sidebar.header("② パレットの仕様（mm & kg）")
pal_L = st.sidebar.number_input("パレット 長さ Lp (mm)", min_value=200, value=1600, step=10)
pal_W = st.sidebar.number_input("パレット 幅   Wp (mm)", min_value=200, value=1400, step=10)
pal_H = st.sidebar.number_input("パレット 高さ Hp (mm)", min_value=10,  value=150, step=5)
pallet_weight = st.sidebar.number_input("パレット 1枚あたり重量 (kg)", min_value=0.0, value=50.0, step=0.5)

st.sidebar.header("③ 荷台（トラック）内寸（mm）")
truck_L = st.sidebar.number_input("荷台 長さ Lt (mm)", min_value=1000, value=9600, step=50)
truck_W = st.sidebar.number_input("荷台 幅   Wt (mm)", min_value=1000, value=2400, step=50)
truck_H_in = st.sidebar.number_input("荷台 高さ Ht (mm)", min_value=1000, value=2600, step=50)
st.sidebar.caption("※ 計算上の有効高さ = 入力高さ − 200 mm（描画は入力高さどおり）")

st.sidebar.header("④ パレット上の積み付けパターン")
pattern = st.sidebar.selectbox(
    "パターン",
    ["5俵ばい（2-3繰り返し）", "8俵（偶数=端/中央, 奇数=縦2×4）", "自由入力（行パターン）"]
)
gap_xy = st.sidebar.number_input("袋と袋の隙間 (mm)", min_value=0, value=10, step=1)
if pattern == "5俵ばい（2-3繰り返し）":
    row_pattern_text = "2-3"
elif pattern == "自由入力（行パターン）":
    row_pattern_text = st.sidebar.text_input("行ごとの個数（例: 2-3-2-3）", value="2-3")
else:
    row_pattern_text = None
layers = st.sidebar.number_input("段数（例：8段/10段など）", min_value=1, value=10, step=1)
alternate_row_rotation = st.sidebar.checkbox(
    "（自由/5俵用）行ごとに90°回転を交互にする",
    value=True,
    disabled=(pattern=="8俵（偶数=端/中央, 奇数=縦2×4）")
)

st.sidebar.header("⑤ パレットの荷台積載")
enable_pallets = st.sidebar.checkbox("パレット積みを有効化する", value=True)
pallet_gap = st.sidebar.number_input("パレット同士の隙間 (mm)", min_value=0, value=10, step=1)

st.sidebar.header("⑤-2 パレットの回転設定")
allow_pallet_rotation = st.sidebar.checkbox("パレットの回転を許可する（必要なときだけ90°回転）", value=False)

st.sidebar.header("⑥ ばら積み")
enable_bara = st.sidebar.checkbox("ばら積みを有効化する", value=True)
gap_bara = st.sidebar.number_input("ばら積みの隙間 (mm)", min_value=0, value=10, step=1)

st.sidebar.header("⑦ 最大製品（袋）積載重量の上限")
enable_max_w = st.sidebar.checkbox("上限を有効化（パレット重量は除外）", value=False)
max_product_weight = st.sidebar.number_input("最大製品積載重量 (kg)", min_value=0.0, value=5000.0, step=100.0)

# ========= 表示オプション =========
HIDE_AXES = True
BG_COLOR = "#FFFFFF"
PAD_MM = 200.0
ORTHO = True
EDGE_ON = True
EDGE_WIDTH = 2

run_sim = st.sidebar.button("シミュレーション実行")

# ========= 本処理 =========
st.title("紙袋 × パレット × 荷台 3D積載シミュレーター（忠実寸法＋上限制御＋非重なり保証）")

if run_sim:
    # --- 有効範囲（計算のみ） ---
    h_eff = max(0, truck_H_in - 200)    # 高さ：-200（描画はフル）
    margin_xy = 20.0                    # 平面：-20（10mm×2）
    rect_x0, rect_y0 = margin_xy/2.0, margin_xy/2.0
    rect_L, rect_W = max(0.0, truck_L - margin_xy), max(0.0, truck_W - margin_xy)

    # 上限（袋数）計算
    INF = 10**12
    max_bags_allowed = (int(max_product_weight // bag_weight) if (enable_max_w and bag_weight > 0) else INF)

    # === パレット1枚あたりの袋配置（高さ制限を反映） ===
    if enable_pallets:
        if pattern == "8俵（偶数=端/中央, 奇数=縦2×4）":
            bag_boxes_on_pallet, actual_layers, per_layer = palletize_8hyo_even_positions(
                pal_L, pal_W, pal_H, bag_L, bag_W, bag_H, gap_xy, layers, h_eff
            )
        elif pattern == "5俵ばい（2-3繰り返し）":
            bag_boxes_on_pallet, actual_layers, per_layer = palletize_5hyo_edges_center(
                pal_L, pal_W, pal_H, bag_L, bag_W, bag_H, gap_xy, layers, alternate_row_rotation, h_eff
            )
        else:
            rows_pattern = parse_row_pattern(row_pattern_text) if row_pattern_text else None
            if not rows_pattern:
                st.error("行パターンが不正です。例：2-3-2-3"); st.stop()
            bag_boxes_on_pallet, actual_layers, per_layer = palletize_generic(
                pal_L, pal_W, pal_H, bag_L, bag_W, bag_H, gap_xy, rows_pattern, layers, alternate_row_rotation, h_eff
            )
    else:
        bag_boxes_on_pallet, actual_layers, per_layer = [], 0, None

    # パレットのフットプリント（袋占有を考慮）
    if enable_pallets and bag_boxes_on_pallet:
        xs, ys = [], []
        for (o, s) in bag_boxes_on_pallet:
            x,y,_ = o; L,W,_ = s
            xs += [x, x+L]; ys += [y, y+W]
        occ_L, occ_W = (max(xs)-min(xs), max(ys)-min(ys))
        pallet_foot_L = max(pal_L, occ_L)
        pallet_foot_W = max(pal_W, occ_W)
    else:
        pallet_foot_L = pal_L
        pallet_foot_W = pal_W

    # === パレットレイアウト（ギャップ込み + 非重なり保証） ===
    pallet_origins = []
    pal_L_draw = pal_W_draw = 0
    pallet_boxes3d = []
    pal_rotated = False

    if enable_pallets and (pallet_foot_L > 0 and pallet_foot_W > 0):
        cols0, rows0 = grid_fit_with_gap(rect_L, rect_W, pallet_foot_L, pallet_foot_W, pallet_gap)
        cols1, rows1 = grid_fit_with_gap(rect_L, rect_W, pallet_foot_W, pallet_foot_L, pallet_gap)
        if allow_pallet_rotation and (cols1*rows1 > cols0*rows0):
            pal_L_draw, pal_W_draw = pallet_foot_W, pallet_foot_L
            cols, rows = cols1, rows1
            pal_rotated = True
        else:
            pal_L_draw, pal_W_draw = pallet_foot_L, pallet_foot_W
            cols, rows = cols0, rows0
            pal_rotated = False

        used_L_full = cols*(pal_L_draw + pallet_gap) - (pallet_gap if cols>0 else 0)
        used_W_full = rows*(pal_W_draw + pallet_gap) - (pallet_gap if rows>0 else 0)

        per_pallet_capacity = len(bag_boxes_on_pallet) if bag_boxes_on_pallet else 0
        if per_pallet_capacity == 0:
            need_pallets = 0
        else:
            est_needed = math.ceil(min(max_bags_allowed, INF) / per_pallet_capacity)
            need_pallets = min(cols*rows, est_needed)

        def candidate_slots(anchor_left, anchor_front, n_cols, n_rows):
            if need_pallets == 0:
                return []
            use_cols = min(n_cols, max(1, math.ceil(need_pallets / max(1, n_rows))))
            use_rows = min(n_rows, max(1, math.ceil(need_pallets / max(1, use_cols))))
            used_L = use_cols*(pal_L_draw + pallet_gap) - (pallet_gap if use_cols>0 else 0)
            used_W = use_rows*(pal_W_draw + pallet_gap) - (pallet_gap if use_rows>0 else 0)
            off_x = rect_x0 if anchor_left else (rect_x0 + rect_L - used_L)
            off_y = rect_y0 if anchor_front else (rect_y0 + rect_W - used_W)
            slots = [(off_x + c*(pal_L_draw + pallet_gap),
                      off_y + r*(pal_W_draw + pallet_gap), 0.0)
                     for r in range(use_rows) for c in range(use_cols)]
            return slots

        def filter_slots_non_overlap(slots):
            kept_origins = []
            kept_boxes3d = []
            pal_H_total = pal_H + actual_layers*bag_H if enable_pallets else pal_H
            for (sx,sy,sz) in slots:
                cand_o = (sx,sy,0.0)
                cand_s = (pal_L_draw, pal_W_draw, pal_H_total if pal_H_total>0 else h_eff)
                if any(aabb_overlap_strict(cand_o, cand_s, o2, s2) for (o2,s2) in kept_boxes3d):
                    continue
                kept_origins.append((sx,sy,sz))
                kept_boxes3d.append((cand_o, cand_s))
                if len(kept_origins) >= need_pallets:
                    break
            return kept_origins, kept_boxes3d

        if enable_bara:
            best = None
            for anchor_left in [True, False]:
                for anchor_front in [True, False]:
                    slots_raw = candidate_slots(anchor_left, anchor_front, cols, rows)
                    slots, boxes3d = filter_slots_non_overlap(slots_raw)
                    # スコアは「この配置で後のばら積みがどれだけ詰めるか」で評価
                    pallet_rects_eval = [(x, y, pal_L_draw, pal_W_draw) for (x,y,_) in slots]
                    z_layers_tmp = compute_bara_z_layers(h_eff, bag_H)
                    packed_tmp = pack_bara_maxrects(
                        (rect_x0, rect_y0, rect_L, rect_W),
                        pallet_rects_eval,
                        bag_L, bag_W, gap_bara, bag_H, z_layers_tmp, quota=None
                    )
                    score = len(packed_tmp)
                    if (best is None) or (score > best[0]):
                        best = (score, slots, boxes3d)
            pallet_origins = best[1] if best else []
            pallet_boxes3d = best[2] if best else []
        else:
            off_x = rect_x0 + max(0.0, (rect_L - used_L_full)/2.0)
            off_y = rect_y0 + max(0.0, (rect_W - used_W_full)/2.0)
            use_cols = min(cols, max(1, math.ceil(need_pallets / max(1, rows))))
            use_rows = min(rows, max(1, math.ceil(need_pallets / max(1, use_cols))))
            slots_raw = [(off_x + c*(pal_L_draw + pallet_gap),
                          off_y + r*(pal_W_draw + pallet_gap), 0.0)
                         for r in range(use_rows) for c in range(use_cols)]
            pallet_origins, pallet_boxes3d = filter_slots_non_overlap(slots_raw)

    # === パレット上の袋（上限＋非重なり保証） ===
    selected_pallet_bags = []
    remain_quota = max_bags_allowed
    if enable_pallets and bag_boxes_on_pallet and len(pallet_origins)>0 and remain_quota>0:
        per_pallet = len(bag_boxes_on_pallet)
        bag_boxes3d = []

        for slot in pallet_origins:
            if remain_quota <= 0: break
            take = min(per_pallet, remain_quota)
            for i in range(take):
                (o,s) = bag_boxes_on_pallet[i]
                (bx,by,bz) = o
                (Lb_local, Wb_local, Hb_local) = s
                if pal_rotated:
                    cand_o = (slot[0] + by, slot[1] + bx, slot[2] + bz)
                    cand_s = (Wb_local, Lb_local, Hb_local)
                else:
                    cand_o = (slot[0] + bx, slot[1] + by, slot[2] + bz)
                    cand_s = (Lb_local, Wb_local, Hb_local)
                conflict = any(aabb_overlap_strict(cand_o, cand_s, o2, s2) for (o2,s2) in bag_boxes3d)
                if not conflict and pallet_boxes3d:
                    conflict = any(
                        aabb_overlap_strict(cand_o, cand_s, po, ps)
                        for (po,ps) in pallet_boxes3d
                        if not (abs(po[0]-slot[0])<EPS and abs(po[1]-slot[1])<EPS)
                    )
                if conflict:
                    continue
                selected_pallet_bags.append((cand_o, cand_s))
                bag_boxes3d.append((cand_o, cand_s))
                remain_quota -= 1
                if remain_quota <= 0:
                    break

        used_origins = set((int(o[0]), int(o[1])) for (o,_) in selected_pallet_bags)
        pallet_origins = [p for p in pallet_origins if (int(p[0]),int(p[1])) in used_origins]
        pallet_boxes3d = [pb for pb in pallet_boxes3d if (int(pb[0][0]),int(pb[0][1])) in used_origins] if pallet_boxes3d else []

    # === ばら積み（MaxRects: パレットに引っ張られない連続面詰め） ===
    selected_bara_bags = []
    if enable_bara and remain_quota > 0:
        z_layers = compute_bara_z_layers(h_eff, bag_H)
        if z_layers > 0:
            pallet_rects_for_bara = [(x, y,
                                      (pal_L_draw if pal_L_draw>0 else pal_L),
                                      (pal_W_draw if pal_W_draw>0 else pal_W))
                                     for (x,y,_) in (pallet_origins or [])]
            placed_list = pack_bara_maxrects(
                (rect_x0, rect_y0, rect_L, rect_W),
                pallet_rects_for_bara,
                bag_L, bag_W, gap_bara, bag_H, z_layers,
                quota=remain_quota
            )
            selected_bara_bags += placed_list
            remain_quota -= len(placed_list)

    # === （最終ガード）有効体積内・去重 ===
    def within_rect(o, s):
        (x,y,z) = o; (L,W,H) = s
        return (rect_x0 - EPS <= x and x+L <= rect_x0+rect_L + EPS and
                rect_y0 - EPS <= y and y+W <= rect_y0+rect_W + EPS and
                0 - EPS <= z and z+H <= h_eff + EPS)

    selected_pallet_bags = [bw for bw in selected_pallet_bags if within_rect(*bw)]
    selected_bara_bags = [bw for bw in selected_bara_bags if within_rect(*bw)]

    def bbox_key(o, s):
        (x,y,z) = o; (L,W,H) = s
        return (round(x), round(y), round(z), round(L), round(W), round(H))
    seen = set(); dedup = []
    for bw in selected_pallet_bags + selected_bara_bags:
        k = bbox_key(*bw)
        if k not in seen:
            seen.add(k); dedup.append(bw)
    selected_pallet_bags = [bw for bw in dedup if bw in selected_pallet_bags]
    selected_bara_bags = [bw for bw in dedup if bw not in selected_pallet_bags]

    # === 集計 ===
    bags_on_pallet_kept = len(selected_pallet_bags)
    bags_bara_kept = len(selected_bara_bags)
    total_bags_kept = bags_on_pallet_kept + bags_bara_kept
    total_weight_bags = total_bags_kept * bag_weight
    total_weight_pallets = (len(pallet_origins) if enable_pallets else 0) * pallet_weight
    total_weight = total_weight_bags + total_weight_pallets
    total_vol_m3 = total_bags_kept * mm3_to_m3(bag_L*bag_W*bag_H) + (len(pallet_origins) if enable_pallets else 0) * mm3_to_m3(pal_L*pal_W*pal_H)

    # ========= 上部メトリクス =========
    st.markdown("### 集計（概要）")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("総袋数", f"{total_bags_kept:,} 袋")
    with m2:
        st.metric("パレット枚数", f"{len(pallet_origins):,} 枚")
    with m3:
        st.metric("製品重量（袋）", f"{total_weight_bags:,.0f} kg")
    with m4:
        st.metric("総重量（袋＋パレット）", f"{total_weight:,.0f} kg")

    st.caption(f"・ばら積み: {bags_bara_kept} 袋 / パレット上: {bags_on_pallet_kept} 袋 / 体積合計: {total_vol_m3:.3f} m³")

    # ========= 詳細（左:計画 / 右:採用結果） =========
    c1, c2, _ = st.columns(3)
    with c1:
        st.subheader("パレット積載（1枚あたりの計画）")
        st.write(f"- パターン: {pattern}")
        st.write(f"- 段数（入力）: {layers} 段 / 実適用: {actual_layers if enable_pallets else 0} 段（高さ制限）")
        if enable_pallets and pattern != "8俵（偶数=端/中央, 奇数=縦2×4）":
            st.write(f"- 1層あたり（参考）: {per_layer if per_layer is not None else '-'} 袋")
        if enable_pallets:
            st.write(f"- 1パレット最大: **{len(bag_boxes_on_pallet)} 袋**（高さ制限反映）")
    with c2:
        st.subheader("採用結果（重量上限を反映）")
        if enable_pallets:
            st.write(f"- 使用パレット: **{len(pallet_origins)} 枚**（空パレット無し）")
            st.write(f"- パレット採用袋数: **{bags_on_pallet_kept} 袋**")
        if enable_bara:
            st.write(f"- ばら積み採用袋数: **{bags_bara_kept} 袋**")
        if enable_max_w:
            st.write(f"- 最大製品重量設定: {max_product_weight:,.0f} kg（袋のみ）")

    st.markdown("---")
    st.subheader("3Dビュー（荷台=半透明 / 回転中心=荷台中央 / 底面固定 / 直交投影 / データ等倍）")

    fig = go.Figure()

    # 荷台（半透明 + 輪郭）
    truck_mesh_origin = center_origin((0,0,0), truck_L, truck_W, truck_H_in)
    fig.add_trace(cuboid_mesh(
        origin=truck_mesh_origin, size=(truck_L, truck_W, truck_H_in),
        color="#B0B0B0", opacity=0.18, name="荷台", showlegend=True
    ))
    truck_edges = edges_group_trace(
        [(truck_mesh_origin, (truck_L, truck_W, truck_H_in))],
        color="#888888", width=1, name="荷台枠", showlegend=False
    )
    if truck_edges: fig.add_trace(truck_edges)

    # 有効平面ガイド（描画のみ）
    guide_h = 2.0
    for o,s in [
        ((rect_x0, rect_y0, 0), (rect_L, guide_h, guide_h)),
        ((rect_x0, rect_y0+rect_W-guide_h, 0), (rect_L, guide_h, guide_h)),
        ((rect_x0, rect_y0, 0), (guide_h, rect_W, guide_h)),
        ((rect_x0+rect_L-guide_h, rect_y0, 0), (guide_h, rect_W, guide_h)),
    ]:
        fig.add_trace(cuboid_mesh(
            origin=center_origin(o, truck_L, truck_W, truck_H_in),
            size=s, color="#DDDDDD", opacity=1.0))

    pallet_color = "#C69C6D"
    COLOR_ODD  = "#FFF3B0"
    COLOR_EVEN = "#F6A5A5"
    BARA_COLOR = "#7B9E46"

    # パレット本体
    if enable_pallets and pallet_origins:
        pal_boxes_disp = []
        for (px,py,pz) in pallet_origins:
            pal_boxes_disp.append(
                (center_origin((px,py,pz), truck_L, truck_W, truck_H_in),
                 (pal_L_draw if pal_L_draw>0 else pal_L,
                  (pal_W_draw if pal_W_draw>0 else pal_W),
                  pal_H))
            )
        pal_mesh = cuboid_mesh_group(pal_boxes_disp, color=pallet_color, name="パレット", showlegend=True)
        if pal_mesh: fig.add_trace(pal_mesh)
        pal_edges = edges_group_trace(pal_boxes_disp, color="#333333", width=2)
        if pal_edges: fig.add_trace(pal_edges)

    def infer_layer_index_from_z(z, Hp, Hb):
        if Hb <= 0:
            return 0
        return int(round((z - Hp) / Hb))

    odd_boxes = []
    even_boxes = []
    for (o, s) in selected_pallet_bags:
        x,y,z = o
        L,W,H = s
        lidx = infer_layer_index_from_z(z, pal_H, bag_H)
        target = odd_boxes if ((lidx + 1) % 2 == 1) else even_boxes
        target.append((center_origin((x,y,z), truck_L, truck_W, truck_H_in), (L,W,H)))

    bara_group = [(center_origin(o, truck_L, truck_W, truck_H_in), s) for (o,s) in selected_bara_bags]

    if odd_boxes:
        mesh_odd = cuboid_mesh_group(odd_boxes, color=COLOR_ODD, name="紙袋（奇数段・パレット）", showlegend=True)
        if mesh_odd: fig.add_trace(mesh_odd)
        if EDGE_ON:
            ed = edges_group_trace(odd_boxes, color="#000000", width=EDGE_WIDTH)
            if ed: fig.add_trace(ed)

    if even_boxes:
        mesh_even = cuboid_mesh_group(even_boxes, color=COLOR_EVEN, name="紙袋（偶数段・パレット）", showlegend=True)
        if mesh_even: fig.add_trace(mesh_even)
        if EDGE_ON:
            ed = edges_group_trace(even_boxes, color="#000000", width=EDGE_WIDTH)
            if ed: fig.add_trace(ed)

    if bara_group:
        mesh_bara = cuboid_mesh_group(bara_group, color=BARA_COLOR, name="紙袋（ばら）", showlegend=True)
        if mesh_bara: fig.add_trace(mesh_bara)
        if EDGE_ON:
            ed = edges_group_trace(bara_group, color="#000000", width=EDGE_WIDTH)
            if ed: fig.add_trace(ed)

    xr = (-(truck_L/2 + PAD_MM),  (truck_L/2 + PAD_MM))
    yr = (-(truck_W/2 + PAD_MM),  (truck_W/2 + PAD_MM))
    zr = (-(truck_H_in/2 + PAD_MM), (truck_H_in/2 + PAD_MM))
    scene_aspectmode = "data"

    default_camera = dict(
        eye=dict(x=1.6, y=1.6, z=1.6),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="orthographic") if ORTHO else dict(type="perspective"),
    )

    fig.update_scenes(
        xaxis_title="X (mm) [長手]", yaxis_title="Y (mm) [幅]", zaxis_title="Z (mm) [高さ]",
        aspectmode=scene_aspectmode,
        xaxis=dict(range=xr, showgrid=not HIDE_AXES, zeroline=not HIDE_AXES, visible=not HIDE_AXES),
        yaxis=dict(range=yr, showgrid=not HIDE_AXES, zeroline=not HIDE_AXES, visible=not HIDE_AXES),
        zaxis=dict(range=zr, showgrid=not HIDE_AXES, zeroline=not HIDE_AXES, visible=not HIDE_AXES),
        bgcolor=BG_COLOR
    )

    _apply_persisted_camera(fig, default_camera, uirev_key="keep-cam-v2")

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode="turntable",
    )

    events = plotly_events(
        fig,
        click_event=False,
        hover_event=False,
        select_event=False,
        key="plot3d",
        override_width="100%",
        override_height=None,
    )

    def _handle_events(ev):
        if isinstance(ev, dict):
            cam = _extract_camera_from_relayout(ev)
            if cam is not None:
                st.session_state["cam3d"] = cam

    if isinstance(events, list):
        for ev in events:
            _handle_events(ev)
    else:
        _handle_events(events)

    st.caption("操作：左ドラッグ=回転（床固定） / 右ドラッグ=平行移動（パン） / ホイール=ズーム / ダブルクリック=リセット")

else:
    st.write("左のパラメータを設定し、**「シミュレーション実行」**を押してください。")
    st.caption("パレット積みをOFFにすると、ばら積みのみのシミュレーションが可能です。")
