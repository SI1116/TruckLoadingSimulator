# app.py
# 依存: pip install streamlit numpy plotly

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go

st.set_page_config(page_title="紙袋×パレット×荷台 3D積載シミュレーター（8俵・端合わせ/センター・ばら安全）", layout="wide")

# ========= ユーティリティ =========
def mm3_to_m3(v_mm3: float) -> float:
    return v_mm3 / 1e9

def grid_fit_with_gap(cap_len, cap_wid, item_len, item_wid, gap):
    """パレット同士の隙間(gap)込みのグリッド充填"""
    cols = int((cap_len + gap) // (item_len + gap))
    rows = int((cap_wid + gap) // (item_wid + gap))
    return max(0, cols), max(0, rows)

def parse_row_pattern(text: str):
    try:
        parts = [p.strip() for p in text.split('-') if p.strip()!='']
        nums = [int(p) for p in parts]
        if all(n>0 for n in nums):
            return nums
    except:
        pass
    return None

# ---- Plotly Mesh3d ----
def cuboid_mesh(origin, size, color, opacity=1.0, name=None, showlegend=False):
    x0,y0,z0 = origin; L,W,H = size
    x = np.array([x0, x0+L, x0+L, x0,   x0,   x0+L, x0+L, x0])
    y = np.array([y0, y0,   y0+W, y0+W, y0,   y0,   y0+W, y0+W])
    z = np.array([z0, z0,   z0,   z0,   z0+H, z0+H, z0+H, z0+H])
    I = [0,1,2, 0,2,3,  4,5,6, 4,6,7,  0,1,5, 0,5,4,  1,2,6, 1,6,5,  2,3,7, 2,7,6,  3,0,4, 3,4,7]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=I[0::3], j=I[1::3], k=I[2::3],
        color=color, opacity=opacity, flatshading=True,
        name=name or "", showlegend=showlegend
    )

# === 表示用：荷台中心(0,0,0)に平行移動 ===
def center_origin(origin, truck_L, truck_W, truck_H_draw):
    cx, cy, cz = truck_L/2.0, truck_W/2.0, truck_H_draw/2.0
    x, y, z = origin
    return (x - cx, y - cy, z - cz)

# ========= パレット上：一般パターン（袋間のみギャップ） =========
def palletize_generic(Lp, Wp, Hp, Lb, Wb, Hb, gap_xy, rows_pattern, layers, alternate_row_rotation):
    bag_boxes = []
    z0 = Hp
    per_layer_count = 0
    for l in range(layers):
        y = 0.0
        r = 0
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
            if y_next - gap_xy + bag_W > Wp + 1e-6:
                break
            y = y_next
            r += 1
    return bag_boxes, layers, per_layer_count // max(1, layers)

# ========= 8俵（偶数段：左右=端合わせ/中央=センター） =========
def palletize_8hyo_even_positions(Lp, Wp, Hp, Lb, Wb, Hb, gap_xy, layers):
    """
    奇数段：縦向き 2行×4列（袋間のみ gap）。X=中央寄せ、Y=端合わせ（上下ピタ）に近い表示。
    偶数段：
      - 左右の横向き3段： y_top=0（上端ピタ）, y_mid=(Wp - Wb)/2（中央）, y_bot=Wp - Wb（下端ピタ）
        Xは左列x=0、右列x=Lp-Lb（端ピタ）。袋間のみ gap を考慮（ただし1列1個なのでX方向はギャップ無関係）。
      - 中央の縦向き2段： Yは (Wp - (2*Lb + gap))/2 から Lb と gap を挟んで2個、Xは (Lp - Wb)/2（センター）。
    端との隙間は作らない。袋間のみ gap_xy を適用。
    """
    boxes = []
    z0 = Hp
    SIZE_TATE = (Wb, Lb, Hb)  # 見た目縦長
    SIZE_YOKO = (Lb, Wb, Hb)  # 見た目横長

    def center_start(total, needed):
        return max(0.0, (total - needed) / 2.0)

    for l in range(layers):
        z = z0 + l*Hb
        odd = ((l+1) % 2 == 1)

        if odd:
            # 奇数段：縦2×4（XはWbピッチでセンター、YはLbピッチで端合わせ）
            x_pitch = Wb + gap_xy
            y_pitch = Lb + gap_xy
            cols = min(4, int((Lp + gap_xy) // x_pitch))
            rows = min(2, int((Wp + gap_xy) // y_pitch))
            need_x = cols*x_pitch - (gap_xy if cols>0 else 0.0)
            x0 = center_start(Lp, need_x)
            # 端合わせ（上下ピタ）：上=0, 下=Wp-Lb（rows=2の前提）
            if rows >= 2:
                ys = [0.0, Wp - Lb]
            elif rows == 1:
                ys = [center_start(Wp, Lb)]
            else:
                ys = []
            for y in ys:
                for c in range(cols):
                    x = x0 + c*x_pitch
                    if x + Wb <= Lp + 1e-6 and y + Lb <= Wp + 1e-6:
                        boxes.append(((x, y, z), SIZE_TATE))
        else:
            # 偶数段：左右=横3段（上/中/下=端/中央/端）、中央=縦2段（センター）
            # 左右横列
            x_left = 0.0
            x_right = Lp - Lb
            y_top = 0.0
            y_mid = max(0.0, (Wp - Wb)/2.0)
            y_bot = Wp - Wb
            for y in [y_top, y_mid, y_bot]:
                if y + Wb <= Wp + 1e-6:
                    if x_left + Lb <= Lp + 1e-6:
                        boxes.append(((x_left, y, z), SIZE_YOKO))
                    if x_right + Lb <= Lp + 1e-6:
                        boxes.append(((x_right, y, z), SIZE_YOKO))
            # 中央縦列（2個＋袋間gapをYセンターに）
            need_y_center = 2*Lb + gap_xy
            y_c1 = center_start(Wp, need_y_center)
            y_c2 = y_c1 + Lb + gap_xy
            x_center = center_start(Lp, Wb)
            if y_c1 + Lb <= Wp + 1e-6 and x_center + Wb <= Lp + 1e-6:
                boxes.append(((x_center, y_c1, z), SIZE_TATE))
            if y_c2 + Lb <= Wp + 1e-6 and x_center + Wb <= Lp + 1e-6:
                boxes.append(((x_center, y_c2, z), SIZE_TATE))

    return boxes, layers, 8

# ========= ばら積み：安全版（重複/はみ出しなし） =========
def pack_rect_grid(x0, y0, Lr, Wr, aL, aW, gap, H, z_layers):
    """矩形(Lr×Wr)に aL×aW の袋をグリッド充填（袋間のみ gap、端隙間0）。"""
    boxes = []
    if Lr <= 0 or Wr <= 0 or aL <= 0 or aW <= 0 or z_layers <= 0:
        return boxes
    cols = int((Lr + gap) // (aL + gap))
    rows = int((Wr + gap) // (aW + gap))
    if cols <= 0 or rows <= 0:
        return boxes
    for iz in range(z_layers):
        z = iz * H
        for r in range(rows):
            for c in range(cols):
                x = x0 + c*(aL + gap)
                y = y0 + r*(aW + gap)
                # 端はみ出し防止
                if (x + aL <= x0 + Lr + 1e-6) and (y + aW <= y0 + Wr + 1e-6):
                    boxes.append(((x, y, z), (aL, aW, H)))
    return boxes

def fill_rect_with_bags_best_safe(x0, y0, Lr, Wr, Lb, Wb, gap, H, layers_z):
    """
    メイン：Lb×Wb を第一優先で敷き、その使用領域の
      右帯(remL×usedW)、下帯(usedL×remW)、右下コーナー(remL×remW) を
      すべて副向きで埋める。領域分割で**重複を防止**。
    """
    if Lr <= 0 or Wr <= 0 or layers_z <= 0:
        return []

    # 1) メイン向き（Lb×Wb）
    cols_p = int((Lr + gap) // (Lb + gap))
    rows_p = int((Wr + gap) // (Wb + gap))
    usedL_p = cols_p*Lb + (cols_p-1)*gap if cols_p>0 else 0
    usedW_p = rows_p*Wb + (rows_p-1)*gap if rows_p>0 else 0

    boxes = []
    boxes += pack_rect_grid(x0, y0, usedL_p, usedW_p, Lb, Wb, gap, H, layers_z)

    remL = max(0.0, Lr - usedL_p)
    remW = max(0.0, Wr - usedW_p)

    # 2) 右帯（remL × usedW_p）を副向きで
    boxes += pack_rect_grid(x0 + usedL_p, y0, remL, usedW_p, Wb, Lb, gap, H, layers_z)
    # 3) 下帯（usedL_p × remW）
    boxes += pack_rect_grid(x0, y0 + usedW_p, usedL_p, remW, Wb, Lb, gap, H, layers_z)
    # 4) 右下コーナー（remL × remW）
    boxes += pack_rect_grid(x0 + usedL_p, y0 + usedW_p, remL, remW, Wb, Lb, gap, H, layers_z)

    # もう一方の優先（Wb×Lb）での総数も試し、より多い方を採用
    boxes_alt = []
    cols_q = int((Lr + gap) // (Wb + gap))
    rows_q = int((Wr + gap) // (Lb + gap))
    usedL_q = cols_q*Wb + (cols_q-1)*gap if cols_q>0 else 0
    usedW_q = rows_q*Lb + (rows_q-1)*gap if rows_q>0 else 0
    boxes_alt += pack_rect_grid(x0, y0, usedL_q, usedW_q, Wb, Lb, gap, H, layers_z)
    remL_q = max(0.0, Lr - usedL_q)
    remW_q = max(0.0, Wr - usedW_q)
    boxes_alt += pack_rect_grid(x0 + usedL_q, y0, remL_q, usedW_q, Lb, Wb, gap, H, layers_z)
    boxes_alt += pack_rect_grid(x0, y0 + usedW_q, usedL_q, remW_q, Lb, Wb, gap, H, layers_z)
    boxes_alt += pack_rect_grid(x0 + usedL_q, y0 + usedW_q, remL_q, remW_q, Lb, Wb, gap, H, layers_z)

    return boxes if len(boxes) >= len(boxes_alt) else boxes_alt

def compute_bara_z_layers(h_eff, bag_H):
    return 0 if bag_H <= 0 else max(0, int(h_eff // bag_H))

# ========= 入力UI =========
st.sidebar.header("① 紙袋の仕様（mm & kg）")
bag_L = st.sidebar.number_input("紙袋 長さ L (mm)", min_value=100, value=600, step=10)
bag_W = st.sidebar.number_input("紙袋 幅   W (mm)", min_value=100, value=400, step=10)
bag_H = st.sidebar.number_input("紙袋 高さ H (mm)", min_value=50,  value=120, step=5)
bag_weight = st.sidebar.number_input("紙袋 1袋あたり重量 (kg)", min_value=0.0, value=20.0, step=0.5)

st.sidebar.header("② パレットの仕様（mm & kg）")
pal_L = st.sidebar.number_input("パレット 長さ Lp (mm)", min_value=200, value=1100, step=10)
pal_W = st.sidebar.number_input("パレット 幅   Wp (mm)", min_value=200, value=1100, step=10)
pal_H = st.sidebar.number_input("パレット 高さ Hp (mm)", min_value=10,  value=120, step=5)
pallet_weight = st.sidebar.number_input("パレット 1枚あたり重量 (kg)", min_value=0.0, value=50.0, step=0.5)

st.sidebar.header("③ 荷台（トラック）内寸（mm）")
truck_L = st.sidebar.number_input("荷台 長さ Lt (mm)", min_value=1000, value=9600, step=50)
truck_W = st.sidebar.number_input("荷台 幅   Wt (mm)", min_value=1000, value=2400, step=50)
truck_H_in = st.sidebar.number_input("荷台 高さ Ht (mm)", min_value=1000, value=2600, step=50)
st.sidebar.caption("※ 計算上の有効高さ = 入力高さ − 200 mm（描画は入力高さどおり）")

st.sidebar.header("④ パレット上の積み付けパターン")
pattern = st.sidebar.selectbox(
    "パターン",
    ["5俵ばい（3-2繰り返し）", "6俵ばい（3-3繰り返し）", "8俵（偶数=端/中央, 奇数=縦2×4）", "自由入力（行パターン）"]
)
gap_xy = st.sidebar.number_input("袋と袋の隙間 (mm)", min_value=0, value=10, step=1)
if pattern == "5俵ばい（3-2繰り返し）":
    row_pattern_text = "3-2"
elif pattern == "6俵ばい（3-3繰り返し）":
    row_pattern_text = "3-3"
elif pattern == "自由入力（行パターン）":
    row_pattern_text = st.sidebar.text_input("行ごとの個数（例: 3-2-3-2）", value="3-2")
else:
    row_pattern_text = None
layers = st.sidebar.number_input("段数（例：8段/10段など）", min_value=1, value=10, step=1)
alternate_row_rotation = st.sidebar.checkbox("（自由/5俵/6俵用）行ごとに90°回転を交互にする", value=True, disabled=(pattern=="8俵（偶数=端/中央, 奇数=縦2×4）"))

st.sidebar.header("⑤ パレットの荷台積載")
enable_pallets = st.sidebar.checkbox("パレット積みを有効化する", value=True)
pallet_stack_layers = st.sidebar.number_input("パレットの縦積み段数（通常は1）", min_value=1, value=1, step=1, disabled=not enable_pallets)
pallet_gap = st.sidebar.number_input("パレット同士の隙間 (mm)", min_value=0, value=10, step=1)

st.sidebar.header("⑥ ばら積み")
enable_bara = st.sidebar.checkbox("ばら積みを有効化する", value=True)
gap_bara = st.sidebar.number_input("ばら積みの隙間 (mm)", min_value=0, value=10, step=1)

st.sidebar.header("⑦ 最大製品（袋）積載重量の上限")
enable_max_w = st.sidebar.checkbox("上限を有効化（パレット重量は除外）", value=False)
max_product_weight = st.sidebar.number_input("最大製品積載重量 (kg)", min_value=0.0, value=5000.0, step=100.0)

st.sidebar.header("⑧ 表示オプション")
hide_axes = st.sidebar.checkbox("軸・グリッドを非表示", value=True)
bg_color = st.sidebar.color_picker("背景色", "#FFFFFF")
visual_gap_height_mm = st.sidebar.number_input("高さ方向の見かけ隙間 (mm)", min_value=0.0, value=1.0, step=0.5)
pad_pct = st.sidebar.slider("周囲の余白（%）", 0, 20, 8)
aspect_mode = st.sidebar.selectbox("縦横比モード", ["実寸（manual）", "データ自動（data）", "立方体（cube）"])
view_preset = st.sidebar.selectbox("初期視点", ["斜め全体（推奨）", "俯瞰（トップ）", "正面（後扉側）", "側面（左側）"])
ortho = st.sidebar.checkbox("直交投影（遠近感なし）", value=False)

run_sim = st.sidebar.button("シミュレーション実行")

# ========= 本処理 =========
st.title("紙袋 × パレット × 荷台 3D積載シミュレーター（8俵・端合わせ/センター・ばら安全）")

if run_sim:
    # --- 有効範囲（計算のみ） ---
    h_eff = max(0, truck_H_in - 200)
    margin_xy = 20.0
    rect_x0, rect_y0 = margin_xy/2.0, margin_xy/2.0
    rect_L, rect_W = max(0.0, truck_L - margin_xy), max(0.0, truck_W - margin_xy)

    # === パレット1枚あたりの袋配置 ===
    if enable_pallets:
        if pattern == "8俵（偶数=端/中央, 奇数=縦2×4）":
            bag_boxes_on_pallet, actual_layers, per_layer = palletize_8hyo_even_positions(
                pal_L, pal_W, pal_H, bag_L, bag_W, bag_H, gap_xy, layers
            )
        else:
            rows_pattern = parse_row_pattern(row_pattern_text) if row_pattern_text else None
            if not rows_pattern and pattern != "8俵（偶数=端/中央, 奇数=縦2×4）":
                st.error("行パターンが不正です。例：3-2-3-2"); st.stop()
            bag_boxes_on_pallet, actual_layers, per_layer = palletize_generic(
                pal_L, pal_W, pal_H, bag_L, bag_W, bag_H, gap_xy, rows_pattern, layers, alternate_row_rotation
            )

        if len(bag_boxes_on_pallet) == 0:
            st.warning("このパレットサイズ・行パターンでは1袋も積載できません。ばら積みのみをお試しください。")
    else:
        bag_boxes_on_pallet, actual_layers, per_layer = [], 0, None

    # パレット総高さ
    pallet_total_H = pal_H + layers * bag_H if enable_pallets else 0
    if enable_pallets and pallet_total_H > h_eff:
        st.warning(f"注意：パレット総高さ {int(pallet_total_H)} mm が有効高さ {int(h_eff)} mm を超えています。")

    # パレットのフットプリント（袋占有考慮）
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

    # === パレットレイアウト（ギャップ込み） ===
    pallet_origins = []
    layout_cols = layout_rows = 0
    pal_L_draw = pal_W_draw = 0
    if enable_pallets and (pallet_foot_L > 0 and pallet_foot_W > 0):
        cols0, rows0 = grid_fit_with_gap(rect_L, rect_W, pallet_foot_L, pallet_foot_W, pallet_gap)
        cols1, rows1 = grid_fit_with_gap(rect_L, rect_W, pallet_foot_W, pallet_foot_L, pallet_gap)
        if cols1*rows1 > cols0*rows0:
            orientation = 90
            pal_L_draw, pal_W_draw = pallet_foot_W, pallet_foot_L
            cols, rows = cols1, rows1
        else:
            orientation = 0
            pal_L_draw, pal_W_draw = pallet_foot_L, pallet_foot_W
            cols, rows = cols0, rows0
        layout_cols, layout_rows = cols, rows

        used_L_full = cols*(pal_L_draw + pallet_gap) - (pallet_gap if cols>0 else 0)
        used_W_full = rows*(pal_W_draw + pallet_gap) - (pallet_gap if rows>0 else 0)

        # 採用可能パレット枚数（重量上限前）
        per_pallet_capacity = len(bag_boxes_on_pallet) if bag_boxes_on_pallet else 0
        max_bags_allowed = int(max_product_weight // bag_weight) if (enable_max_w and bag_weight>0) else 10**9
        need_pallets = min(cols*rows, math.ceil(min(max_bags_allowed, 10**9) / max(1, per_pallet_capacity))) if (enable_pallets and per_pallet_capacity>0) else 0

        # 片寄せ評価（ばらが有効なら4アンカーから選択）
        def candidate_slots(anchor_left, anchor_front, n_cols, n_rows):
            if need_pallets == 0:
                return [], 0, 0, 0, 0
            use_cols = min(n_cols, max(1, math.ceil(need_pallets / max(1, n_rows))))
            use_rows = min(n_rows, max(1, math.ceil(need_pallets / max(1, use_cols))))
            used_L = use_cols*(pal_L_draw + pallet_gap) - (pallet_gap if use_cols>0 else 0)
            used_W = use_rows*(pal_W_draw + pallet_gap) - (pallet_gap if use_rows>0 else 0)
            off_x = rect_x0 if anchor_left else (rect_x0 + rect_L - used_L)
            off_y = rect_y0 if anchor_front else (rect_y0 + rect_W - used_W)
            slots = [(off_x + c*(pal_L_draw + pallet_gap),
                      off_y + r*(pal_W_draw + pallet_gap), 0.0)
                     for r in range(use_rows) for c in range(use_cols)]
            return slots, used_L, used_W, off_x, off_y

        def score_bara(slots, used_L, used_W, off_x, off_y):
            z_layers = compute_bara_z_layers(h_eff, bag_H)
            if z_layers <= 0: return 0
            x0, y0 = off_x, off_y
            x1, y1 = off_x + used_L, off_y + used_W
            total = 0
            L_left = x0 - rect_x0
            if L_left > 1e-6:
                total += len(fill_rect_with_bags_best_safe(rect_x0, rect_y0, L_left, rect_W, bag_L, bag_W, gap_bara, bag_H, z_layers))
            L_right = (rect_x0 + rect_L) - x1
            if L_right > 1e-6:
                total += len(fill_rect_with_bags_best_safe(x1, rect_y0, L_right, rect_W, bag_L, bag_W, gap_bara, bag_H, z_layers))
            W_front = y0 - rect_y0
            if W_front > 1e-6:
                total += len(fill_rect_with_bags_best_safe(x0, rect_y0, used_L, W_front, bag_L, bag_W, gap_bara, bag_H, z_layers))
            W_back = (rect_y0 + rect_W) - y1
            if W_back > 1e-6:
                total += len(fill_rect_with_bags_best_safe(x0, y1, used_L, W_back, bag_L, bag_W, gap_bara, bag_H, z_layers))
            return total

        if enable_bara:
            best = None
            for anchor_left in [True, False]:
                for anchor_front in [True, False]:
                    slots, usedL, usedW, ox, oy = candidate_slots(anchor_left, anchor_front, cols, rows)
                    sc = score_bara(slots, usedL, usedW, ox, oy)
                    if (best is None) or (sc > best[0]):
                        best = (sc, slots)
            pallet_origins = best[1] if best else []
        else:
            # 中央寄せ
            off_x = rect_x0 + max(0.0, (rect_L - used_L_full)/2.0)
            off_y = rect_y0 + max(0.0, (rect_W - used_W_full)/2.0)
            use_cols = min(cols, max(1, math.ceil(need_pallets / max(1, rows))))
            use_rows = min(rows, max(1, math.ceil(need_pallets / max(1, use_cols))))
            for r in range(use_rows):
                for c in range(use_cols):
                    pallet_origins.append((off_x + c*(pal_L_draw + pallet_gap),
                                           off_y + r*(pal_W_draw + pallet_gap), 0.0))
            pallet_origins = pallet_origins[:need_pallets]

    # === パレット上の袋（重量上限に合わせて採用） ===
    selected_pallet_bags = []
    if enable_pallets and bag_boxes_on_pallet and len(pallet_origins)>0:
        per_pallet = len(bag_boxes_on_pallet)
        max_bags_allowed = int(max_product_weight // bag_weight) if (enable_max_w and bag_weight>0) else 10**9
        remain = max_bags_allowed
        for slot in pallet_origins:
            if remain <= 0: break
            take = min(per_pallet, remain)
            for i in range(take):
                (o,s) = bag_boxes_on_pallet[i]
                (bx,by,bz) = o
                selected_pallet_bags.append(((slot[0]+bx, slot[1]+by, slot[2]+bz), s))
            remain -= take
        used_pallets = math.ceil((max_bags_allowed - max(0,remain)) / max(1, per_pallet))
        pallet_origins = pallet_origins[:used_pallets]
    else:
        used_pallets = 0

    # === ばら積み（安全版） ===
    bara_boxes_all = []
    if enable_bara:
        z_layers = compute_bara_z_layers(h_eff, bag_H)
        if z_layers > 0:
            if enable_pallets and len(pallet_origins)>0:
                xs = [x for (x,_,_) in pallet_origins]
                ys = [y for (_,y,_) in pallet_origins]
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    used_L = (max_x - min_x) + (pal_L_draw if pal_L_draw>0 else pal_L)
                    used_W = (max_y - min_y) + (pal_W_draw if pal_W_draw>0 else pal_W)
                    off_x, off_y = min_x, min_y
                else:
                    used_L = used_W = off_x = off_y = 0.0

                L_left = (off_x - rect_x0)
                if L_left > 1e-6:
                    bara_boxes_all += fill_rect_with_bags_best_safe(rect_x0, rect_y0, L_left, rect_W, bag_L, bag_W, gap_bara, bag_H, z_layers)
                x_right = off_x + used_L
                L_right = (rect_x0 + rect_L) - x_right
                if L_right > 1e-6:
                    bara_boxes_all += fill_rect_with_bags_best_safe(x_right, rect_y0, L_right, rect_W, bag_L, bag_W, gap_bara, bag_H, z_layers)
                W_front = (off_y - rect_y0)
                if W_front > 1e-6:
                    bara_boxes_all += fill_rect_with_bags_best_safe(off_x, rect_y0, used_L, W_front, bag_L, bag_W, gap_bara, bag_H, z_layers)
                y_back = off_y + used_W
                W_back = (rect_y0 + rect_W) - y_back
                if W_back > 1e-6:
                    bara_boxes_all += fill_rect_with_bags_best_safe(off_x, y_back, used_L, W_back, bag_L, bag_W, gap_bara, bag_H, z_layers)
            else:
                bara_boxes_all += fill_rect_with_bags_best_safe(rect_x0, rect_y0, rect_L, rect_W, bag_L, bag_W, gap_bara, bag_H, z_layers)

    # --- 重量上限の最終適用（パレット優先→残りをばら） ---
    if enable_max_w and bag_weight > 0:
        already = len(selected_pallet_bags)
        max_bags_allowed = int(max_product_weight // bag_weight)
        remain = max(0, max_bags_allowed - already)
        selected_bara_bags = bara_boxes_all[:remain]
    else:
        selected_bara_bags = bara_boxes_all

    # === 集計 ===
    bags_on_pallet_kept = len(selected_pallet_bags)
    bags_bara_kept = len(selected_bara_bags)
    total_bags_kept = bags_on_pallet_kept + bags_bara_kept

    total_weight_bags = total_bags_kept * bag_weight
    total_weight_pallets = (len(pallet_origins) if enable_pallets else 0) * pallet_weight
    total_weight = total_weight_bags + total_weight_pallets

    total_vol_m3 = total_bags_kept * mm3_to_m3(bag_L*bag_W*bag_H) + (len(pallet_origins) if enable_pallets else 0) * mm3_to_m3(pal_L*pal_W*pal_H)

    # ========= 表示 =========
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("パレット積載（1枚あたりの計画）")
        st.write(f"- パターン: {pattern}")
        st.write(f"- 段数（設定）: {layers} 段")
        if pattern != "8俵（偶数=端/中央, 奇数=縦2×4）":
            st.write(f"- 1層あたり（参考）: {per_layer if per_layer is not None else '-'} 袋")
        if enable_pallets:
            st.write(f"- 1パレット最大: **{len(bag_boxes_on_pallet)} 袋**")
        else:
            st.write("- パレット積み：**OFF**（ばら積みのみ）")
    with c2:
        st.subheader("採用結果（重量上限を反映）")
        if enable_pallets:
            st.write(f"- 使用パレット: **{len(pallet_origins)} 枚**（空パレットは未配置）")
            st.write(f"- パレット採用袋数: **{bags_on_pallet_kept} 袋**")
        if enable_bara:
            st.write(f"- ばら積み採用袋数: **{bags_bara_kept} 袋**")
        st.write(f"- 紙袋合計: **{total_bags_kept} 袋**")
        if enable_max_w:
            st.write(f"- 最大製品重量: {max_product_weight:,.0f} kg（袋のみ）")
    with c3:
        st.subheader("重量・体積 / 有効範囲")
        st.write(f"- 紙袋合計重量: {total_weight_bags:,.1f} kg")
        st.write(f"- パレット合計重量: {total_weight_pallets:,.1f} kg")
        st.write(f"- **総積載重量**: **{total_weight:,.1f} kg**")
        st.write(f"- 合計体積（参考）: {total_vol_m3:,.3f} m³")
        st.write(f"- 有効高さ（計算のみ）: **{h_eff} mm**（描画は {truck_H_in} mm）")
        st.write(f"- 有効平面: **{rect_L:.0f} × {rect_W:.0f} mm**（上下左右 各10 mmマージン）")
        if enable_pallets:
            st.write(f"- パレット間隙間: **{pallet_gap} mm**")
        if enable_bara and enable_pallets:
            st.write("- パレットは**片寄せ配置**（ばら積み収容が最大のアンカーを自動選択）")

    st.markdown("---")
    st.subheader("3Dビュー（荷台=半透明 / 回転中心=荷台中央 / 底面は水平）")

    fig = go.Figure()

    # 荷台（半透明）
    fig.add_trace(cuboid_mesh(
        origin=center_origin((0,0,0), truck_L, truck_W, truck_H_in),
        size=(truck_L, truck_W, truck_H_in),
        color="#B0B0B0", opacity=0.18, name="荷台", showlegend=True
    ))

    # 有効平面ガイド
    guide_h = 2.0
    fig.add_trace(cuboid_mesh(
        origin=center_origin((rect_x0, rect_y0, 0), truck_L, truck_W, truck_H_in),
        size=(rect_L, guide_h, guide_h), color="#DDDDDD", opacity=1.0))
    fig.add_trace(cuboid_mesh(
        origin=center_origin((rect_x0, rect_y0+rect_W-guide_h, 0), truck_L, truck_W, truck_H_in),
        size=(rect_L, guide_h, guide_h), color="#DDDDDD", opacity=1.0))
    fig.add_trace(cuboid_mesh(
        origin=center_origin((rect_x0, rect_y0, 0), truck_L, truck_W, truck_H_in),
        size=(guide_h, rect_W, guide_h), color="#DDDDDD", opacity=1.0))
    fig.add_trace(cuboid_mesh(
        origin=center_origin((rect_x0+rect_L-guide_h, rect_y0, 0), truck_L, truck_W, truck_H_in),
        size=(guide_h, rect_W, guide_h), color="#DDDDDD", opacity=1.0))

    pallet_color = "#C69C6D"
    bag_colors = ["#E07B39", "#E0A339", "#E06B6B", "#6BA8E0", "#6BE09E"]
    show_pal_legend = True
    show_bagP_legend = True
    show_bagB_legend = True

    # パレット本体
    if enable_pallets:
        for (px,py,pz) in pallet_origins:
            fig.add_trace(cuboid_mesh(
                origin=center_origin((px,py,pz), truck_L, truck_W, truck_H_in),
                size=(pal_L if pal_L>0 else pal_L_draw, pal_W if pal_W>0 else pal_W_draw, pal_H),
                color=pallet_color, opacity=1.0, name="パレット", showlegend=show_pal_legend
            ))
            show_pal_legend = False

    # 袋（段境界の視認ギャップ）
    def draw_bag(origin, size, color, legend_name, showlegend):
        (x,y,z) = origin; (L,W,H) = size
        shrink_z = min(visual_gap_height_mm, max(0.0, H - 0.1))
        origin_disp = center_origin((x, y, z + shrink_z/2.0), truck_L, truck_W, truck_H_in)
        fig.add_trace(cuboid_mesh(
            origin=origin_disp, size=(L, W, H - shrink_z),
            color=color, opacity=1.0, name=legend_name, showlegend=showlegend
        ))

    for (o,s) in selected_pallet_bags:
        color = bag_colors[int((o[1] // max(1,(s[1]+gap_xy))) % len(bag_colors))]
        draw_bag(o, s, color, "紙袋（パレット）", show_bagP_legend); show_bagP_legend = False

    for (o,s) in selected_bara_bags:
        draw_bag(o, s, "#7B9E46", "紙袋（ばら）", show_bagB_legend); show_bagB_legend = False

    # 表示範囲（原点=荷台中心）
    max_dim = max(truck_L, truck_W, truck_H_in)
    pad = max_dim * (pad_pct/100.0)
    xr = (-(truck_L/2 + pad),  (truck_L/2 + pad))
    yr = (-(truck_W/2 + pad),  (truck_W/2 + pad))
    zr = (-(truck_H_in/2 + pad), (truck_H_in/2 + pad))

    # 縦横比
    if aspect_mode == "実寸（manual）":
        m = float(max(truck_L, truck_W, truck_H_in))
        aspectratio = dict(x=truck_L/m, y=truck_W/m, z=truck_H_in/m); scene_aspectmode = "manual"
    elif aspect_mode == "データ自動（data）":
        aspectratio = None; scene_aspectmode = "data"
    else:
        aspectratio = dict(x=1, y=1, z=1); scene_aspectmode = "manual"

    # 視点
    if view_preset == "斜め全体（推奨）":
        eye = dict(x=1.8, y=1.8, z=1.2)
    elif view_preset == "俯瞰（トップ）":
        eye = dict(x=0.01, y=0.01, z=3.0)
    elif view_preset == "正面（後扉側）":
        eye = dict(x=0.01, y=3.0, z=0.6)
    else:
        eye = dict(x=3.0, y=0.01, z=0.6)

    camera = dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))
    if ortho:
        camera["projection"] = dict(type="orthographic")

    fig.update_scenes(
        xaxis_title="X (mm) [長手]", yaxis_title="Y (mm) [幅]", zaxis_title="Z (mm) [高さ]",
        aspectmode=scene_aspectmode, aspectratio=aspectratio,
        xaxis=dict(range=xr, showgrid=not hide_axes, zeroline=not hide_axes, visible=not hide_axes),
        yaxis=dict(range=yr, showgrid=not hide_axes, zeroline=not hide_axes, visible=not hide_axes),
        zaxis=dict(range=zr, showgrid=not hide_axes, zeroline=not hide_axes, visible=not hide_axes),
        bgcolor=bg_color
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=camera,
        dragmode="turntable",
        uirevision="fixed"
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False})

    st.caption("※ 偶数段は左右=端合わせ（上/中/下）、中央=縦2段センター。袋間のみ隙間適用。ばら積みは領域分割で重複/はみ出しを防止。")
else:
    st.write("左のパラメータを設定し、**「シミュレーション実行」**を押してください。")
    st.caption("パレット積みをOFFにすると、ばら積みのみのシミュレーションが可能です。")
