#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Medicine Shielding Calculator - Web App
Streamlit 版本，适配手机与桌面
基于 V2.3 核心逻辑
制作者: Fang KeMing & DeepSeek
版本: Web 1.1
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import math
from datetime import datetime, timedelta

# ==================== 配置中文字体（解决 Glyph missing 警告） ====================
try:
    # 尝试查找系统中已安装的中文字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'Arial Unicode MS']
    for font in preferred_fonts:
        if font in font_list:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    else:
        # 如果都没找到，用默认无衬线字体（可能会警告但不会出错）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 核素参数库 ====================
NUCLIDE_DATA = {
    "F-18": {
        "half_life_min": 109.8,
        "decay_constant": 0.00631,
        "gamma_const": 0.143,
        "tvl": {"Lead": 16.6, "Concrete": 176, "Steel": 65}
    },
    "I-131": {
        "half_life_min": 11520.0,
        "decay_constant": 0.0000602,
        "gamma_const": 0.0595,
        "tvl": {"Lead": 10.0, "Concrete": 110, "Steel": 40}
    },
    "C-11": {
        "half_life_min": 20.4,
        "decay_constant": 0.0340,
        "gamma_const": 0.143,
        "tvl": {"Lead": 16.6, "Concrete": 176, "Steel": 65}
    },
    "N-13": {
        "half_life_min": 9.97,
        "decay_constant": 0.0695,
        "gamma_const": 0.143,
        "tvl": {"Lead": 16.6, "Concrete": 176, "Steel": 65}
    },
    "O-15": {
        "half_life_min": 2.04,
        "decay_constant": 0.340,
        "gamma_const": 0.143,
        "tvl": {"Lead": 16.6, "Concrete": 176, "Steel": 65}
    }
}
DEFAULT_NUCLIDE = "F-18"
DEFAULT_SELF_ABS = 0.64
DEFAULT_DOSE_RATE_LIMIT = 2.5

# ==================== 核心计算函数 ====================
def decay_correction(activity_ref, ref_time_str, current_time_str, nuclide_key):
    try:
        ref_dt = datetime.strptime(ref_time_str, "%Y-%m-%d %H:%M")
        cur_dt = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("时间格式错误，请使用 'YYYY-MM-DD HH:MM'")
    delta_min = (cur_dt - ref_dt).total_seconds() / 60.0
    if delta_min < 0:
        raise ValueError("当前时间必须晚于参考时间")
    lam = NUCLIDE_DATA[nuclide_key]["decay_constant"]
    A_cur = activity_ref * math.exp(-lam * delta_min)
    percent = (A_cur / activity_ref) * 100 if activity_ref != 0 else 0
    return A_cur, percent, delta_min

def calc_thickness_instant(activity_mbq, distance_m, material, dose_rate_limit,
                           correction_factor, use_ct=False, ct_dose_rate=0.0,
                           gamma_const=None, tvl_dict=None):
    if gamma_const is None or tvl_dict is None:
        raise ValueError("Missing nuclear data")
    if material not in tvl_dict:
        raise ValueError(f"Material '{material}' not supported")
    tvl = tvl_dict[material]
    gamma_dose = (activity_mbq * gamma_const * correction_factor) / (distance_m ** 2)
    total_dose = gamma_dose + (ct_dose_rate if use_ct else 0.0)
    if total_dose <= dose_rate_limit:
        return 0.0, total_dose
    B = dose_rate_limit / total_dose
    thickness = tvl * math.log10(1.0 / B)
    return thickness, total_dose

def compute_unshielded_dose_rate(activity_mbq, distance_m, correction_factor, gamma_const):
    return (activity_mbq * gamma_const * correction_factor) / (distance_m ** 2)

# ==================== Streamlit 页面配置 ====================
st.set_page_config(
    page_title="核医学屏蔽计算器",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 隐藏 Streamlit 默认的右上角菜单和脚注
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {margin-top: -50px;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("☢️ 核医学屏蔽计算器")
st.caption("版本 Web 1.1  |  制作者: Fang KeMing & DeepSeek  |  支持手机与桌面")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 全局设置")
    nuclide = st.selectbox("核素", list(NUCLIDE_DATA.keys()), index=0)
    material = st.selectbox("屏蔽材料", ["Lead", "Concrete", "Steel"], index=0)
    abs_factor = st.number_input("自吸收因子", min_value=0.01, max_value=1.0, value=DEFAULT_SELF_ABS, step=0.01)
    dose_limit = st.number_input("目标剂量率 (μSv/h)", min_value=0.1, value=DEFAULT_DOSE_RATE_LIMIT, step=0.1)
    use_ct = st.checkbox("考虑CT贡献")
    ct_dose = st.number_input("CT剂量率 (μSv/h)", min_value=0.0, value=1.0, step=0.1, disabled=not use_ct)
    decay_enabled = st.checkbox("启用衰变修正", value=True)
    current_time = st.text_input("当前时间 (YYYY-MM-DD HH:MM)", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
    st.markdown("---")
    st.markdown("**说明**：双源模式下，两个源共用以上全局参数，但每个源可独立设置活度和参考时间。")

# ==================== Tab 1: 单源简单计算 ====================
st.header("1️⃣ 单源点源屏蔽")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("源参数")
    ref_act_single = st.number_input("参考活度 (MBq)", value=370.0, step=10.0, key="single_act")
    ref_time_single = st.text_input("参考时间", value=(datetime.now() - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"), key="single_ref")
    distance = st.number_input("距离 (m)", min_value=0.1, value=3.0, step=0.1, key="single_dist")

    if st.button("计算单源", key="calc_single"):
        try:
            if decay_enabled:
                A_cur, percent, _ = decay_correction(ref_act_single, ref_time_single, current_time, nuclide)
                st.success(f"当前活度: {A_cur:.2f} MBq (衰减至 {percent:.1f}%)")
            else:
                A_cur = ref_act_single
                st.info("衰变修正未启用，直接使用参考活度")
        except Exception as e:
            st.error(f"衰变修正错误: {e}")
            st.stop()

        data = NUCLIDE_DATA[nuclide]
        gamma_const = data["gamma_const"]
        tvl_dict = data["tvl"]
        if material not in tvl_dict:
            st.error(f"材料 {material} 不支持该核素")
            st.stop()

        try:
            thickness, dose_rate = calc_thickness_instant(
                A_cur, distance, material, dose_limit, abs_factor,
                use_ct=use_ct, ct_dose_rate=ct_dose,
                gamma_const=gamma_const, tvl_dict=tvl_dict
            )
            col1_metric, col2_metric = st.columns(2)
            col1_metric.metric("所需屏蔽厚度", f"{thickness:.2f} mm")
            col2_metric.metric("未屏蔽剂量率", f"{dose_rate:.3f} μSv/h")
        except Exception as e:
            st.error(f"计算错误: {e}")
            st.stop()

        # 绘制厚度随距离变化曲线
        dist_range = np.linspace(0.2, max(5.0, distance*2), 50)
        thicks = []
        for d in dist_range:
            try:
                th, _ = calc_thickness_instant(
                    A_cur, d, material, dose_limit, abs_factor,
                    use_ct=use_ct, ct_dose_rate=ct_dose,
                    gamma_const=gamma_const, tvl_dict=tvl_dict
                )
                thicks.append(th if th > 0 else 0)
            except:
                thicks.append(np.nan)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(dist_range, thicks, 'b-', linewidth=2)
        ax.plot(distance, thickness, 'ro', markersize=8)
        ax.set_xlabel("距离 (m)")
        ax.set_ylabel("所需厚度 (mm)")
        ax.set_title(f"厚度随距离变化 ({material})")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

with col2:
    st.subheader("📊 结果与说明")
    st.markdown("""
    - **TVL (十分之一值层)** 根据所选核素和材料自动选取。
    - 公式：厚度 = TVL × log₁₀( 未屏蔽剂量率 / 目标剂量率 )
    - 未屏蔽剂量率包括自吸收修正和CT贡献（若勾选）。
    """)
    data = NUCLIDE_DATA[nuclide]
    st.info(f"核素 {nuclide} 半衰期: {data['half_life_min']:.1f} 分钟\nΓ = {data['gamma_const']} μSv·m²/(h·MBq)")

# ==================== Tab 2: 双源房间布局 ====================
st.header("2️⃣ 双源房间布局屏蔽计算")

with st.expander("🔽 设置两个源的位置和活度", expanded=True):
    col_src1, col_src2 = st.columns(2)
    with col_src1:
        st.subheader("源 1")
        src1_x = st.number_input("X (m)", value=2.5, step=0.1, key="src1x")
        src1_y = st.number_input("Y (m)", value=2.0, step=0.1, key="src1y")
        src1_z = st.number_input("Z (m)", value=1.5, step=0.1, key="src1z")
        ref_act1 = st.number_input("参考活度 (MBq)", value=370.0, step=10.0, key="act1")
        ref_time1 = st.text_input("参考时间", value=(datetime.now() - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"), key="ref1")
    with col_src2:
        st.subheader("源 2")
        src2_x = st.number_input("X (m)", value=3.5, step=0.1, key="src2x")
        src2_y = st.number_input("Y (m)", value=2.0, step=0.1, key="src2y")
        src2_z = st.number_input("Z (m)", value=1.5, step=0.1, key="src2z")
        ref_act2 = st.number_input("参考活度 (MBq)", value=185.0, step=10.0, key="act2")
        ref_time2 = st.text_input("参考时间", value=(datetime.now() - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M"), key="ref2")

    st.subheader("房间尺寸")
    col_dim = st.columns(3)
    L = col_dim[0].number_input("长度 X (m)", value=6.0, step=0.5)
    W = col_dim[1].number_input("宽度 Y (m)", value=4.0, step=0.5)
    H = col_dim[2].number_input("高度 Z (m)", value=3.0, step=0.5)

    offset = st.number_input("离墙偏移 (m)", value=0.3, step=0.1)

    st.subheader("门和窗（可选）")
    col_door = st.columns(3)
    door_side = col_door[0].selectbox("门1侧边", ["N","S","E","W"], index=0, key="door1side")
    door_pos = col_door[1].number_input("门1位置 (m)", value=3.0, step=0.5, key="door1pos")
    door_width = col_door[2].number_input("门1宽度 (m)", value=1.0, step=0.1, key="door1w")

    col_win = st.columns(3)
    win_side = col_win[0].selectbox("窗侧边", ["N","S","E","W"], index=3, key="winside")
    win_pos = col_win[1].number_input("窗位置 (m)", value=3.0, step=0.5, key="winpos")
    win_width = col_win[2].number_input("窗宽度 (m)", value=0.8, step=0.1, key="winw")

    col_door2 = st.columns(3)
    door2_side = col_door2[0].selectbox("门2侧边", ["N","S","E","W"], index=1, key="door2side")
    door2_pos = col_door2[1].number_input("门2位置 (m)", value=2.0, step=0.5, key="door2pos")
    door2_width = col_door2[2].number_input("门2宽度 (m)", value=0.9, step=0.1, key="door2w")

if st.button("计算双源房间", key="calc_room"):
    try:
        if decay_enabled:
            A1, p1, _ = decay_correction(ref_act1, ref_time1, current_time, nuclide)
            A2, p2, _ = decay_correction(ref_act2, ref_time2, current_time, nuclide)
            st.success(f"源1当前活度: {A1:.2f} MBq ({p1:.1f}%)  源2当前活度: {A2:.2f} MBq ({p2:.1f}%)")
        else:
            A1, A2 = ref_act1, ref_act2
            st.info("衰变修正未启用")
    except Exception as e:
        st.error(f"衰变修正错误: {e}")
        st.stop()

    data = NUCLIDE_DATA[nuclide]
    gamma_const = data["gamma_const"]
    tvl_dict = data["tvl"]
    if material not in tvl_dict:
        st.error(f"材料 {material} 不支持该核素")
        st.stop()

    sources = [(A1, (src1_x, src1_y, src1_z)), (A2, (src2_x, src2_y, src2_z))]
    points = {}
    wall_centers = {
        'N': (L/2, W, src1_z),
        'S': (L/2, 0, src1_z),
        'E': (L, W/2, src1_z),
        'W': (0, W/2, src1_z)
    }
    for side, (cx, cy, cz) in wall_centers.items():
        points[f"Wall {side}"] = (cx, cy, cz)
    points["Ceiling"] = (L/2, W/2, H)
    points["Floor"] = (L/2, W/2, 0)

    def get_point(side, pos, z):
        if side == 'N': return (pos, W, z)
        elif side == 'S': return (pos, 0, z)
        elif side == 'E': return (L, pos, z)
        elif side == 'W': return (0, pos, z)
    if door_side in ['N','S','E','W']:
        points["Door1"] = get_point(door_side, door_pos, src1_z)
    if door2_side in ['N','S','E','W']:
        points["Door2"] = get_point(door2_side, door2_pos, src1_z)
    if win_side in ['N','S','E','W']:
        points["Window"] = get_point(win_side, win_pos, src1_z)

    result_points = {}
    for name, (px, py, pz) in points.items():
        total_dose = 0.0
        dists = []
        for (act, (sx, sy, sz)) in sources:
            dist = math.dist((sx, sy, sz), (px, py, pz)) + offset
            dists.append(dist)
            dose = compute_unshielded_dose_rate(act, dist, abs_factor, gamma_const)
            total_dose += dose
        if use_ct:
            total_dose += ct_dose
        if total_dose <= dose_limit:
            thick = 0.0
        else:
            B = dose_limit / total_dose
            thick = tvl_dict[material] * math.log10(1.0 / B)
        result_points[name] = (px, py, pz, dists, total_dose, thick)

    st.subheader("📋 计算结果")
    table_data = []
    for name, (px, py, pz, dists, dose, thick) in result_points.items():
        table_data.append({
            "位置": name,
            "距离1 (m)": f"{dists[0]:.2f}",
            "距离2 (m)": f"{dists[1]:.2f}" if len(dists)>1 else "-",
            "总剂量率 (μSv/h)": f"{dose:.3f}",
            "厚度 (mm)": f"{thick:.2f}"
        })
    st.table(table_data)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.add_patch(plt.Rectangle((0,0), L, W, fill=True, facecolor='lightgray', edgecolor='black', linewidth=2))
    ax1.text(L/2, -0.3, f"{L} m", ha='center', va='top')
    ax1.text(-0.3, W/2, f"{W} m", ha='right', va='center', rotation=90)
    colors = ['red', 'magenta']
    for i, (act, (sx, sy, sz)) in enumerate(sources):
        ax1.plot(sx, sy, marker='*', color=colors[i], markersize=12, label=f'源{i+1}')
    for name, (px, py, pz, dists, dose, thick) in result_points.items():
        if 'Wall' in name:
            marker, color = 's', 'blue'
        elif name in ['Ceiling','Floor']:
            marker, color = '^', 'green'
        else:
            marker, color = 'D', 'orange'
        ax1.plot(px, py, marker=marker, color=color, markersize=8)
        ax1.text(px+0.1, py+0.1, f"{thick:.1f}mm", fontsize=7)
    for label in ['Door1','Door2','Window']:
        if label in result_points:
            px, py, pz, _, _, _ = result_points[label]
            if label == 'Door1' or label == 'Door2':
                lc = 'k-'; lt = label
            else:
                lc = 'b-'; lt = 'Window'
            if abs(py - W) < 0.01:
                ax1.plot([px-0.2, px+0.2], [W, W], lc, linewidth=3)
                ax1.text(px, W+0.2, lt, ha='center', fontsize=8)
            elif abs(py) < 0.01:
                ax1.plot([px-0.2, px+0.2], [0, 0], lc, linewidth=3)
                ax1.text(px, -0.2, lt, ha='center', fontsize=8)
            elif abs(px - L) < 0.01:
                ax1.plot([L, L], [py-0.2, py+0.2], lc, linewidth=3)
                ax1.text(L+0.2, py, lt, va='center', fontsize=8)
            elif abs(px) < 0.01:
                ax1.plot([0, 0], [py-0.2, py+0.2], lc, linewidth=3)
                ax1.text(-0.2, py, lt, va='center', ha='right', fontsize=8)
    ax1.set_xlim(-0.5, L+0.5)
    ax1.set_ylim(-0.5, W+0.5)
    ax1.set_aspect('equal')
    ax1.set_title("俯视图")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(122, projection='3d')
    floor_pts = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,0)]
    fx, fy, fz = zip(*floor_pts)
    ax2.plot(fx, fy, fz, 'k-', linewidth=1)
    ceil_pts = [(0,0,H), (L,0,H), (L,W,H), (0,W,H), (0,0,H)]
    cx, cy, cz = zip(*ceil_pts)
    ax2.plot(cx, cy, cz, 'k-', linewidth=1)
    for (x,y) in [(0,0), (L,0), (L,W), (0,W)]:
        ax2.plot([x,x], [y,y], [0,H], 'k--', linewidth=0.8, alpha=0.5)
    for i, (act, (sx, sy, sz)) in enumerate(sources):
        ax2.scatter(sx, sy, sz, color=colors[i], s=80, marker='*', label=f'源{i+1}')
    for name, (px, py, pz, dists, dose, thick) in result_points.items():
        if 'Wall' in name:
            marker, color = 's', 'blue'
        elif name in ['Ceiling','Floor']:
            marker, color = '^', 'green'
        else:
            marker, color = 'D', 'orange'
        ax2.scatter(px, py, pz, marker=marker, color=color, s=60)
        ax2.text(px, py, pz, f"{thick:.1f}", fontsize=7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("3D视图")
    ax2.view_init(elev=25, azim=-60)
    ax2.legend(loc='upper left')

    st.pyplot(fig)

st.markdown("---")
st.caption("本应用基于桌面版 V2.3 核心逻辑，适配移动端。所有计算仅作参考，实际屏蔽设计请遵循相关法规。")