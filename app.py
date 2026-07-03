import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import math
from datetime import datetime, timedelta
import platform
import matplotlib.font_manager as fm

# ============================================================
# 1. 修复 matplotlib 中文显示（适配 Streamlit Cloud）
# ============================================================
def set_chinese_font():
    if platform.system() == 'Linux':
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_keywords = ['cjk', 'hei', 'song', 'ming', 'noto', 'sc', 'cn', 'chinese']
        chinese_fonts = [f for f in font_list if any(key in f.lower() for key in chinese_keywords)]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ============================================================
# 2. 核素参数库（扩展了更多核素）
# ============================================================
NUCLIDE_DATA = {
    "F-18": {
        "half_life_min": 109.8,
        "decay_constant": 0.00631,
        "gamma_const": 0.143,
        "tvl": {'Lead': 16.6, 'Concrete': 176, 'Steel': 65}
    },
    "Ga-68": {
        "half_life_min": 67.7,
        "decay_constant": 0.01024,
        "gamma_const": 0.143,
        "tvl": {'Lead': 16.6, 'Concrete': 176, 'Steel': 65}
    },
    "C-11": {
        "half_life_min": 20.4,
        "decay_constant": 0.0340,
        "gamma_const": 0.143,
        "tvl": {'Lead': 16.6, 'Concrete': 176, 'Steel': 65}
    },
    "N-13": {
        "half_life_min": 9.97,
        "decay_constant": 0.0695,
        "gamma_const": 0.143,
        "tvl": {'Lead': 16.6, 'Concrete': 176, 'Steel': 65}
    },
    "O-15": {
        "half_life_min": 2.04,
        "decay_constant": 0.340,
        "gamma_const": 0.143,
        "tvl": {'Lead': 16.6, 'Concrete': 176, 'Steel': 65}
    },
    "I-131": {
        "half_life_min": 11520.0,
        "decay_constant": 0.0000602,
        "gamma_const": 0.0595,
        "tvl": {'Lead': 10.0, 'Concrete': 110, 'Steel': 40}
    },
    # 可继续添加更多核素
}

DEFAULT_NUCLIDE = "F-18"
DEFAULT_SELF_ABS = 0.64
DEFAULT_DOSE_RATE_LIMIT = 2.5
TVL_KEYS = ['Lead', 'Concrete', 'Steel']

# ============================================================
# 3. 核心计算函数（完全来自桌面版）
# ============================================================
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
                           correction_factor,
                           use_ct=False, ct_dose_rate=0.0,
                           gamma_const=None, tvl_dict=None):
    if gamma_const is None or tvl_dict is None:
        raise ValueError("Missing nuclear data (gamma_const, tvl)")
    if material not in tvl_dict:
        raise ValueError(f"Material '{material}' not supported for this nuclide")
    tvl = tvl_dict[material]
    gamma_dose = (activity_mbq * gamma_const * correction_factor) / (distance_m ** 2)
    total_dose = gamma_dose + (ct_dose_rate if use_ct else 0.0)
    if total_dose <= dose_rate_limit:
        return 0.0
    B = dose_rate_limit / total_dose
    return tvl * math.log10(1.0 / B)

def calc_thickness_weekly(activity_mbq, distance_m, material,
                          weekly_workload_mbq_min, occupancy_factor,
                          dose_limit_weekly,
                          correction_factor,
                          gamma_const=None, tvl_dict=None):
    if gamma_const is None or tvl_dict is None:
        raise ValueError("Missing nuclear data (gamma_const, tvl)")
    if material not in tvl_dict:
        raise ValueError(f"Material '{material}' not supported for this nuclide")
    tvl = tvl_dict[material]
    total_activity_h = weekly_workload_mbq_min / 60.0
    gamma_corr = gamma_const * correction_factor
    unshielded_dose = gamma_corr * total_activity_h / (distance_m ** 2) * occupancy_factor
    if unshielded_dose <= dose_limit_weekly:
        return 0.0
    B = dose_limit_weekly / unshielded_dose
    return tvl * math.log10(1.0 / B)

def compute_unshielded_dose_rate(activity_mbq, distance_m, correction_factor, gamma_const):
    return (activity_mbq * gamma_const * correction_factor) / (distance_m ** 2)

# ============================================================
# 4. Streamlit 页面设置
# ============================================================
st.set_page_config(page_title="核医学屏蔽计算器 Web", layout="wide")
st.title("☢️ Nuclear Medicine Shielding Calculator")
st.markdown("**版本 V2.3 Web** | 制作者: Fang KeMing & DeepSeek | 2026-07-03")
st.warning("免责声明：本软件为免费软件，作者对使用结果不承担任何责任，请自行承担风险。")

# ============================================================
# 5. 导航标签（Simple / Room）
# ============================================================
tab_simple, tab_room = st.tabs(["🔵 Simple Point Source", "🟢 Room Layout (双源)"])

# ============================================================
# 6. Simple Mode (点源)
# ============================================================
with tab_simple:
    st.header("点源屏蔽计算")
    with st.sidebar:
        st.subheader("Simple Mode 参数")
        # 核素选择
        nuclide_s = st.selectbox("核素", list(NUCLIDE_DATA.keys()), key="s_nuclide")
        # 衰变修正
        st.subheader("衰变修正")
        decay_enabled_s = st.checkbox("启用衰变修正", value=True, key="s_decay")
        col_ref1, col_ref2 = st.columns(2)
        with col_ref1:
            ref_act_s = st.number_input("参考活度 (MBq)", value=370.0, step=10.0, key="s_ref_act")
            ref_time_s = st.text_input("参考时间 (YYYY-MM-DD HH:MM)", 
                                       value=(datetime.now() - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),
                                       key="s_ref_time")
        with col_ref2:
            cur_time_s = st.text_input("当前时间", value=datetime.now().strftime("%Y-%m-%d %H:%M"), key="s_cur_time")
            if st.button("应用衰变修正", key="s_apply"):
                try:
                    if decay_enabled_s:
                        A_cur, pct, _ = decay_correction(ref_act_s, ref_time_s, cur_time_s, nuclide_s)
                        st.session_state['s_cur_act'] = A_cur
                        st.session_state['s_cur_pct'] = pct
                    else:
                        st.session_state['s_cur_act'] = ref_act_s
                except Exception as e:
                    st.error(f"衰变修正错误: {e}")
        if 's_cur_act' in st.session_state:
            st.write(f"当前活度: **{st.session_state['s_cur_act']:.2f} MBq** " + 
                     (f"({st.session_state['s_cur_pct']:.1f}%)" if decay_enabled_s else "(未修正)"))
        else:
            st.session_state['s_cur_act'] = ref_act_s

        st.subheader("几何与材料")
        distance_s = st.number_input("距离 (m)", value=3.0, min_value=0.1, step=0.1, key="s_dist")
        abs_s = st.slider("自吸收因子", 0.0, 1.0, DEFAULT_SELF_ABS, 0.01, key="s_abs")
        material_s = st.selectbox("屏蔽材料", TVL_KEYS, key="s_mat")
        mode_s = st.selectbox("计算模式", ["Instantaneous Dose Rate", "Weekly Dose"], key="s_mode")
        # 瞬时参数
        if mode_s == "Instantaneous Dose Rate":
            dose_limit_s = st.number_input("目标剂量率 (μSv/h)", value=DEFAULT_DOSE_RATE_LIMIT, step=0.1, key="s_dose_limit")
            use_ct_s = st.checkbox("考虑CT贡献", value=False, key="s_ct")
            ct_dose_s = st.number_input("CT剂量率 (μSv/h)", value=1.0, step=0.1, disabled=not use_ct_s, key="s_ct_dose")
        else:
            workload_s = st.number_input("周工作量 (MBq·min)", value=1665000, step=10000, key="s_workload")
            occ_s = st.number_input("占用因子", value=0.25, step=0.05, key="s_occ")
            week_limit_s = st.number_input("周剂量限值 (μSv)", value=20, step=1, key="s_week_limit")
        log_scale_s = st.checkbox("Y轴对数", value=False, key="s_log")
        calc_btn_s = st.button("计算并绘图", key="s_calc")

    # 主区域显示结果
    if calc_btn_s or 's_calc_done' not in st.session_state:
        # 获取当前活度
        try:
            if decay_enabled_s:
                A_cur, _, _ = decay_correction(ref_act_s, ref_time_s, cur_time_s, nuclide_s)
            else:
                A_cur = ref_act_s
        except Exception as e:
            st.error(f"衰变修正错误: {e}")
            st.stop()
        # 获取参数
        try:
            dist = distance_s
            abs_factor = abs_s
            mat = material_s
            data = NUCLIDE_DATA[nuclide_s]
            gamma_const = data["gamma_const"]
            tvl_dict = data["tvl"]
            if mat not in tvl_dict:
                st.error(f"核素 {nuclide_s} 不支持材料 {mat}")
                st.stop()
        except Exception as e:
            st.error(f"输入错误: {e}")
            st.stop()

        # 计算厚度
        try:
            if mode_s == "Instantaneous Dose Rate":
                thickness = calc_thickness_instant(A_cur, dist, mat, dose_limit_s, abs_factor,
                                                   use_ct=use_ct_s, ct_dose_rate=ct_dose_s if use_ct_s else 0.0,
                                                   gamma_const=gamma_const, tvl_dict=tvl_dict)
                st.metric("所需屏蔽厚度", f"{thickness:.2f} mm")
                # 显示未屏蔽剂量率
                unshielded = compute_unshielded_dose_rate(A_cur, dist, abs_factor, gamma_const)
                if use_ct_s:
                    unshielded += ct_dose_s
                st.metric("未屏蔽剂量率", f"{unshielded:.3f} μSv/h")
            else:
                thickness = calc_thickness_weekly(A_cur, dist, mat, workload_s, occ_s, week_limit_s, abs_factor,
                                                  gamma_const=gamma_const, tvl_dict=tvl_dict)
                st.metric("所需屏蔽厚度", f"{thickness:.2f} mm")
        except Exception as e:
            st.error(f"计算错误: {e}")
            st.stop()

        # 绘图：厚度 vs 距离
        st.subheader("屏蔽厚度随距离变化")
        x_max = max(2 * dist, 1.0)
        dist_range = np.linspace(0.01, x_max, 100)
        thicknesses = []
        for d in dist_range:
            try:
                if mode_s == "Instantaneous Dose Rate":
                    th = calc_thickness_instant(A_cur, d, mat, dose_limit_s, abs_factor,
                                                use_ct=use_ct_s, ct_dose_rate=ct_dose_s if use_ct_s else 0.0,
                                                gamma_const=gamma_const, tvl_dict=tvl_dict)
                else:
                    th = calc_thickness_weekly(A_cur, d, mat, workload_s, occ_s, week_limit_s, abs_factor,
                                               gamma_const=gamma_const, tvl_dict=tvl_dict)
                if th == 0.0:
                    th = 1e-6
                thicknesses.append(th)
            except:
                thicknesses.append(np.nan)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(dist_range, thicknesses, 'bs-', label="所需厚度")
        ax.plot(dist, thickness if thickness != 0 else 1e-6, 'ro', markersize=10, label="当前设置")
        ax.set_xlabel("距离 (m)")
        ax.set_ylabel("所需厚度 (mm)")
        if log_scale_s:
            ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title(f"厚度 vs 距离 ({mat}, {nuclide_s})")
        st.pyplot(fig)

        # 材料比较按钮
        if st.button("材料比较", key="s_compare"):
            thicks = []
            materials = list(tvl_dict.keys())
            for mat_c in materials:
                try:
                    if mode_s == "Instantaneous Dose Rate":
                        th = calc_thickness_instant(A_cur, dist, mat_c, dose_limit_s, abs_factor,
                                                    use_ct=use_ct_s, ct_dose_rate=ct_dose_s if use_ct_s else 0.0,
                                                    gamma_const=gamma_const, tvl_dict=tvl_dict)
                    else:
                        th = calc_thickness_weekly(A_cur, dist, mat_c, workload_s, occ_s, week_limit_s, abs_factor,
                                                   gamma_const=gamma_const, tvl_dict=tvl_dict)
                    thicks.append(th)
                except:
                    thicks.append(0.0)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            bars = ax2.bar(materials, thicks, color=['gray','orange','green'])
            ax2.set_ylabel("厚度 (mm)")
            ax2.set_title(f"材料比较 ({nuclide_s})")
            for bar, val in zip(bars, thicks):
                ax2.text(bar.get_x()+bar.get_width()/2, val+0.5, f"{val:.1f}", ha='center')
            st.pyplot(fig2)

        # 显示公式（纯文本）
        with st.expander("公式说明"):
            st.text("【瞬时剂量率】\nd = TVL × log10( D_unshielded / D_target )\nD_unshielded = ( A × Γ × f ) / r²\n\n【周剂量】\nd = TVL × log10( D_week_unsh / D_week_limit )\nD_week_unsh = Γ × f × (MBq·h) / r² × occ")

# ============================================================
# 7. Room Layout Mode (双源)
# ============================================================
with tab_room:
    st.header("房间布局屏蔽计算 (双源)")
    with st.sidebar:
        st.subheader("Room Mode 参数")
        nuclide_r = st.selectbox("核素", list(NUCLIDE_DATA.keys()), key="r_nuclide")
        # 衰变修正
        st.subheader("衰变修正 (双源)")
        decay_enabled_r = st.checkbox("启用衰变修正", value=True, key="r_decay")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**源1**")
            ref_act1_r = st.number_input("参考活度1 (MBq)", value=370.0, step=10.0, key="r_ref_act1")
            ref_time1_r = st.text_input("参考时间1", 
                                        value=(datetime.now() - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),
                                        key="r_ref_time1")
        with col2:
            st.write("**源2**")
            ref_act2_r = st.number_input("参考活度2 (MBq)", value=185.0, step=10.0, key="r_ref_act2")
            ref_time2_r = st.text_input("参考时间2",
                                        value=(datetime.now() - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M"),
                                        key="r_ref_time2")
        cur_time_r = st.text_input("当前时间", value=datetime.now().strftime("%Y-%m-%d %H:%M"), key="r_cur_time")
        if st.button("应用衰变修正", key="r_apply"):
            try:
                if decay_enabled_r:
                    A1, p1, _ = decay_correction(ref_act1_r, ref_time1_r, cur_time_r, nuclide_r)
                    A2, p2, _ = decay_correction(ref_act2_r, ref_time2_r, cur_time_r, nuclide_r)
                    st.session_state['r_cur_act1'] = A1
                    st.session_state['r_cur_act2'] = A2
                else:
                    st.session_state['r_cur_act1'] = ref_act1_r
                    st.session_state['r_cur_act2'] = ref_act2_r
            except Exception as e:
                st.error(f"衰变修正错误: {e}")
        if 'r_cur_act1' in st.session_state:
            st.write(f"源1当前活度: **{st.session_state['r_cur_act1']:.2f} MBq**")
            st.write(f"源2当前活度: **{st.session_state['r_cur_act2']:.2f} MBq**")
        else:
            st.session_state['r_cur_act1'] = ref_act1_r
            st.session_state['r_cur_act2'] = ref_act2_r

        st.subheader("房间几何")
        L = st.number_input("房间长度 X (m)", value=6.0, step=0.5, key="r_len")
        W = st.number_input("房间宽度 Y (m)", value=4.0, step=0.5, key="r_wid")
        H = st.number_input("房间高度 Z (m)", value=3.0, step=0.5, key="r_height")

        st.subheader("源坐标")
        c1, c2, c3 = st.columns(3)
        with c1:
            x1 = st.number_input("源1 X", value=2.5, step=0.1, key="r_src1_x")
            y1 = st.number_input("源1 Y", value=2.0, step=0.1, key="r_src1_y")
            z1 = st.number_input("源1 Z", value=1.5, step=0.1, key="r_src1_z")
        with c2:
            x2 = st.number_input("源2 X", value=3.5, step=0.1, key="r_src2_x")
            y2 = st.number_input("源2 Y", value=2.0, step=0.1, key="r_src2_y")
            z2 = st.number_input("源2 Z", value=1.5, step=0.1, key="r_src2_z")

        st.subheader("其他参数")
        offset_r = st.number_input("偏移 (m)", value=0.3, step=0.05, key="r_offset")
        abs_r = st.slider("自吸收因子", 0.0, 1.0, DEFAULT_SELF_ABS, 0.01, key="r_abs")
        material_r = st.selectbox("屏蔽材料", TVL_KEYS, key="r_mat")
        mode_r = st.selectbox("计算模式", ["Instantaneous Dose Rate", "Weekly Dose"], key="r_mode")
        if mode_r == "Instantaneous Dose Rate":
            dose_limit_r = st.number_input("目标剂量率 (μSv/h)", value=DEFAULT_DOSE_RATE_LIMIT, step=0.1, key="r_dose_limit")
            use_ct_r = st.checkbox("考虑CT贡献", value=False, key="r_ct")
            ct_dose_r = st.number_input("CT剂量率 (μSv/h)", value=1.0, step=0.1, disabled=not use_ct_r, key="r_ct_dose")
        else:
            workload_r = st.number_input("周工作量 (MBq·min)", value=1665000, step=10000, key="r_workload")
            occ_r = st.number_input("占用因子", value=0.25, step=0.05, key="r_occ")
            week_limit_r = st.number_input("周剂量限值 (μSv)", value=20, step=1, key="r_week_limit")

        st.subheader("门与窗")
        door1_w = st.number_input("门1宽度 (m)", value=1.0, step=0.1, key="r_door1_w")
        door1_side = st.selectbox("门1侧边", ["N","S","E","W"], key="r_door1_side")
        door1_pos = st.number_input("门1位置 (m)", value=3.0, step=0.1, key="r_door1_pos")
        door2_w = st.number_input("门2宽度 (m)", value=0.9, step=0.1, key="r_door2_w")
        door2_side = st.selectbox("门2侧边", ["N","S","E","W"], key="r_door2_side")
        door2_pos = st.number_input("门2位置 (m)", value=2.0, step=0.1, key="r_door2_pos")
        win_w = st.number_input("窗宽度 (m)", value=0.8, step=0.1, key="r_win_w")
        win_side = st.selectbox("窗侧边", ["N","S","E","W"], key="r_win_side")
        win_pos = st.number_input("窗位置 (m)", value=3.0, step=0.1, key="r_win_pos")

        calc_btn_r = st.button("计算房间", key="r_calc")

    # 主区域显示结果
    if calc_btn_r or 'r_calc_done' not in st.session_state:
        # 获取当前活度
        try:
            if decay_enabled_r:
                A1, _, _ = decay_correction(ref_act1_r, ref_time1_r, cur_time_r, nuclide_r)
                A2, _, _ = decay_correction(ref_act2_r, ref_time2_r, cur_time_r, nuclide_r)
            else:
                A1 = ref_act1_r
                A2 = ref_act2_r
        except Exception as e:
            st.error(f"衰变修正错误: {e}")
            st.stop()
        # 验证坐标
        try:
            if L <= 0 or W <= 0 or H <= 0:
                raise ValueError("房间尺寸必须 > 0")
            for (x,y,z) in [(x1,y1,z1),(x2,y2,z2)]:
                if not (0 <= x <= L and 0 <= y <= W and 0 <= z <= H):
                    raise ValueError(f"源坐标 ({x},{y},{z}) 超出房间范围")
        except ValueError as e:
            st.error(f"输入错误: {e}")
            st.stop()
        # 获取核素数据
        data = NUCLIDE_DATA[nuclide_r]
        gamma_const = data["gamma_const"]
        tvl_dict = data["tvl"]
        if material_r not in tvl_dict:
            st.error(f"核素 {nuclide_r} 不支持材料 {material_r}")
            st.stop()
        # 准备计算
        if mode_r == "Weekly Dose":
            st.warning("双源模式下推荐使用瞬时剂量率模式，将自动切换。")
            mode_r = "Instantaneous Dose Rate"
        dose_limit = dose_limit_r if mode_r == "Instantaneous Dose Rate" else 20  # 备用
        use_ct = use_ct_r if mode_r == "Instantaneous Dose Rate" else False
        ct_dose = ct_dose_r if use_ct else 0.0

        sources = [(A1, (x1,y1,z1)), (A2, (x2,y2,z2))]
        # 定义关注点
        calc_points = {}
        wall_centers = {
            'N': (L/2, W, z1),
            'S': (L/2, 0, z1),
            'E': (L, W/2, z1),
            'W': (0, W/2, z1)
        }
        for side, (cx,cy,cz) in wall_centers.items():
            calc_points[f"Wall {side}"] = (cx, cy, cz)
        calc_points["Ceiling"] = (L/2, W/2, H)
        calc_points["Floor"] = (L/2, W/2, 0)

        def get_point_on_wall(side, pos, z_level):
            if side == 'N': return (pos, W, z_level)
            elif side == 'S': return (pos, 0, z_level)
            elif side == 'E': return (L, pos, z_level)
            elif side == 'W': return (0, pos, z_level)
            else: raise ValueError("Side must be N/S/E/W")

        for label, (side, pos) in [("Door1", (door1_side, door1_pos)), 
                                   ("Door2", (door2_side, door2_pos)),
                                   ("Window", (win_side, win_pos))]:
            if side in ['N','S','E','W']:
                calc_points[label] = get_point_on_wall(side, pos, z1)

        # 计算每个点
        result_points = {}
        for name, (px,py,pz) in calc_points.items():
            total_dose = 0.0
            dists = []
            for (act, (sx,sy,sz)) in sources:
                dist = math.dist((sx,sy,sz), (px,py,pz)) + offset_r
                dists.append(dist)
                dose = compute_unshielded_dose_rate(act, dist, abs_r, gamma_const)
                total_dose += dose
            if use_ct:
                total_dose += ct_dose
            if total_dose <= dose_limit:
                thick = 0.0
            else:
                B = dose_limit / total_dose
                thick = tvl_dict[material_r] * math.log10(1.0 / B)
            result_points[name] = (px,py,pz, dists, total_dose, thick)

        # 输出表格
        st.subheader("计算结果")
        table_data = []
        for name, (px,py,pz, dists, total_dose, thick) in result_points.items():
            dist_str = " | ".join([f"{d:.2f}" for d in dists])
            table_data.append({"点位": name, "距离源1/源2 (m)": dist_str, 
                               "总剂量率 (μSv/h)": f"{total_dose:.3f}", 
                               "所需厚度 (mm)": f"{thick:.2f}"})
        st.table(table_data)

        # 绘图：2D + 3D
        st.subheader("房间布局图")
        fig_room = plt.figure(figsize=(12,5))
        ax2 = fig_room.add_subplot(121)
        ax3 = fig_room.add_subplot(122, projection='3d')

        # 2D top view
        ax2.add_patch(patches.Rectangle((0,0), L, W, fill=True, facecolor='lightgray', edgecolor='black', linewidth=2))
        ax2.text(L/2, -0.3, f"{L} m", ha='center', va='top')
        ax2.text(-0.3, W/2, f"{W} m", ha='right', va='center', rotation=90)
        colors = ['red', 'magenta']
        for i, (x,y,z) in enumerate([(x1,y1,z1),(x2,y2,z2)]):
            ax2.plot(x, y, marker='*', color=colors[i], markersize=12, label=f'源{i+1}')
        for label, (px,py,pz, dists, total_dose, thick) in result_points.items():
            if 'Wall' in label:
                marker = 's'; color='blue'; size=8
            elif label in ['Ceiling','Floor']:
                marker = '^'; color='green'; size=8
            else:
                marker = 'D'; color='orange'; size=8
            ax2.plot(px, py, marker=marker, color=color, markersize=size)
            ax2.text(px+0.1, py+0.1, f"{thick:.1f}mm", fontsize=7, color='darkblue')
        # 门/窗标记
        for label in ['Door1','Door2','Window']:
            if label in result_points:
                px,py,pz,_,_,_ = result_points[label]
                if label == 'Door1':
                    lc='k-'; lt='Door1'
                elif label == 'Door2':
                    lc='k-'; lt='Door2'
                else:
                    lc='b-'; lt='Window'
                # 判断墙面
                if abs(py - W) < 0.01:
                    ax2.plot([px-0.2, px+0.2], [W, W], lc, linewidth=3)
                    ax2.text(px, W+0.2, lt, ha='center', fontsize=8)
                elif abs(py) < 0.01:
                    ax2.plot([px-0.2, px+0.2], [0, 0], lc, linewidth=3)
                    ax2.text(px, -0.2, lt, ha='center', fontsize=8)
                elif abs(px - L) < 0.01:
                    ax2.plot([L, L], [py-0.2, py+0.2], lc, linewidth=3)
                    ax2.text(L+0.2, py, lt, va='center', fontsize=8)
                elif abs(px) < 0.01:
                    ax2.plot([0, 0], [py-0.2, py+0.2], lc, linewidth=3)
                    ax2.text(-0.2, py, lt, va='center', ha='right', fontsize=8)
        ax2.set_xlim(-0.5, L+0.5); ax2.set_ylim(-0.5, W+0.5)
        ax2.set_aspect('equal')
        ax2.set_title("俯视图 (XY)")
        ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        # 3D
        floor_pts = [(0,0,0), (L,0,0), (L,W,0), (0,W,0), (0,0,0)]
        fx,fy,fz = zip(*floor_pts)
        ax3.plot(fx, fy, fz, 'k-', linewidth=1)
        ceil_pts = [(0,0,H), (L,0,H), (L,W,H), (0,W,H), (0,0,H)]
        cx,cy,cz = zip(*ceil_pts)
        ax3.plot(cx, cy, cz, 'k-', linewidth=1)
        for (x,y) in [(0,0), (L,0), (L,W), (0,W)]:
            ax3.plot([x,x], [y,y], [0,H], 'k--', linewidth=0.8, alpha=0.5)
        for i, (x,y,z) in enumerate([(x1,y1,z1),(x2,y2,z2)]):
            ax3.scatter(x, y, z, color=colors[i], s=80, marker='*', label=f'源{i+1}')
        for label, (px,py,pz, dists, total_dose, thick) in result_points.items():
            if 'Wall' in label:
                marker='s'; color='blue'; size=60
            elif label in ['Ceiling','Floor']:
                marker='^'; color='green'; size=60
            else:
                marker='D'; color='orange'; size=60
            ax3.scatter(px, py, pz, marker=marker, color=color, s=size)
            ax3.text(px, py, pz, f"{thick:.1f}", fontsize=7, color='darkred')
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        ax3.set_title("3D视图 (双源)")
        ax3.view_init(elev=25, azim=-60)
        ax3.legend(loc='upper left', fontsize=8)
        fig_room.tight_layout()
        st.pyplot(fig_room)
