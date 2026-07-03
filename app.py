#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Medicine Shielding Calculator - Web App
Streamlit 版本，适配手机与桌面
基于 V2.3 核心逻辑
制作者: Fang KeMing & DeepSeek
版本: Web 1.2
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
from datetime import datetime, timedelta

# ============================================================
# 1. 修复 matplotlib 中文显示问题（适用于 Streamlit Cloud）
# ============================================================
def set_chinese_font():
    """自动设置 matplotlib 中文字体，适配 Linux 云端环境"""
    # 如果是 Linux 系统（Streamlit Cloud 运行环境）
    if platform.system() == 'Linux':
        # 方法：查找系统中已安装的中文字体
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # 过滤出可能包含中文的字体
        chinese_keywords = ['cjk', 'hei', 'song', 'ming', 'noto', 'sc', 'cn', 'chinese']
        chinese_fonts = [f for f in font_list if any(key in f.lower() for key in chinese_keywords)]
        if chinese_fonts:
            # 使用第一个找到的中文字体
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
        else:
            # 如果没有找到，使用通用的 sans-serif 并尝试指定常见中文字体名
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    else:
        # Windows 或 macOS 通常自带中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
    # 解决负号显示异常
    plt.rcParams['axes.unicode_minus'] = False

# 在程序启动时调用一次
set_chinese_font()

# ============================================================
# 2. 核素数据定义
# ============================================================
ISOTOPES = {
    "F-18": 109.8,      # 分钟
    "Ga-68": 67.7,
    "C-11": 20.3,
    "O-15": 2.0,
    "N-13": 10.0,
    "自定义": None      # 自定义时手动输入半衰期
}

# ============================================================
# 3. 页面标题和说明
# ============================================================
st.set_page_config(page_title="核医学衰变计算器", layout="centered")
st.title("☢️ 核医学衰变计算器")
st.markdown("计算放射性核素经过一定时间后的剩余活度，并绘制衰变曲线。")

# ============================================================
# 4. 用户输入区域（侧边栏 + 主界面）
# ============================================================
with st.sidebar:
    st.header("参数设置")
    # 核素选择
    isotope = st.selectbox("选择核素", list(ISOTOPES.keys()))
    
    if isotope == "自定义":
        half_life = st.number_input("输入半衰期（分钟）", min_value=0.1, value=60.0, step=1.0, format="%.1f")
    else:
        half_life = ISOTOPES[isotope]
        st.write(f"半衰期：**{half_life}** 分钟")
    
    # 初始活度
    initial_activity = st.number_input("初始活度 (MBq)", min_value=0.0, value=100.0, step=10.0, format="%.1f")
    
    # 衰变时间（支持多种单位）
    time_unit = st.selectbox("时间单位", ["分钟", "小时", "天"])
    time_value = st.number_input(f"请输入时间（{time_unit}）", min_value=0.0, value=10.0, step=1.0)
    
    # 转换为分钟
    if time_unit == "小时":
        time_minutes = time_value * 60
    elif time_unit == "天":
        time_minutes = time_value * 1440
    else:
        time_minutes = time_value
    
    # 绘图时间范围（从0到输入时间，可调）
    plot_duration = st.number_input("绘图时间范围（分钟）", min_value=1, value=max(60, int(time_minutes * 1.5)), step=10)

# ============================================================
# 5. 核心计算
# ============================================================
if half_life is None or half_life <= 0:
    st.error("请正确设置半衰期！")
    st.stop()

decay_constant = np.log(2) / half_life   # λ = ln2 / T½
remaining_activity = initial_activity * np.exp(-decay_constant * time_minutes)

# 计算衰变到某个时间的活度（用于绘制曲线）
time_points = np.linspace(0, plot_duration, 500)
activity_points = initial_activity * np.exp(-decay_constant * time_points)

# ============================================================
# 6. 结果显示
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("半衰期", f"{half_life:.1f} 分钟")
col2.metric("经过时间", f"{time_minutes:.1f} 分钟")
col3.metric("剩余活度", f"{remaining_activity:.2f} MBq")

# 显示一个简单的衰变表（每隔一定间隔）
st.subheader("📊 衰变数据表")
interval = max(1, int(plot_duration // 20))  # 自动分20个点
table_times = np.arange(0, min(plot_duration, time_minutes * 2) + interval, interval)
table_activities = initial_activity * np.exp(-decay_constant * table_times)
data = {"时间 (分钟)": table_times, "活度 (MBq)": table_activities}
st.dataframe(data, use_container_width=True)

# ============================================================
# 7. 绘制衰变曲线（含中文）
# ============================================================
st.subheader("📈 衰变曲线")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_points, activity_points, label="活度衰变", color='#1f77b4', linewidth=2)
# 标记当前时间点
ax.scatter([time_minutes], [remaining_activity], color='red', s=80, zorder=5, label=f"当前 ({time_minutes:.1f} min)")
ax.axvline(x=time_minutes, color='gray', linestyle='--', alpha=0.7)
ax.axhline(y=remaining_activity, color='gray', linestyle='--', alpha=0.7)

ax.set_xlabel("时间 (分钟)", fontsize=12)
ax.set_ylabel("活度 (MBq)", fontsize=12)
ax.set_title("放射性衰变曲线", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 在图中显示公式
formula = f"A(t) = {initial_activity:.1f} · exp(-{decay_constant:.4f} · t)"
ax.text(0.02, 0.95, formula, transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

st.pyplot(fig)

# ============================================================
# 8. 附加信息（核素特性等）
# ============================================================
with st.expander("ℹ️ 常见核素半衰期"):
    st.write("""
    - **F-18**：109.8 分钟（约 1.83 小时）
    - **Ga-68**：67.7 分钟（约 1.13 小时）
    - **C-11**：20.3 分钟
    - **O-15**：2.0 分钟
    - **N-13**：10.0 分钟
    """)
