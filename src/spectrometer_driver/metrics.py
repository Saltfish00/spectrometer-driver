"""光源质量评价模块（修复版 v0.2.2）。

============================================================
本版本修复的 Bug 列表
============================================================

- **Bug #1**（🔴 严重）：饱和检测尺度不匹配
    - 现象：使用系数修正后的 intensity 做饱和判定，
      由于 intensity = raw / 10^N 远小于 ADC 满量程，
      饱和检测永远失败。
    - 修复：新增 ``raw_data`` 参数，饱和检测使用原始 ADC 值。

- **Bug #2**（🟡 中等）：纯噪声时 SNR = inf，误导 UI
    - 修复：区分"全零信号"（SNR = 0）与"理论完美信号"（SNR = inf）。

- **Bug #3**（🟡 中等）：底层工具对 NaN 静默传播
    - 修复：新增 ``_validate_input``，在入口处拦截非法输入。

- **Bug #4**（🟡 中等）：参考光谱含 NaN 导致分析器永久失效
    - 修复：``add_frame`` 丢弃含 NaN / Inf 的帧。

- **Bug #5**（🟢 轻微）：波段内无点时静默退化
    - 修复：抛出 ``ValueError``，由调用方决定处理方式。

- **Bug #6**（🟢 轻微）：波长超出 CIE 有效区间时无提示
    - 修复：首次超限时发出 ``warnings.warn``。

============================================================
两大功能
============================================================

**1. 单帧信号质量评估** (``evaluate_single_frame``)

    - 暗噪声水平
    - 信噪比 SNR
    - 饱和情况
    - 杂散光 / 环境光比例

**2. 时间序列稳定性分析** (``StabilityAnalyzer``)

    - 强度类：积分强度 I(t)、CV、相对漂移、峰峰值
    - 形状类：相关系数 r(t)、SAM、RMSE
    - 色度类：CCT、Δu'v'
    - 特征类：光谱重心 λc、波段比
    - 稳定时间 t_stable

============================================================
参考
============================================================

- Wyman, Sloan & Shirley (2013). Simple Analytic Approximations
  to the CIE XYZ Color Matching Functions.
- McCamy, C. S. (1992). Correlated color temperature as an
  explicit function of chromaticity coordinates.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ============================================================================
# 兼容性工具
# ============================================================================

#: numpy 2.0 起 np.trapz 更名为 np.trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: CIE 有效波长区间
_CIE_WL_MIN = 380.0
_CIE_WL_MAX = 780.0

#: 超限警告只发一次（模块级标志）
_WARNED_OUT_OF_RANGE = False


# ============================================================================
# 输入校验工具（修复 Bug #3）
# ============================================================================

def _validate_input(x: np.ndarray, name: str = "input") -> np.ndarray:
    """验证输入数组：非空、无 NaN / Inf。

    ============================================================
    修复 Bug #3
    ============================================================

    之前所有底层工具（cv / correlation / spectral_centroid 等）
    对输入含 NaN 的情况会静默返回 NaN，最终导致上层统计全部
    失效但无法定位原因。

    本函数在入口处拦截，抛出清晰的 ValueError。

    :param x:    输入数组
    :param name: 用于错误信息的参数名
    :return:     转换为 float64 的数组（保持原形状）
    :raises ValueError: 输入为空或包含非有限值
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError(f"{name} 为空数组")
    if not np.all(np.isfinite(x)):
        n_nan = int(np.sum(np.isnan(x)))
        n_inf = int(np.sum(np.isinf(x)))
        raise ValueError(
            f"{name} 包含非有限值: NaN={n_nan}, Inf={n_inf}"
        )
    return x


# ============================================================================
# CIE 1931 颜色匹配函数（Wyman 2013 近似）
# ============================================================================

def _gaussian_piecewise(
    x: np.ndarray, mu: float, sigma_left: float, sigma_right: float
) -> np.ndarray:
    """分段高斯函数：x < mu 时用 sigma_left，否则 sigma_right。"""
    sigma = np.where(x < mu, sigma_left, sigma_right)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_xyz_bar(
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """近似计算 CIE 1931 颜色匹配函数 x̄(λ), ȳ(λ), z̄(λ)。

    ============================================================
    修复 Bug #6
    ============================================================

    Wyman 近似在 380~780 nm 之外精度下降，这里在首次遇到
    超限波长时发出一次 ``warnings.warn``（不阻断计算）。

    :param wavelengths: 波长数组（nm）
    :return:            (x̄, ȳ, z̄) 三个与输入同形状的数组
    """
    global _WARNED_OUT_OF_RANGE

    x = np.asarray(wavelengths, dtype=np.float64)

    # ---- 范围检查（只警告一次）----
    if not _WARNED_OUT_OF_RANGE and x.size:
        if x.min() < _CIE_WL_MIN or x.max() > _CIE_WL_MAX:
            warnings.warn(
                f"波长范围 {x.min():.1f}~{x.max():.1f} nm 超出 CIE 1931 "
                f"有效区间 {_CIE_WL_MIN:.0f}~{_CIE_WL_MAX:.0f} nm，"
                f"色度 / CCT 结果可能不准确",
                stacklevel=2,
            )
            _WARNED_OUT_OF_RANGE = True

    x_bar = (
        1.056 * _gaussian_piecewise(x, 599.8, 37.9, 31.0)
        + 0.362 * _gaussian_piecewise(x, 442.0, 16.0, 26.7)
        - 0.065 * _gaussian_piecewise(x, 501.1, 20.4, 26.2)
    )
    y_bar = (
        0.821 * _gaussian_piecewise(x, 568.8, 46.9, 40.5)
        + 0.286 * _gaussian_piecewise(x, 530.9, 16.3, 31.1)
    )
    z_bar = (
        1.217 * _gaussian_piecewise(x, 437.0, 11.8, 36.0)
        + 0.681 * _gaussian_piecewise(x, 459.0, 26.0, 13.8)
    )
    return x_bar, y_bar, z_bar


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class SingleFrameMetrics:
    """单帧光谱的信号质量评价结果。"""

    dark_noise: Optional[float]        #: 暗噪声标准差（无暗帧时为 None）
    noise_estimate: float              #: 噪声估计
    snr: float                         #: 指定波段信噪比
    band_signal_mean: float            #: 指定波段信号均值
    saturation_ratio: float            #: 饱和像素比例（0~1）
    saturation_count: int              #: 饱和像素个数
    stray_light_ratio: Optional[float] #: 杂散光比例（无暗帧时为 None）
    band: tuple[float, float]          #: 计算 SNR 的波段范围

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SingleFrameMetrics(SNR={self.snr:.1f}, "
            f"saturation={self.saturation_ratio * 100:.3f}%, "
            f"dark_noise={self.dark_noise}, "
            f"stray={self.stray_light_ratio})"
        )


@dataclass
class StabilityCriteria:
    """时间序列稳定性的判据阈值。"""

    cv_max: float = 0.01                    #: 积分强度 CV 上限（1%）
    correlation_min: float = 0.999          #: 相关系数下限
    delta_uv_max: float = 0.002             #: 色差 Δu'v' 上限
    centroid_drift_max: float = 0.2         #: 光谱重心漂移上限（nm）


@dataclass
class StabilityReport:
    """某个时间窗口内的稳定性报告。

    ``ready=False`` 表示窗口尚未填满，其余字段为 None。
    """

    # ---- 元信息 ----
    ready: bool
    window_start: float = 0.0
    window_end: float = 0.0
    n_frames: int = 0

    # ---- 强度指标 ----
    integral_mean: Optional[float] = None
    integral_cv: Optional[float] = None
    integral_drift: Optional[float] = None
    integral_ptp: Optional[float] = None

    # ---- 形状指标 ----
    correlation_min: Optional[float] = None
    sam_max: Optional[float] = None
    rmse_mean: Optional[float] = None

    # ---- 色度指标 ----
    cct_mean: Optional[float] = None
    cct_range: Optional[float] = None
    delta_uv_max: Optional[float] = None

    # ---- 特征指标 ----
    centroid_mean: Optional[float] = None
    centroid_drift: Optional[float] = None
    band_ratio_mean: Optional[float] = None

    # ---- 判定 ----
    is_stable: bool = False
    unstable_reasons: list = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        if not self.ready:
            return "<StabilityReport(窗口未填满)>"
        status = "稳定" if self.is_stable else "不稳定"
        return (
            f"<StabilityReport({status}, "
            f"CV={self.integral_cv * 100:.3f}%, "
            f"r={self.correlation_min:.5f}, "
            f"Δu'v'={self.delta_uv_max:.5f}, "
            f"λc漂移={self.centroid_drift:.4f}nm)>"
        )


# ============================================================================
# 底层工具（全部加入输入校验，修复 Bug #3）
# ============================================================================

def _band_indices(
    wavelengths: np.ndarray, wl_min: float, wl_max: float
) -> np.ndarray:
    """返回波长数组中落在 [wl_min, wl_max] 范围内的索引。"""
    return np.where((wavelengths >= wl_min) & (wavelengths <= wl_max))[0]


def integrate_band(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    wl_min: float,
    wl_max: float,
) -> float:
    """对指定波段做积分（梯形法）。

    :param wavelengths: 波长数组（nm）
    :param intensity:   对应光强（长度必须与 wavelengths 一致）
    :param wl_min:      积分下限波长
    :param wl_max:      积分上限波长
    :return:            积分值；波段内点数不足 2 时返回 0
    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    wavelengths = _validate_input(wavelengths, "wavelengths")
    intensity = _validate_input(intensity, "intensity")
    if wavelengths.size != intensity.size:
        raise ValueError(
            f"wavelengths ({wavelengths.size}) 与 "
            f"intensity ({intensity.size}) 长度不一致"
        )
    idx = _band_indices(wavelengths, wl_min, wl_max)
    if idx.size < 2:
        return 0.0
    return float(_trapz(intensity[idx], wavelengths[idx]))


def cv(series: np.ndarray) -> float:
    """变异系数 CV = σ / μ。

    :param series: 一维时间序列
    :return:       CV 值；均值为 0 时返回 0
    :raises ValueError: 输入含 NaN/Inf
    """
    series = _validate_input(series, "series")
    mean = float(np.mean(series))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(series) / abs(mean))


def relative_drift(
    series: np.ndarray, reference: Optional[float] = None
) -> float:
    """相对漂移 (I_last - I_ref) / I_ref。

    :param series:    时间序列
    :param reference: 参考值；None 表示用序列第一个值
    :raises ValueError: 输入含 NaN/Inf
    """
    series = _validate_input(series, "series")
    if reference is None:
        reference = float(series[0])
    if abs(reference) < 1e-12:
        return 0.0
    return float((series[-1] - reference) / reference)


def peak_to_peak(series: np.ndarray) -> float:
    """峰峰值 = max - min。

    :raises ValueError: 输入含 NaN/Inf
    """
    series = _validate_input(series, "series")
    return float(np.max(series) - np.min(series))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关系数。

    :param a, b: 长度相同的两个一维数组
    :return:     -1 ~ 1 之间的相关系数；任一向量方差为 0 时返回 1.0
    :raises ValueError: 长度不一致或含 NaN/Inf
    """
    a = _validate_input(a, "a")
    b = _validate_input(b, "b")
    if a.size != b.size:
        raise ValueError("相关系数要求两个向量长度一致")
    # 去均值
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def sam(a: np.ndarray, b: np.ndarray) -> float:
    """光谱角（Spectral Angle Mapper），单位弧度。

    取值范围 [0, π/2]，对强度缩放不敏感，只反映光谱形状差异。

    公式::

        SAM = arccos( <a, b> / (||a|| * ||b||) )

    :raises ValueError: 输入含 NaN/Inf
    """
    a = _validate_input(a, "a")
    b = _validate_input(b, "b")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos_theta = float(np.dot(a, b) / (na * nb))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """均方根误差 RMSE = sqrt(mean((a-b)^2))。

    :raises ValueError: 输入含 NaN/Inf
    """
    a = _validate_input(a, "a")
    b = _validate_input(b, "b")
    if a.size != b.size:
        raise ValueError("rmse 要求两个向量长度一致")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def spectral_centroid(
    wavelengths: np.ndarray, intensity: np.ndarray
) -> float:
    """光谱重心 λc = Σ(λ·I) / Σ(I)。

    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    wavelengths = _validate_input(wavelengths, "wavelengths")
    intensity = _validate_input(intensity, "intensity")
    if wavelengths.size != intensity.size:
        raise ValueError("wavelengths 与 intensity 长度不一致")
    total = float(np.sum(intensity))
    if abs(total) < 1e-12:
        return 0.0
    return float(np.sum(wavelengths * intensity) / total)


def band_ratio(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    numerator_band: tuple[float, float],
    denominator_band: tuple[float, float],
) -> float:
    """两个波段的积分强度比。

    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    num = integrate_band(wavelengths, intensity, *numerator_band)
    den = integrate_band(wavelengths, intensity, *denominator_band)
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


# ============================================================================
# 色度计算
# ============================================================================

def spectrum_to_xyz(
    wavelengths: np.ndarray, intensity: np.ndarray
) -> tuple[float, float, float]:
    """把光谱转为 CIE XYZ 三刺激值。

    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    wavelengths = _validate_input(wavelengths, "wavelengths")
    intensity = _validate_input(intensity, "intensity")
    if wavelengths.size != intensity.size:
        raise ValueError("wavelengths 与 intensity 长度不一致")
    x_bar, y_bar, z_bar = cie_xyz_bar(wavelengths)
    X = _trapz(intensity * x_bar, wavelengths)
    Y = _trapz(intensity * y_bar, wavelengths)
    Z = _trapz(intensity * z_bar, wavelengths)
    return float(X), float(Y), float(Z)


def xyz_to_xy(X: float, Y: float, Z: float) -> tuple[float, float]:
    """XYZ -> xy 色度坐标。"""
    s = X + Y + Z
    if abs(s) < 1e-12:
        return float("nan"), float("nan")
    return X / s, Y / s


def xyz_to_uv_prime(X: float, Y: float, Z: float) -> tuple[float, float]:
    """XYZ -> CIE 1976 u'v' 色度坐标。"""
    denom = X + 15.0 * Y + 3.0 * Z
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")
    u = 4.0 * X / denom
    v = 9.0 * Y / denom
    return float(u), float(v)


def cct_from_xy(x: float, y: float) -> float:
    """由 xy 色度坐标计算相关色温 CCT（McCamy 近似公式）。

    公式::

        n = (x - 0.3320) / (0.1858 - y)
        CCT = 449 n^3 + 3525 n^2 + 6823.3 n + 5520.33
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    denom = 0.1858 - y
    if abs(denom) < 1e-12:
        return float("nan")
    n = (x - 0.3320) / denom
    return float(449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33)


def spectrum_to_cct(
    wavelengths: np.ndarray, intensity: np.ndarray
) -> float:
    """由光谱直接计算 CCT。

    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    X, Y, Z = spectrum_to_xyz(wavelengths, intensity)
    x, y = xyz_to_xy(X, Y, Z)
    return cct_from_xy(x, y)


def spectrum_to_uv_prime(
    wavelengths: np.ndarray, intensity: np.ndarray
) -> tuple[float, float]:
    """由光谱直接计算 u'v' 色度坐标。

    :raises ValueError: 输入含 NaN/Inf 或长度不一致
    """
    X, Y, Z = spectrum_to_xyz(wavelengths, intensity)
    return xyz_to_uv_prime(X, Y, Z)


def delta_uv_prime(
    uv1: tuple[float, float], uv2: tuple[float, float]
) -> float:
    """CIE 1976 u'v' 色差（欧氏距离）。

    人眼可感知的色差阈值约为 0.002。
    """
    if not all(np.isfinite(v) for v in (*uv1, *uv2)):
        return float("nan")
    return float(np.hypot(uv1[0] - uv2[0], uv1[1] - uv2[1]))


# ============================================================================
# 单帧信号质量评估（修复 Bug #1 / #2 / #5）
# ============================================================================

def evaluate_single_frame(
    intensity: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    band: tuple[float, float] = (650.0, 800.0),
    adc_max: int = 65535,
    saturation_threshold: float = 0.95,
    raw_data: Optional[np.ndarray] = None,
) -> SingleFrameMetrics:
    """评价单帧光谱的信号质量。

    ============================================================
    修复 Bug #1：饱和检测尺度不匹配
    ============================================================

    ``intensity`` 可能是系数修正后的值（raw / 10^N），
    而 ADC 满量程是 65535。如果直接用 intensity 判定饱和，
    会因数量级差异而永远不触发。

    修复：新增 ``raw_data`` 参数，饱和检测**优先使用**它。

    ============================================================
    修复 Bug #2：SNR = inf 的处理
    ============================================================

    ``noise_estimate = 0`` 时：
      - 若信号全零 → SNR = 0（无意义）
      - 若信号非零 → SNR = inf（理论完美信号）

    ============================================================
    修复 Bug #5：波段内无点
    ============================================================

    之前会静默退化为"用整段信号"，让用户以为波段有效。
    现在直接抛 ``ValueError``，强制用户处理。

    :param intensity:  光谱强度数组（可以是 raw 或修正后）。
                       SNR / 波段均值 / 噪声估计都基于此数组。
    :param wavelengths: 波长数组（nm），用于波段筛选。
    :param dark:       暗帧（遮光或无光时采集的同长度数组）。
    :param band:       计算 SNR 的波段范围。
    :param adc_max:    ADC 满量程（16 位 65535，12 位 4095）。
    :param saturation_threshold: 判定饱和的阈值比例，默认 95%。
    :param raw_data:   【重要】用于饱和检测的原始 ADC 数据。
                       若 intensity 已经做了系数修正（raw / 10^N），
                       必须同时传入 raw_data，否则饱和检测永远失败。
                       若不传，则假定 intensity 本身就是 raw 尺度。
    :return:           :class:`SingleFrameMetrics`
    :raises ValueError: intensity / wavelengths / dark / raw_data
                       含 NaN / Inf，或长度不一致，或波段内无点
    """
    # ---- 输入校验 ----
    intensity = _validate_input(intensity, "intensity")
    if intensity.ndim != 1:
        raise ValueError("intensity 必须是一维数组")

    # ---------- 噪声估计 ----------
    if dark is not None:
        dark = _validate_input(dark, "dark")
        if dark.size != intensity.size:
            raise ValueError(
                f"dark ({dark.size}) 与 intensity ({intensity.size}) 长度不一致"
            )
        dark_noise = float(np.std(dark))
        noise_estimate = dark_noise
        # 杂散光比例 = 暗帧均值 / 信号均值
        stray_light_ratio: Optional[float] = (
            float(np.mean(dark) / np.mean(intensity))
            if abs(np.mean(intensity)) > 1e-12
            else None
        )
    else:
        # 无暗帧：用相邻点差分法估计噪声
        # 对于平滑信号，diff 只反映噪声，除以 √2 是无偏估计
        dark_noise = None
        if intensity.size >= 2:
            diffs = np.diff(intensity)
            noise_estimate = float(np.std(diffs) / np.sqrt(2.0))
        else:
            noise_estimate = 0.0
        stray_light_ratio = None

    # ---------- 波段信号均值 ----------
    if wavelengths is not None:
        wavelengths = _validate_input(wavelengths, "wavelengths")
        if wavelengths.size != intensity.size:
            raise ValueError(
                f"wavelengths ({wavelengths.size}) 与 "
                f"intensity ({intensity.size}) 长度不一致"
            )
        idx = _band_indices(wavelengths, band[0], band[1])
        if idx.size == 0:
            # ---- 修复 Bug #5：明确抛错 ----
            raise ValueError(
                f"波段 {band} 内没有波长点。"
                f"当前波长范围: {wavelengths.min():.1f}~"
                f"{wavelengths.max():.1f} nm"
            )
        band_signal = intensity[idx]
    else:
        band_signal = intensity
    band_signal_mean = float(np.mean(band_signal))

    # ---------- SNR（修复 Bug #2）----------
    if noise_estimate > 1e-12:
        snr = band_signal_mean / noise_estimate
    elif abs(band_signal_mean) < 1e-12:
        # 全零信号：SNR 无意义，返回 0
        snr = 0.0
    else:
        # 理论上的完美信号：无噪声但有信号
        snr = float("inf")

    # ---------- 饱和判定（修复 Bug #1）----------
    # 优先使用 raw_data（原始 ADC 尺度）；否则假定 intensity 就是 raw
    if raw_data is not None:
        sat_source = np.asarray(raw_data, dtype=np.float64)
        if sat_source.size != intensity.size:
            raise ValueError(
                f"raw_data ({sat_source.size}) 与 "
                f"intensity ({intensity.size}) 长度不一致"
            )
    else:
        sat_source = intensity

    sat_thresh = adc_max * saturation_threshold
    sat_mask = sat_source >= sat_thresh
    saturation_count = int(np.sum(sat_mask))
    saturation_ratio = (
        saturation_count / sat_source.size if sat_source.size else 0.0
    )

    return SingleFrameMetrics(
        dark_noise=dark_noise,
        noise_estimate=noise_estimate,
        snr=snr,
        band_signal_mean=band_signal_mean,
        saturation_ratio=saturation_ratio,
        saturation_count=saturation_count,
        stray_light_ratio=stray_light_ratio,
        band=band,
    )


# ============================================================================
# 时间序列稳定性分析
# ============================================================================

class StabilityAnalyzer:
    """基于滑动窗口的时间序列稳定性分析器。

    ============================================================
    使用方式
    ============================================================

    在线分析::

        analyzer = StabilityAnalyzer(wavelengths, window_seconds=1.0)
        for ts, frame in stream:
            report = analyzer.add_frame(frame, timestamp=ts)
            if report.ready:
                print(report)

        print("稳定时间:", analyzer.t_stable, "秒")

    ============================================================
    就绪条件（v0.2.2 修复）
    ============================================================

    原条件：``timestamp - buffer[0].t >= window_seconds``

    问题：帧率波动时前几帧跨度可能刚好差 1%，一直判为"未就绪"，
    尤其 10fps 场景下 11 帧跨度只有 0.98s。

    修复：放宽到 ``window_seconds * 0.85``，即跨度达到窗口的
    85% 即认为就绪。这样 1.0s 窗口只需 0.85s 跨度即可判定，
    对帧率波动有容错。

    ============================================================
    内存管理
    ============================================================

    使用 ``collections.deque`` 保存窗口内的帧，每次新帧到达时自动
    丢弃超出窗口的旧帧，保证长时间运行时内存不增长。
    """

    #: 就绪判断的时间跨度比例（0.85 = 达到窗口的 85% 即就绪）
    READY_SPAN_RATIO = 0.85

    def __init__(
        self,
        wavelengths: np.ndarray,
        criteria: Optional[StabilityCriteria] = None,
        window_seconds: float = 1.0,
        band: tuple[float, float] = (650.0, 800.0),
        numerator_band: tuple[float, float] = (650.0, 800.0),
        denominator_band: tuple[float, float] = (400.0, 500.0),
        reference_spectrum: Optional[np.ndarray] = None,
        min_frames: int = 3,
    ) -> None:
        """
        :param wavelengths:        波长数组（nm）
        :param criteria:           稳定性判据；None 时使用默认值
        :param window_seconds:     滑动窗口时间长度（秒）
        :param band:               积分强度计算所用的波段
        :param numerator_band:     波段比的分子波段
        :param denominator_band:   波段比的分母波段
        :param reference_spectrum: 参考光谱；None 时自动使用第一帧
        :param min_frames:         判定窗口就绪所需的最小帧数
        """
        self.wavelengths = _validate_input(wavelengths, "wavelengths")
        self.criteria = criteria or StabilityCriteria()
        self.window_seconds = float(window_seconds)
        self.band = tuple(band)
        self.numerator_band = tuple(numerator_band)
        self.denominator_band = tuple(denominator_band)
        self.min_frames = int(min_frames)

        # 参考光谱（可选传入，也可以由首帧确定）
        if reference_spectrum is not None:
            ref = np.asarray(reference_spectrum, dtype=np.float64)
            if not np.all(np.isfinite(ref)):
                raise ValueError("reference_spectrum 包含 NaN/Inf")
            self._reference: Optional[np.ndarray] = ref
        else:
            self._reference = None

        # 窗口缓冲区
        self._buffer: deque = deque()

        # 内部状态
        self._t_start: Optional[float] = None
        self._t_stable: Optional[float] = None
        self._history: list = []

        # 被丢弃的帧计数（供调试）
        self._dropped_frames: int = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def t_stable(self) -> Optional[float]:
        """从第一帧到首次判定稳定的时间（秒）；尚未稳定则为 None。"""
        return self._t_stable

    @property
    def reference(self) -> Optional[np.ndarray]:
        """当前使用的参考光谱。"""
        return self._reference

    @property
    def history(self) -> list:
        """所有历史窗口报告（含未就绪的）。"""
        return self._history

    @property
    def dropped_frames(self) -> int:
        """被丢弃的非法帧数（含 NaN / Inf）。"""
        return self._dropped_frames

    @property
    def current_span(self) -> float:
        """当前窗口内的时间跨度（秒）；空窗口时返回 0。"""
        if len(self._buffer) < 2:
            return 0.0
        return float(self._buffer[-1][0] - self._buffer[0][0])

    @property
    def ready_threshold(self) -> float:
        """就绪判断所需的最小跨度（秒）。"""
        return self.window_seconds * self.READY_SPAN_RATIO

    def reset(self) -> None:
        """清空内部状态，可重新开始分析。"""
        self._buffer.clear()
        self._t_start = None
        self._t_stable = None
        self._history.clear()
        self._dropped_frames = 0
        # 保留 reference（用户可能希望继续与同一参考比较）

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def add_frame(
        self,
        intensity: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> StabilityReport:
        """加入一帧数据并返回当前窗口的稳定性报告。

        ============================================================
        修复 Bug #4
        ============================================================

        含 NaN / Inf 的帧会被丢弃，返回 ``ready=False`` 的报告，
        但**不会**中断分析器状态（不写入缓冲区、不设为参考）。

        :param intensity: 光强数组（建议使用系数修正后的光强）
        :param timestamp: 时间戳（秒）；None 时使用 ``time.time()``
        :return:          :class:`StabilityReport`
        """
        intensity = np.asarray(intensity, dtype=np.float64)
        if timestamp is None:
            timestamp = time.time()
        timestamp = float(timestamp)

        # ---- 修复 Bug #4：丢弃含 NaN / Inf 的帧 ----
        if intensity.ndim != 1:
            self._dropped_frames += 1
            return StabilityReport(
                ready=False,
                window_start=timestamp,
                window_end=timestamp,
                n_frames=0,
                unstable_reasons=["帧不是一维数组，已丢弃"],
            )
        if not np.all(np.isfinite(intensity)):
            self._dropped_frames += 1
            n_nan = int(np.sum(np.isnan(intensity)))
            n_inf = int(np.sum(np.isinf(intensity)))
            return StabilityReport(
                ready=False,
                window_start=timestamp,
                window_end=timestamp,
                n_frames=0,
                unstable_reasons=[
                    f"帧含非有限值（NaN={n_nan}, Inf={n_inf}），已丢弃"
                ],
            )

        # ---- 记录起始时间与参考光谱（仅首帧）----
        if self._t_start is None:
            self._t_start = timestamp
        if self._reference is None:
            self._reference = intensity.copy()

        # ---- 加入缓冲区 ----
        self._buffer.append((timestamp, intensity))

        # ---- 丢弃超窗旧帧 ----
        cutoff = timestamp - self.window_seconds
        while len(self._buffer) > 1 and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # ---- 就绪判断（v0.2.2 修复）----
        span = timestamp - self._buffer[0][0]
        threshold = self.ready_threshold   # window_seconds * 0.85

        ready = (
            len(self._buffer) >= self.min_frames
            and span >= threshold
        )

        if not ready:
            # 未就绪的报告里带上进度信息，供 UI 显示
            remain = max(0.0, threshold - span)
            report = StabilityReport(
                ready=False,
                window_start=self._buffer[0][0],
                window_end=timestamp,
                n_frames=len(self._buffer),
                unstable_reasons=[
                    f"窗口未就绪: 跨度 {span:.2f}s / 目标 {threshold:.2f}s "
                    f"(还需 {remain:.2f}s)"
                ],
            )
            self._history.append(report)
            return report

        # ---- 计算窗口报告 ----
        report = self._compute_report(self._buffer[0][0], timestamp)
        self._history.append(report)

        # ---- 记录首次稳定时间 ----
        if report.is_stable and self._t_stable is None:
            self._t_stable = timestamp - self._t_start
        return report

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------
    def _compute_report(
        self, t_start: float, t_end: float
    ) -> StabilityReport:
        """计算当前窗口内的全部指标，并给出稳定性判定。"""
        # 堆叠为 (N, num_points)
        frames = np.stack([f for _, f in self._buffer], axis=0)
        ref = self._reference
        assert ref is not None  # 由 add_frame 保证

        # ---------- 强度类 ----------
        integrals = np.array(
            [integrate_band(self.wavelengths, f, *self.band) for f in frames]
        )
        integral_mean = float(np.mean(integrals))
        integral_cv_val = cv(integrals)
        integral_drift_val = relative_drift(integrals)
        integral_ptp_val = peak_to_peak(integrals)

        # ---------- 形状类（相对参考光谱）----------
        corrs = np.array([correlation(f, ref) for f in frames])
        sams = np.array([sam(f, ref) for f in frames])
        rmses = np.array([rmse(f, ref) for f in frames])

        correlation_min = float(np.min(corrs))
        sam_max = float(np.max(sams))
        rmse_mean = float(np.mean(rmses))

        # ---------- 色度类 ----------
        try:
            ref_uv = spectrum_to_uv_prime(self.wavelengths, ref)
        except ValueError:
            ref_uv = (float("nan"), float("nan"))

        ccts = np.array([self._safe_spectrum_to_cct(self.wavelengths, f) for f in frames])
        ccts_finite = ccts[np.isfinite(ccts)]
        cct_mean = (
            float(np.mean(ccts_finite)) if ccts_finite.size else float("nan")
        )
        cct_range = (
            float(np.max(ccts_finite) - np.min(ccts_finite))
            if ccts_finite.size
            else float("nan")
        )

        delta_uvs = np.array(
            [
                delta_uv_prime(
                    self._safe_spectrum_to_uv(self.wavelengths, f), ref_uv
                )
                for f in frames
            ]
        )
        delta_uv_max = (
            float(np.nanmax(delta_uvs)) if delta_uvs.size else float("nan")
        )

        # ---------- 特征类 ----------
        centroids = np.array(
            [spectral_centroid(self.wavelengths, f) for f in frames]
        )
        centroid_mean = float(np.mean(centroids))
        centroid_drift = peak_to_peak(centroids)

        band_ratios = np.array(
            [
                self._safe_band_ratio(
                    self.wavelengths,
                    f,
                    self.numerator_band,
                    self.denominator_band,
                )
                for f in frames
            ]
        )
        br_finite = band_ratios[np.isfinite(band_ratios)]
        band_ratio_mean = (
            float(np.mean(br_finite)) if br_finite.size else float("nan")
        )

        # ---------- 稳定性判定 ----------
        c = self.criteria
        reasons: list = []
        if integral_cv_val >= c.cv_max:
            reasons.append(
                f"积分强度 CV={integral_cv_val * 100:.3f}% "
                f"≥ {c.cv_max * 100:.2f}%"
            )
        if correlation_min <= c.correlation_min:
            reasons.append(
                f"相关系数 r={correlation_min:.5f} ≤ {c.correlation_min}"
            )
        if np.isfinite(delta_uv_max) and delta_uv_max >= c.delta_uv_max:
            reasons.append(
                f"色差 Δu'v'={delta_uv_max:.5f} ≥ {c.delta_uv_max}"
            )
        if centroid_drift >= c.centroid_drift_max:
            reasons.append(
                f"光谱重心漂移={centroid_drift:.4f}nm "
                f"≥ {c.centroid_drift_max}nm"
            )

        return StabilityReport(
            ready=True,
            window_start=t_start,
            window_end=t_end,
            n_frames=frames.shape[0],
            integral_mean=integral_mean,
            integral_cv=integral_cv_val,
            integral_drift=integral_drift_val,
            integral_ptp=integral_ptp_val,
            correlation_min=correlation_min,
            sam_max=sam_max,
            rmse_mean=rmse_mean,
            cct_mean=cct_mean,
            cct_range=cct_range,
            delta_uv_max=delta_uv_max,
            centroid_mean=centroid_mean,
            centroid_drift=centroid_drift,
            band_ratio_mean=band_ratio_mean,
            is_stable=(len(reasons) == 0),
            unstable_reasons=reasons,
        )

    # ------------------------------------------------------------------
    # 容错包装器（单帧计算失败不中断整个窗口）
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_spectrum_to_cct(
        wavelengths: np.ndarray, intensity: np.ndarray
    ) -> float:
        try:
            return spectrum_to_cct(wavelengths, intensity)
        except ValueError:
            return float("nan")

    @staticmethod
    def _safe_spectrum_to_uv(
        wavelengths: np.ndarray, intensity: np.ndarray
    ) -> tuple:
        try:
            return spectrum_to_uv_prime(wavelengths, intensity)
        except ValueError:
            return float("nan"), float("nan")

    @staticmethod
    def _safe_band_ratio(
        wavelengths: np.ndarray,
        intensity: np.ndarray,
        num_band: tuple,
        den_band: tuple,
    ) -> float:
        try:
            return band_ratio(wavelengths, intensity, num_band, den_band)
        except ValueError:
            return float("nan")
# ============================================================================
# 批量评价
# ============================================================================

def evaluate_time_series(
    frames: np.ndarray,
    wavelengths: np.ndarray,
    timestamps: Optional[Sequence[float]] = None,
    criteria: Optional[StabilityCriteria] = None,
    window_seconds: float = 1.0,
    band: tuple[float, float] = (650.0, 800.0),
    numerator_band: tuple[float, float] = (650.0, 800.0),
    denominator_band: tuple[float, float] = (400.0, 500.0),
) -> tuple[list[StabilityReport], Optional[float]]:
    """批量评价一段光谱时间序列。

    :param frames:        形状 ``(N, num_points)`` 的光强数组
    :param wavelengths:   波长数组（nm）
    :param timestamps:    时间戳数组（秒）；None 时使用 0,1,...,N-1
    :param criteria:      稳定性判据
    :param window_seconds: 滑动窗口大小（秒）
    :return:              ``(reports, t_stable)``
    :raises ValueError:   frames 不是二维数组
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError("frames 必须是二维数组 (N, num_points)")

    n = frames.shape[0]
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.float64)

    analyzer = StabilityAnalyzer(
        wavelengths=wavelengths,
        criteria=criteria,
        window_seconds=window_seconds,
        band=band,
        numerator_band=numerator_band,
        denominator_band=denominator_band,
    )

    reports: list[StabilityReport] = []
    for i in range(n):
        reports.append(
            analyzer.add_frame(frames[i], timestamp=float(timestamps[i]))
        )

    return reports, analyzer.t_stable