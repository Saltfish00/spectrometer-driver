"""光源质量评价模块单元测试。"""

import numpy as np
import pytest

from spectrometer_driver.metrics import (
    evaluate_single_frame,
    StabilityAnalyzer,
    evaluate_time_series,
    integrate_band,
    cv,
    relative_drift,
    peak_to_peak,
    correlation,
    sam,
    spectral_centroid,
    band_ratio,
    cie_xyz_bar,
    spectrum_to_cct,
    delta_uv_prime,
    cct_from_xy,
)


# ============================================================================
# 底层工具
# ============================================================================

def test_integrate_band_rectangle():
    """常数光谱的积分 = 常数 × 带宽。"""
    wl = np.linspace(600, 900, 301)
    intensity = np.full_like(wl, 2.0)
    val = integrate_band(wl, intensity, 650, 800)
    assert abs(val - 2.0 * 150) < 1e-6


def test_cv_known_values():
    """验证 CV 的已知值。"""
    assert abs(cv(np.array([10, 10, 10, 10])) - 0.0) < 1e-12
    # [1..5]: mean=3, std=1.4142, CV=0.4714
    assert abs(cv(np.array([1, 2, 3, 4, 5])) - 0.4714) < 1e-3


def test_relative_drift():
    assert abs(relative_drift(np.array([100, 100, 100]))) < 1e-12
    assert abs(relative_drift(np.array([100, 105, 110])) - 0.10) < 1e-12


def test_peak_to_peak():
    assert peak_to_peak(np.array([1, 5, 3, 8, 2])) == 7


def test_correlation_perfect():
    a = np.array([1.0, 2, 3, 4, 5])
    assert abs(correlation(a, a) - 1.0) < 1e-12
    assert abs(correlation(a, -a) + 1.0) < 1e-12


def test_sam():
    a = np.array([1.0, 2, 3, 4, 5])
    assert abs(sam(a, a)) < 1e-12
    # 同方向不同幅度
    assert abs(sam(a, 10 * a)) < 1e-12
    # 正交向量 SAM = π/2
    b = np.array([0.0, 0, 0, 0, 1])
    a_orth = np.array([1.0, 0, 0, 0, 0])
    assert abs(sam(a_orth, b) - np.pi / 2) < 1e-12


def test_spectral_centroid_symmetric():
    """对称高斯峰的重心 = 中心波长。"""
    wl = np.linspace(400, 800, 401)
    intensity = np.exp(-0.5 * ((wl - 600) / 20) ** 2)
    centroid = spectral_centroid(wl, intensity)
    assert abs(centroid - 600.0) < 0.5


def test_band_ratio():
    """常数光谱的波段比 = 带宽比。"""
    wl = np.linspace(400, 900, 501)
    intensity = np.ones_like(wl)
    ratio = band_ratio(wl, intensity, (650, 800), (400, 500))
    assert abs(ratio - 1.5) < 1e-6


# ============================================================================
# 色度
# ============================================================================

def test_cie_xyz_bar_smooth():
    """颜色匹配函数应平滑且非负。"""
    wl = np.linspace(380, 780, 401)
    xb, yb, zb = cie_xyz_bar(wl)
    assert np.all(xb >= 0)
    assert np.all(yb >= 0)
    assert np.all(zb >= 0)
    # ȳ 的峰值在 555 nm 附近
    assert abs(wl[np.argmax(yb)] - 555) < 20


def test_cct_from_xy_d65():
    """D65 白点的色温应在 6500 K 左右。"""
    cct = cct_from_xy(0.3127, 0.3290)
    assert 6300 < cct < 6700


def test_spectrum_to_cct_blackbody_like():
    """模拟 3000K 黑体光谱，验证 CCT 在合理范围。"""
    wl = np.linspace(380, 1000, 621)
    h, c, k = 6.626e-34, 3e8, 1.381e-23
    T = 3000
    lam_m = wl * 1e-9
    intensity = (2 * h * c**2) / lam_m**5 / (np.exp(h * c / (lam_m * k * T)) - 1)
    cct = spectrum_to_cct(wl, intensity)
    assert 2700 < cct < 3300


def test_delta_uv_same_point():
    """同一点的色差应为 0。"""
    assert delta_uv_prime((0.2, 0.4), (0.2, 0.4)) == 0.0
    assert abs(delta_uv_prime((0.2, 0.4), (0.201, 0.4)) - 0.001) < 1e-9


# ============================================================================
# 单帧评估
# ============================================================================

def test_evaluate_single_frame_basic():
    wl = np.linspace(400, 900, 501)
    intensity = np.linspace(100, 200, 501)
    dark = np.zeros_like(wl)

    m = evaluate_single_frame(intensity, wl, dark)
    assert m.dark_noise == 0.0
    assert m.saturation_count == 0
    assert m.stray_light_ratio == 0.0
    # 650-800 均值在 150 附近
    assert 140 < m.band_signal_mean < 170


def test_evaluate_single_frame_saturation():
    """接近 ADC 满量程应判定饱和。"""
    wl = np.linspace(400, 900, 101)
    intensity = np.full(101, 65000.0)
    m = evaluate_single_frame(intensity, wl, adc_max=65535)
    assert m.saturation_count > 0
    assert m.saturation_ratio > 0


def test_evaluate_single_frame_no_dark():
    """无暗帧时应使用差分法估计噪声。"""
    wl = np.linspace(400, 900, 501)
    intensity = 1000 + np.random.default_rng(0).normal(0, 5, 501)
    m = evaluate_single_frame(intensity, wl)
    assert m.dark_noise is None
    assert m.noise_estimate > 0
    assert m.stray_light_ratio is None


# ============================================================================
# 时间序列稳定性
# ============================================================================

def test_stability_analyzer_stable_signal():
    """稳定的信号应在短时间内判定稳定。"""
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 900, 301)
    base = np.exp(-0.5 * ((wl - 650) / 80) ** 2) * 1000

    analyzer = StabilityAnalyzer(
        wavelengths=wl,
        window_seconds=0.5,
        min_frames=5,
    )

    for i in range(100):
        noise = rng.normal(0, 0.5, wl.size)
        analyzer.add_frame(base + noise, timestamp=i * 0.1)
        if analyzer.t_stable is not None:
            break

    assert analyzer.t_stable is not None
    assert analyzer.t_stable < 2.0


def test_stability_analyzer_unstable_signal():
    """剧烈波动的信号不应被判定为稳定。"""
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 900, 301)
    base = np.exp(-0.5 * ((wl - 650) / 80) ** 2) * 1000

    analyzer = StabilityAnalyzer(
        wavelengths=wl,
        window_seconds=0.5,
        min_frames=5,
    )

    # ±10% 强度波动，远超 CV=1% 阈值
    for i in range(100):
        scale = 1.0 + rng.normal(0, 0.1)
        analyzer.add_frame(base * scale, timestamp=i * 0.1)

    assert analyzer.t_stable is None


def test_stability_analyzer_window_not_ready():
    """窗口未填满时 ready=False。"""
    wl = np.linspace(400, 900, 101)
    analyzer = StabilityAnalyzer(wavelengths=wl, window_seconds=1.0, min_frames=5)
    report = analyzer.add_frame(np.ones_like(wl), timestamp=0.0)
    assert not report.ready
    assert report.integral_cv is None


def test_evaluate_time_series_stable():
    """批量评价稳定的时间序列。"""
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 900, 301)
    base = np.exp(-0.5 * ((wl - 650) / 80) ** 2) * 1000

    frames = base + rng.normal(0, 0.3, size=(80, wl.size))
    timestamps = np.arange(80) * 0.1

    reports, t_stable = evaluate_time_series(
        frames, wl, timestamps, window_seconds=0.5
    )
    assert len(reports) == 80
    assert t_stable is not None


def test_evaluate_time_series_invalid_input():
    """一维输入应抛 ValueError。"""
    with pytest.raises(ValueError):
        evaluate_time_series(np.zeros(100), np.linspace(400, 900, 100))