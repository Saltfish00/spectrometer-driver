"""去噪算法单元测试。"""

import numpy as np
import pytest

from spectrometer_driver.denoising import Denoiser


def test_frame_average_reduces_noise():
    """帧平均应当降低随机噪声（至少 5 倍改善）。"""
    rng = np.random.default_rng(42)
    true_signal = np.sin(np.linspace(0, 4 * np.pi, 500)) * 100
    noise = rng.normal(0, 5, size=(200, 500))
    frames = true_signal + noise

    avg = Denoiser.frame_average(frames)

    err_single = np.abs(frames[0] - true_signal).mean()
    err_avg = np.abs(avg - true_signal).mean()
    # √200 ≈ 14 倍，验证至少提升 5 倍
    assert err_avg < err_single / 5


def test_frame_average_invalid_dim():
    """一维输入应抛 ValueError。"""
    with pytest.raises(ValueError):
        Denoiser.frame_average(np.zeros(100))


def test_median_filter_removes_spike():
    """中值滤波应消除孤立尖峰。"""
    data = np.ones(101)
    data[50] = 1000
    filtered = Denoiser.median_filter(data, kernel_size=5)
    assert abs(filtered[50] - 1.0) < 0.5


def test_savgol_preserves_peak():
    """SG 滤波应保留高斯峰高度。"""
    x = np.linspace(-5, 5, 501)
    peak = np.exp(-x ** 2) * 1000
    noisy = peak + np.random.default_rng(0).normal(0, 20, size=peak.size)
    filtered = Denoiser.savgol_filter(noisy, window_length=11, polyorder=3)
    assert abs(filtered.max() - 1000) < 50


def test_wavelet_denoise():
    """小波去噪应降低噪声标准差。"""
    try:
        import pywt  # noqa: F401
    except ImportError:
        pytest.skip("未安装 PyWavelets")

    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 4 * np.pi, 500)) * 100
    noisy = signal + rng.normal(0, 10, size=500)

    denoised = Denoiser.wavelet_denoise(noisy, wavelet="db8", level=3)
    assert denoised.std() < noisy.std()
    assert len(denoised) == len(noisy)


def test_auto_denoise_all_methods():
    """auto_denoise 对所有方法都应正常返回。"""
    rng = np.random.default_rng(0)
    frames = rng.normal(100, 5, size=(20, 200)).astype(np.float64)
    for method in ["frame_average", "median", "savgol"]:
        result = Denoiser.auto_denoise(frames, method=method)
        assert result.shape == (200,)


def test_auto_denoise_invalid_method():
    """未知方法应抛 ValueError。"""
    with pytest.raises(ValueError):
        Denoiser.auto_denoise(np.zeros((10, 100)), method="not_a_method")