"""TLM 系列串口光谱仪 Python 驱动包。

============================================================
四层架构总览
============================================================

    ┌──────────────────────────────────────────┐
    │  设备层 (spectrometer)                    │
    │  - 面向用户的高级 API                      │
    │  - 属性封装、参数校验、缓存管理             │
    └──────────────────┬───────────────────────┘
                       │ 调用
    ┌──────────────────▼───────────────────────┐
    │  传输层 (transport)                       │
    │  - 串口字节流读写                          │
    │  - 帧头同步、超时控制、线程安全             │
    └──────────────────┬───────────────────────┘
                       │ 调用
    ┌──────────────────▼───────────────────────┐
    │  协议层 (protocol)                        │
    │  - 帧的构建 / 解析                         │
    │  - 校验和计算、结构验证                     │
    └──────────────────────────────────────────┘

    ┌──────────────────────────────────────────┐
    │  功能层（独立模块，可单独使用）             │
    │  - denoising  多帧光谱去噪                 │
    │  - metrics    单帧质量 + 时序稳定性         │
    └──────────────────────────────────────────┘

============================================================
典型用法
============================================================

单帧采集::

    from spectrometer_driver import Spectrometer

    with Spectrometer("/dev/ttyUSB0") as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 100_000
        data = spec.get_spectrum()
        print(data.num_points, data.is_normal)

多帧平均::

    with Spectrometer("COM3") as spec:
        data = spec.acquire_averaged(n_frames=100)

稳定性分析::

    from spectrometer_driver import StabilityAnalyzer

    analyzer = StabilityAnalyzer(wavelengths, window_seconds=1.0)
    for frame in spec.iter_continuous():
        report = analyzer.add_frame(frame.intensity)
        if report.is_stable:
            print("稳定时间:", analyzer.t_stable)
            break
"""

# ---- 核心设备层 ----
from .spectrometer import Spectrometer, SpectrumData

# ---- 去噪层 ----
from .denoising import Denoiser

# ---- 质量评价层 ----
# 把常用工具一并暴露给用户，方便 "from spectrometer_driver import xxx"
from .metrics import (
    # 单帧信号质量
    SingleFrameMetrics,
    evaluate_single_frame,
    # 时间序列稳定性
    StabilityAnalyzer,
    StabilityCriteria,
    StabilityReport,
    evaluate_time_series,
    # 底层指标工具（用户可以单独调用，不依赖设备）
    integrate_band,        # 波段积分
    cv,                    # 变异系数
    relative_drift,        # 相对漂移
    peak_to_peak,          # 峰峰值
    correlation,           # 相关系数
    sam,                   # 光谱角
    rmse,                  # 均方根误差
    spectral_centroid,     # 光谱重心
    band_ratio,            # 波段比
    # 色度计算
    spectrum_to_xyz,       # 光谱 → XYZ
    spectrum_to_cct,       # 光谱 → 色温
    spectrum_to_uv_prime,  # 光谱 → u'v'
    delta_uv_prime,        # 色差
    cie_xyz_bar,           # CIE 颜色匹配函数
)

# ---- 协议层（暴露给高级用户自定义命令）----
from .protocol import Command, build_command_frame, parse_response_frame

# ---- 异常 ----
from .exceptions import (
    SpectrometerError,     # 基类
    ConnectionError,       # 串口连接
    ProtocolError,         # 协议格式
    ChecksumError,         # 校验和
    TimeoutError,          # 超时
    DeviceError,           # 设备返回失败
)

__version__ = "0.2.1"

# 显式声明公开接口，影响 `from pkg import *` 的行为
__all__ = [
    # 核心类
    "Spectrometer",
    "SpectrumData",
    "Denoiser",
    "Command",
    # 质量评价
    "SingleFrameMetrics",
    "evaluate_single_frame",
    "StabilityAnalyzer",
    "StabilityCriteria",
    "StabilityReport",
    "evaluate_time_series",
    # 底层指标
    "integrate_band",
    "cv",
    "relative_drift",
    "peak_to_peak",
    "correlation",
    "sam",
    "rmse",
    "spectral_centroid",
    "band_ratio",
    "spectrum_to_xyz",
    "spectrum_to_cct",
    "spectrum_to_uv_prime",
    "delta_uv_prime",
    "cie_xyz_bar",
    # 协议
    "build_command_frame",
    "parse_response_frame",
    # 异常
    "SpectrometerError",
    "ConnectionError",
    "ProtocolError",
    "ChecksumError",
    "TimeoutError",
    "DeviceError",
]