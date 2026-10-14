# Spectrometer Driver

> TLM 系列串口光谱仪的 Python 驱动包
> 支持单帧采集、连续采集、多帧去噪、光源质量评价，采用分层架构，易于扩展与集成。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)]()

---

## 📋 目录

- [特性](#-特性)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [API 参考](#-api-参考)
  - [Spectrometer 类](#spectrometer-类)
  - [SpectrumData 数据类](#spectrumdata-数据类)
  - [Denoiser 去噪器](#denoiser-去噪器)
  - [光源质量评价](#光源质量评价-api)
- [协议说明](#-协议说明)
- [架构设计](#-架构设计)
- [去噪方法详解](#-去噪方法详解)
- [光源质量评价详解](#-光源质量评价详解)
- [示例脚本](#-示例脚本)
- [测试](#-测试)
- [常见问题](#-常见问题)
- [开发指南](#-开发指南)
- [项目结构](#-项目结构)
- [许可证](#-许可证)

---

## ✨ 特性

- **分层架构**：传输层 / 协议层 / 设备层三级解耦，维护与扩展都简单
- **完整协议支持**：实现了手册中全部 9 条命令
  （单帧采集、连续采集、停止采集、设备信息、手动/自动曝光、
  设置/读取曝光值、波长范围查询）
- **多帧去噪**：内置 5 种主流光谱去噪方法
  （帧平均、中值滤波、Savitzky-Golay、小波、PCA）
- **光源质量评价**：
  - 单帧信号质量：暗噪声、SNR、饱和、杂散光
  - 时间序列稳定性：CV、相关系数、SAM、RMSE、CCT、Δu'v'、
    光谱重心、波段比、稳定时间 t_stable
- **实时滑动窗口**：支持在线监测光源稳定性
- **线程安全**：所有串口操作由 `RLock` 保护
- **健壮通信**：自动帧头同步、校验和验证、超时控制、脏数据清理
- **易用 API**：上下文管理器、属性式读写曝光参数、迭代器式连续采集
- **详尽注释**：每个类、方法、关键算法均有详细中文注释
- **测试完备**：协议层、去噪算法、质量评价均有单元测试

---

## 🚀 安装

### 从源码安装

```bash
git clone https://github.com/yourname/spectrometer-driver.git
cd spectrometer-driver
pip install -e .
```

### 安装可选依赖

```bash
# PCA 去噪
pip install -e ".[pca]"

# 绘图功能
pip install -e ".[plot]"

# 全部功能
pip install -e ".[pca,plot,dev]"
```

### 依赖项

| 包 | 版本 | 用途 | 是否必装 |
|---|---|---|---|
| pyserial | >=3.5 | 串口通信 | ✅ |
| numpy | >=1.20 | 数值计算 | ✅ |
| scipy | >=1.7 | 中值 / SG 滤波 | ✅ |
| PyWavelets | >=1.2 | 小波去噪 | ✅ |
| scikit-learn | >=1.0 | PCA 去噪 | 可选 |
| matplotlib | >=3.4 | 绘图 | 可选 |

---

## 🎯 快速开始

### 单帧采集

```python
from spectrometer_driver import Spectrometer

with Spectrometer("/dev/ttyUSB0") as spec:   # Windows: "COM3"
    print("设备信息:", spec.device_info)
    print("波长范围:", spec.wavelength_range)

    # 手动曝光 + 100ms
    spec.exposure_mode = "manual"
    spec.exposure_time = 100_000  # 单位: 微秒

    # 采集一帧
    data = spec.get_spectrum()
    print(f"点数: {data.num_points}, 曝光正常: {data.is_normal}")

    # 数据字段：
    # data.raw_data     -> numpy uint16 数组，原始光谱数据
    # data.intensity    -> numpy float64 数组，系数修正后的光强
    # data.wavelengths  -> numpy float64 数组，对应波长 (nm)
```

### 多帧平均去噪

```python
from spectrometer_driver import Spectrometer

with Spectrometer("/dev/ttyUSB0") as spec:
    spec.exposure_mode = "manual"
    spec.exposure_time = 50_000

    # 采集 100 帧并做帧平均
    data = spec.acquire_averaged(n_frames=100)

    # data.raw_data        -> 平均后的光谱 (num_points,)
    # data.frames          -> 原始 100 帧 (100, num_points)
    # data.snr_gain        -> 理论信噪比增益 √100 = 10.0
```

### 连续采集（迭代器）

```python
with Spectrometer("/dev/ttyUSB0") as spec:
    spec.exposure_mode = "manual"
    spec.exposure_time = 50_000

    # 采集 200 帧后自动停止
    for i, frame in enumerate(spec.iter_continuous(count=200)):
        if i % 20 == 0:
            print(f"第 {i} 帧, 平均光强={frame.intensity.mean():.2f}")
```

### 光源质量评价

```python
import time
from spectrometer_driver import (
    Spectrometer, StabilityAnalyzer, StabilityCriteria
)

with Spectrometer("/dev/ttyUSB0") as spec:
    spec.exposure_mode = "manual"
    spec.exposure_time = 30_000

    probe = spec.get_spectrum()
    analyzer = StabilityAnalyzer(
        wavelengths=probe.wavelengths,
        criteria=StabilityCriteria(
            cv_max=0.01,                # CV < 1%
            correlation_min=0.999,      # r > 0.999
            delta_uv_max=0.002,         # Δu'v' < 0.002
            centroid_drift_max=0.2,     # 重心漂移 < 0.2 nm
        ),
        window_seconds=1.0,
    )

    t0 = time.time()
    for frame in spec.iter_continuous():
        report = analyzer.add_frame(frame.intensity, timestamp=time.time() - t0)
        if report.ready:
            print(f"CV={report.integral_cv*100:.3f}%, "
                  f"r={report.correlation_min:.5f}, "
                  f"CCT={report.cct_mean:.1f}K, "
                  f"稳定={report.is_stable}")
        if analyzer.t_stable is not None:
            print(f"光源稳定时间: {analyzer.t_stable:.3f} 秒")
            break
```

---

## 📖 API 参考

### Spectrometer 类

主驱动类，代表一台光谱仪设备。

#### 构造函数

```python
Spectrometer(port: str, baudrate: int = 921600, timeout: float = 2.0)
```

| 参数 | 说明 |
|---|---|
| `port` | 串口名称，如 `"/dev/ttyUSB0"` 或 `"COM3"` |
| `baudrate` | 波特率，默认 921600（手册规定值） |
| `timeout` | 单次读写超时（秒） |

#### 生命周期方法

| 方法 | 说明 |
|---|---|
| `open()` | 打开串口并读取设备信息、波长范围 |
| `close()` | 关闭串口（自动停止连续采集） |
| `__enter__/__exit__` | 支持 `with` 语句 |

#### 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `is_open` | bool | 串口是否打开 |
| `device_info` | str | 设备信息字符串 |
| `wavelength_range` | (float, float) | (起始波长, 终止波长) nm |
| `num_points` | int | 光谱像素个数（首帧采集后确定） |
| `wavelengths` | np.ndarray | 均匀分布的波长数组 (nm) |
| `is_continuous` | bool | 是否处于连续采集模式 |
| `exposure_mode` | str | 曝光模式（`"manual"` / `"auto"`，**只写**） |
| `exposure_time` | int | 曝光时间（μs），可读可写 |

#### 采集方法

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_spectrum(timeout=None)` | `SpectrumData` | 单帧采集 |
| `start_continuous()` | None | 启动连续采集 |
| `read_continuous(timeout=None)` | `SpectrumData` | 读取连续模式下一帧 |
| `stop_continuous()` | None | 停止连续采集 |
| `iter_continuous(count=None, timeout=None)` | Iterator | 连续采集迭代器 |
| `acquire_averaged(n_frames, denoise_method="frame_average", **kwargs)` | `SpectrumData` | 采集 N 帧并去噪 |

#### 曝光控制方法

| 方法 | 说明 |
|---|---|
| `set_manual_exposure_mode()` | 切换为手动曝光 |
| `set_auto_exposure_mode()` | 切换为自动曝光 |
| `set_exposure_time(us)` | 设置曝光时间（微秒） |
| `get_exposure_time()` | 读取当前曝光时间（微秒） |

---

### SpectrumData 数据类

一次采集结果的容器（dataclass）。

#### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `exposure_status` | int | 曝光状态：0=正常, 1=过曝, 2=欠曝 |
| `exposure_time_us` | int | 当前曝光时间（μs） |
| `spectrum_coefficient` | int | 光谱系数 N（实际值 = 原始值 / 10^N） |
| `raw_data` | np.ndarray | uint16 原始光谱数组 |
| `wavelengths` | np.ndarray \| None | 对应波长数组 (nm) |
| `frames` | np.ndarray \| None | 多帧采集时保留的所有原始帧 (N, num_points) |

#### 便捷属性

| 属性 | 说明 |
|---|---|
| `intensity` | 系数修正后的光强（float64） |
| `is_normal` / `is_overexposed` / `is_underexposed` | 曝光状态判断 |
| `num_points` | 点数 |
| `snr_gain` | 多帧采集时的信噪比增益（√N） |

---

### Denoiser 去噪器

静态方法集合，用于多帧或单帧光谱去噪。

```python
from spectrometer_driver import Denoiser

# 帧平均
avg = Denoiser.frame_average(frames)          # frames 形状 (N, num_points)

# 中值滤波（脉冲噪声去除）
clean = Denoiser.median_filter(data, kernel_size=5)

# Savitzky-Golay 平滑（保留峰形）
smooth = Denoiser.savgol_filter(data, window_length=11, polyorder=3)

# 小波去噪
denoised = Denoiser.wavelet_denoise(data, wavelet='db8', level=3)

# PCA 去噪（需要 scikit-learn）
denoised = Denoiser.pca_denoise(frames, n_components=5)

# 自动化组合去噪（推荐入口）
result = Denoiser.auto_denoise(frames, method="frame_average")
```

| 方法 | 说明 |
|---|---|
| `frame_average(frames)` | 帧平均，SNR 提升 √N 倍 |
| `median_filter(data, kernel_size=5)` | 中值滤波，去除脉冲噪声 |
| `savgol_filter(data, window_length=11, polyorder=3)` | SG 平滑，保峰形 |
| `wavelet_denoise(data, wavelet='db8', level=3, threshold_mode='soft')` | 小波阈值去噪 |
| `pca_denoise(frames, n_components=5)` | PCA 降维去噪 |
| `auto_denoise(frames, method='frame_average', **kwargs)` | 一站式去噪入口 |

---

### 光源质量评价 API

#### 单帧信号质量

```python
from spectrometer_driver import evaluate_single_frame

metrics = evaluate_single_frame(
    intensity=signal.intensity,
    wavelengths=signal.wavelengths,
    dark=dark.intensity,          # 可选：暗帧
    band=(650.0, 800.0),          # SNR 波段
    adc_max=65535,                # ADC 满量程
    saturation_threshold=0.95,    # 饱和判定阈值
)
```

返回 `SingleFrameMetrics`：

| 字段 | 说明 |
|---|---|
| `dark_noise` | 暗噪声标准差（无暗帧时为 None） |
| `noise_estimate` | 噪声估计（差分法或暗帧给出） |
| `snr` | 指定波段信噪比 |
| `band_signal_mean` | 指定波段信号均值 |
| `saturation_ratio` | 饱和像素比例 |
| `saturation_count` | 饱和像素个数 |
| `stray_light_ratio` | 杂散光/环境光比例 |
| `band` | 用于 SNR 计算的波段 |

#### 时间序列稳定性

```python
from spectrometer_driver import StabilityAnalyzer, StabilityCriteria

analyzer = StabilityAnalyzer(
    wavelengths=probe.wavelengths,
    criteria=StabilityCriteria(
        cv_max=0.01,
        correlation_min=0.999,
        delta_uv_max=0.002,
        centroid_drift_max=0.2,
    ),
    window_seconds=1.0,
    band=(650.0, 800.0),
    numerator_band=(650.0, 800.0),
    denominator_band=(400.0, 500.0),
)

for frame in stream:
    report = analyzer.add_frame(frame.intensity, timestamp=t)
    if report.ready:
        print(report)

print("稳定时间:", analyzer.t_stable)
```

返回 `StabilityReport`：

| 字段 | 类别 | 说明 |
|---|---|---|
| `ready` | 元信息 | 窗口是否已填满 |
| `window_start` / `window_end` | 元信息 | 窗口起止时间 |
| `n_frames` | 元信息 | 窗口内的帧数 |
| `integral_mean` | 强度 | 积分强度均值 |
| `integral_cv` | 强度 | 积分强度变异系数 |
| `integral_drift` | 强度 | 相对漂移 |
| `integral_ptp` | 强度 | 峰峰值 |
| `correlation_min` | 形状 | 与参考光谱的最小相关系数 |
| `sam_max` | 形状 | 与参考光谱的最大光谱角 |
| `rmse_mean` | 形状 | 与参考光谱的平均 RMSE |
| `cct_mean` | 色度 | 平均色温 |
| `cct_range` | 色度 | 色温波动范围 |
| `delta_uv_max` | 色度 | 最大色差 Δu'v' |
| `centroid_mean` | 特征 | 平均光谱重心 |
| `centroid_drift` | 特征 | 光谱重心漂移 |
| `band_ratio_mean` | 特征 | 波段比均值 |
| `is_stable` | 判定 | 本窗口是否稳定 |
| `unstable_reasons` | 判定 | 不稳定原因列表 |

#### 批量离线分析

```python
from spectrometer_driver import evaluate_time_series

# frames: (N, num_points) 光强数组
# wavelengths: 波长数组
# timestamps: 时间戳（秒）
reports, t_stable = evaluate_time_series(
    frames, wavelengths, timestamps,
    window_seconds=1.0,
)

for r in reports:
    if r.ready:
        print(f"稳定={r.is_stable}, CV={r.integral_cv*100:.3f}%")
print("首次稳定时间:", t_stable)
```

#### 底层指标工具（可独立调用）

| 函数 | 说明 |
|---|---|
| `integrate_band(wl, I, wl_min, wl_max)` | 指定波段梯形积分 |
| `cv(series)` | 变异系数 |
| `relative_drift(series, ref=None)` | 相对漂移 |
| `peak_to_peak(series)` | 峰峰值 |
| `correlation(a, b)` | 皮尔逊相关系数 |
| `sam(a, b)` | 光谱角（弧度） |
| `rmse(a, b)` | 均方根误差 |
| `spectral_centroid(wl, I)` | 光谱重心 λc |
| `band_ratio(wl, I, num_band, den_band)` | 波段比 |
| `spectrum_to_xyz(wl, I)` | 光谱 → CIE XYZ |
| `spectrum_to_cct(wl, I)` | 光谱 → 色温 CCT |
| `spectrum_to_uv_prime(wl, I)` | 光谱 → u'v' 色度坐标 |
| `delta_uv_prime(uv1, uv2)` | CIE 1976 u'v' 色差 |
| `cie_xyz_bar(wl)` | CIE 1931 颜色匹配函数 |

---

## 📡 协议说明

以下内容摘自设备手册 V1.2.1，供开发者查阅。

### 串口设置

- 波特率：**921600**
- 校验位：NONE
- 数据位：8
- 停止位：1
- 流控：NONE

### 帧格式

**命名包（主机 → 设备）**

| 字节 | 内容 |
|---|---|
| 0-1 | 包头 `0xCC 0x01` |
| 2-4 | 总包长（3 字节，小端） |
| 5 | 命令类型 |
| 6 ~ N-3 | 命名数据 |
| N-2 | 校验和（之前所有字节之和 & 0xFF） |
| N-1 ~ N | 包尾 `0x0D 0x0A` |

**返回包（设备 → 主机）**

结构相同，仅包头为 `0xCC 0x81`。

### 命令一览

| 命令 | 类型码 | 数据 | 说明 |
|---|---|---|---|
| 单帧获取光谱 | 0x02 | 空 | 返回一帧光谱 |
| 开始连续获取 | 0x03 | 空 | 启动连续推送 |
| 停止获取 | 0x04 | 空 | 停止连续推送 |
| 获取设备信息 | 0x08 | 0x18 | 返回 24 字节设备信息 |
| 设置手动曝光模式 | 0x0A | 空 | 返回 0x00 成功 / 0x15 失败 |
| 设置自动曝光模式 | 0x0B | 空 | 同上 |
| 设置曝光值 | 0x0C | uint32 (μs, LE) | 同上 |
| 获取曝光值 | 0x0D | 空 | 返回 uint32 (μs) |
| 获取波长起止 | 0x0F | 空 | 返回 2 个 uint16 (nm) |

### 光谱数据段解析

单帧和连续帧的响应数据段结构：

```
曝光状态(1) + 曝光时间(4, LE) + 光谱系数(2, LE) + 光谱数据(2*N, LE)
```

- **曝光状态**：0x00 正常 / 0x01 过曝 / 0x02 欠曝
- **曝光时间**：单位 μs，小端 uint32
- **光谱系数 N**：实际光强 = 原始值 / 10^N
- **光谱数据**：每像素 2 字节，小端 uint16，长度 = 像素数 × 2

---

## 🏗 架构设计

```
┌─────────────────────────────────────────────┐
│  Spectrometer (设备层)                       │
│  - 面向用户的高级 API                         │
│  - 属性封装、参数校验、状态管理               │
│  - acquire_averaged / iter_continuous        │
└──────────────────┬──────────────────────────┘
                   │ 使用
┌──────────────────▼──────────────────────────┐
│  SerialTransport (传输层)                    │
│  - 串口读写、帧头同步、超时控制               │
│  - 线程安全 (RLock)                          │
└──────────────────┬──────────────────────────┘
                   │ 使用
┌──────────────────▼──────────────────────────┐
│  protocol (协议层)                           │
│  - build_command_frame / parse_response_frame│
│  - 校验和计算、帧结构验证                     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Denoiser (去噪层，独立模块)                  │
│  - frame_average / median / savgol / wavelet │
│  - pca / auto_denoise                        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  metrics (质量评价层，独立模块)               │
│  - evaluate_single_frame                     │
│  - StabilityAnalyzer / evaluate_time_series  │
│  - 底层指标与色度计算                        │
└─────────────────────────────────────────────┘
```

**设计原则**

1. **分层解耦**：协议层不关心串口，传输层不关心命令语义。
2. **异常细分**：`ConnectionError` / `ProtocolError` / `ChecksumError` /
   `TimeoutError` / `DeviceError` 各自独立，方便上层精准捕获。
3. **线程安全**：`RLock` 保护所有串口访问，多线程场景下可直接共享实例。
4. **上下文管理**：`with` 语句确保串口一定被释放。
5. **数据类只读**：`SpectrumData` 使用 `dataclass`，语义清晰。

---

## 🔬 去噪方法详解

| 方法 | 原理 | 适用场景 | 信噪比提升 |
|---|---|---|---|
| **帧平均** | N 帧逐像素求均值 | 静态光谱、随机噪声 | **√N 倍** |
| **中值滤波** | 邻域中值替代当前像素 | 脉冲噪声（宇宙射线） | 去除尖峰 |
| **Savitzky-Golay** | 局部多项式最小二乘拟合 | 保峰形平滑 | 中等 |
| **小波去噪** | 多尺度分解 + 阈值化 | 多尺度噪声 | 优 |
| **PCA** | 降维保留主成分 | 帧数 ≫ 波长点数 | 显著（可达 5×） |

### 推荐去噪流程

```
原始 N 帧
   │
   ├─► 1. 逐帧中值滤波 (去脉冲噪声)
   │
   ├─► 2. 帧平均 (去随机噪声)
   │
   └─► 3. Savitzky-Golay 或 小波平滑 (进一步降噪)
```

### 使用示例

```python
import numpy as np
from spectrometer_driver import Spectrometer, Denoiser

with Spectrometer("/dev/ttyUSB0") as spec:
    spec.exposure_mode = "manual"
    spec.exposure_time = 50_000

    # 采集 100 帧
    data = spec.acquire_averaged(n_frames=100)
    frames = data.frames  # (100, num_points)

    # 步骤 1: 逐帧中值滤波去除尖峰
    frames_clean = np.stack([
        Denoiser.median_filter(f, kernel_size=5) for f in frames
    ])

    # 步骤 2: 帧平均
    avg = Denoiser.frame_average(frames_clean)

    # 步骤 3: Savitzky-Golay 平滑
    final = Denoiser.savgol_filter(avg, window_length=11, polyorder=3)

    print(f"最终信噪比增益: √100 = {np.sqrt(100):.1f} 倍")
```

### 参数选择建议

| 方法 | 参数 | 推荐值 | 说明 |
|---|---|---|---|
| 帧平均 | `n_frames` | 20 ~ 100 | 采集时间与 SNR 提升的折中 |
| 中值滤波 | `kernel_size` | 3 ~ 7 | 窗口越大去除越强，但会平滑窄峰 |
| SG | `window_length` | 峰宽的 0.5~1 倍 | 必须为奇数 |
| SG | `polyorder` | 2 ~ 4 | 过高会引入振铃 |
| 小波 | `wavelet` | `'db8'` / `'sym6'` | db8 通用性好 |
| 小波 | `level` | 3 ~ 5 | 视信号长度而定 |
| PCA | `n_components` | 3 ~ 10 | 视帧数与信噪比而定 |

---

## 🔬 光源质量评价详解

### 单帧信号质量

| 指标 | 含义 | 计算方式 |
|---|---|---|
| **暗噪声水平** | 遮光时基线的标准差 | `std(dark)` |
| **噪声估计** | 无暗帧时用差分法估计 | `std(diff(I)) / √2` |
| **信噪比 SNR** | 指定波段信号均值 / 噪声 | `mean(I_band) / noise` |
| **饱和像素比例** | 接近 ADC 上限的像素比例 | `count(I ≥ 0.95·ADC) / N` |
| **杂散光/环境光比例** | 遮光时信号 / 光照时信号 | `mean(dark) / mean(signal)` |

### 时间序列稳定性

| 类别 | 指标 | 计算/含义 | 稳定判据参考 |
|---|---|---|---|
| **强度** | 积分强度 I(t) | ∫₆₅₀⁸⁰⁰ S(λ,t)dλ | 越平稳越好 |
| **强度** | 变异系数 CV | σ_I / μ_I | < 1% 稳定，1–3% 轻度，>3% 不稳定 |
| **强度** | 相对漂移 | (I(t) − I_ref) / I_ref | 越小越好 |
| **强度** | 峰峰值 P-P | max I − min I | 反映极端波动 |
| **形状** | 相关系数 r(t) | corr(S_t, R_ref) | > 0.999 稳定 |
| **形状** | 光谱角 SAM | arccos(⟨S,R⟩ / ‖S‖‖R‖) | 对强度不敏感，只看形状 |
| **形状** | RMSE | 与参考光谱整体偏差 | 越小越好 |
| **色度** | 色温 CCT | 由光谱计算（McCamy 近似） | 变化 < 50 K 较稳 |
| **色度** | 色差 Δu'v' | 人眼可感知颜色差异 | < 0.002 稳定 |
| **特征** | 光谱重心 λc | Σ(λ·S) / Σ(S) | 漂移 < 0.2 nm 较稳 |
| **特征** | 波段比 | I₆₅₀₋₈₀₀ / I₄₀₀₋₅₀₀ | 反映色温趋势 |
| **时间** | 稳定时间 t_stable | 滑动窗口内指标连续满足阈值的最早时刻 | 关键输出 |
| **频域** | FFT / Allan 方差 | 分析波动频率和长期稳定性 | 可选 |

### 稳定判定方法

滑动窗口内**同时**满足以下条件即判定稳定：

1. 650–800 nm 积分强度 CV < 1%
2. 相关系数 r > 0.999
3. 色差 Δu'v' < 0.002
4. 光谱重心漂移 < 0.2 nm

窗口大小默认为 1.0 秒，可通过 `window_seconds` 调整。
**连续满足的最早时间**即为 `t_stable`。

### 色度计算说明

1. **CIE 1931 颜色匹配函数**：使用 Wyman et al. (2013) 的多峰高斯近似，
   在 380–780 nm 范围内平均误差 < 1%，无需嵌入庞大的数据表。
2. **XYZ 三刺激值**：X = ∫S(λ)x̄(λ)dλ，Y、Z 同理。
3. **u'v' 色度坐标**：CIE 1976 UCS 空间
   u' = 4X/(X+15Y+3Z)，v' = 9Y/(X+15Y+3Z)。
4. **CCT**：使用 McCamy (1992) 近似公式，适用于 2856K ~ 6500K
   （覆盖卤素灯、白光 LED）。

### 关于卤素灯的特殊说明

卤素灯为**连续热辐射光谱**，通常没有锐线特征峰。因此：

- ⚠️ **不建议**把某个具体峰位（如 819 nm）的漂移作为主要指标
- ✅ **推荐**使用：积分强度、相关系数、色差 Δu'v'、光谱重心 λc

本模块已按此推荐设计默认参数。

---

## 📚 示例脚本

`examples/` 目录下提供的示例：

| 文件 | 说明 |
|---|---|
| `01_basic_usage.py` | 单帧采集与设备信息读取 |
| `02_continuous_acquisition.py` | 连续采集 200 帧并统计 |
| `03_denoising_demo.py` | 对比 5 种去噪方法的效果 |
| `04_live_plot.py` | 实时绘图显示光谱（需 matplotlib） |
| `05_light_source_evaluation.py` | 单帧信号质量评价 |
| `06_stability_analysis.py` | 光源稳定性分析（滑动窗口 + t_stable） |

运行方式：

```bash
python examples/01_basic_usage.py
```

---

## 🧪 测试

```bash
# 运行全部测试
pytest

# 生成覆盖率报告
pytest --cov=spectrometer_driver --cov-report=html

# 仅运行协议层测试
pytest tests/test_protocol.py -v

# 仅运行去噪算法测试
pytest tests/test_denoising.py -v

# 仅运行质量评价测试
pytest tests/test_metrics.py -v
```

测试文件：

| 文件 | 覆盖范围 |
|---|---|
| `test_protocol.py` | 帧构建、解析、校验和 |
| `test_denoising.py` | 5 种去噪算法 |
| `test_metrics.py` | 单帧评价、稳定性分析、色度计算 |

---

## ❓ 常见问题

### Q1: 串口打开失败？

- 检查设备是否被其他程序占用（如原厂软件）
- Linux 下确认当前用户在 `dialout` 组：
  `sudo usermod -aG dialout $USER`（需重新登录）
- Windows 下检查设备管理器中 COM 口号

### Q2: 收到 `ChecksumError`？

- 波特率不匹配（必须 921600）
- 串口线质量差或过长
- 检查是否有其他数据混入（可先 flush 输入缓冲区）

### Q3: 连续采集时卡住？

- 增加 `timeout` 参数
- 检查是否需要先设置手动曝光模式
- 使用 `stop_continuous()` 前清空缓冲区

### Q4: 采集速度慢？

- 减小曝光时间
- 波特率已固定为 921600，无法再提高
- 单帧耗时 ≈ 曝光时间 + 传输时间
  （1338 字节 / 921600 bps ≈ 15ms）

### Q5: 无法获取 `wavelengths`？

- 需要在 `open()` 时成功读取波长范围
- 若设备不支持 0x0F 命令，`wavelengths` 会保持为 None
- 首次采集后 `num_points` 才会被确定

### Q6: 稳定性分析一直不判定稳定？

- 检查 `unstable_reasons` 字段，看具体哪一项不满足
- 光源本身可能确实不稳定（需要更长预热时间）
- 若噪声过大，可以调低 `correlation_min` 或调高 `cv_max`
- 检查窗口是否被填满（`report.ready`）

### Q7: CCT 计算结果异常？

- 检查波长数组与光强数组是否长度一致
- 检查光谱是否覆盖 380–780 nm（色度计算有效范围）
- 极暗的光谱可能导致 XYZ 求和接近 0，产生 NaN

### Q8: 如何扩展其他型号？

参照以下步骤：

1. 在 `protocol.py` 中新增命令枚举
2. 在 `spectrometer.py` 中新增对应方法
3. 若帧格式差异大，可新建 `SpectrometerModelX` 子类继承 `Spectrometer`

---

## 🛠 开发指南

### 代码风格

- 遵循 PEP 8，行宽 100
- 使用 `black` 格式化，`ruff` 静态检查
- 每个公开方法必须带 docstring

```bash
black src/ tests/ examples/
ruff check src/ tests/ examples/
```

### 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
feat: 增加 XXX 去噪算法
fix: 修复连续采集偶发丢帧
docs: 更新 README
refactor: 优化协议层解析逻辑
test: 补充 StabilityAnalyzer 单元测试
```

### 贡献流程

1. Fork 本仓库
2. 创建特性分支 `feature/your-feature`
3. 提交并推送
4. 创建 Pull Request

---

## 📁 项目结构

```
spectrometer-driver/
├── pyproject.toml              # 项目配置与依赖
├── README.md                   # 本文件
├── LICENSE                     # MIT 许可证
├── src/
│   └── spectrometer_driver/
│       ├── __init__.py         # 公共接口导出
│       ├── exceptions.py       # 自定义异常
│       ├── protocol.py         # 协议层：帧构建/解析
│       ├── transport.py        # 传输层：串口收发
│       ├── denoising.py        # 去噪层：5 种算法
│       ├── metrics.py          # 质量评价层
│       └── spectrometer.py     # 设备层：用户 API
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_continuous_acquisition.py
│   ├── 03_denoising_demo.py
│   ├── 04_live_plot.py
│   ├── 05_light_source_evaluation.py
│   └── 06_stability_analysis.py
└── tests/
    ├── __init__.py
    ├── test_protocol.py
    ├── test_denoising.py
    └── test_metrics.py
```

---

## 📄 许可证

MIT License，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 去噪算法参考：`scipy.signal`、`PyWavelets`、`scikit-learn`
- 色度计算参考：Wyman, Sloan & Shirley (2013)、McCamy (1992)


---

## 📊 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.2.0 | 2025-01 | 新增光源质量评价模块（单帧质量 + 时序稳定性 + 色度） |
| 0.1.0 | 2025-01 | 首个版本：协议层、传输层、设备层、去噪模块 |