"""光谱去噪模块：提供多种主流去噪方法与自动化组合策略。

============================================================
算法总览
============================================================

==============  ===================================  ==================
方法            原理                                 适用场景
==============  ===================================  ==================
frame_average   多帧逐像素求均值                     随机噪声（首选）
median_filter   邻域中值替代当前像素                  脉冲噪声（尖峰）
savgol_filter   局部多项式最小二乘拟合                保峰形平滑
wavelet         多尺度分解 + 阈值化细节系数           多尺度噪声
pca             降维保留主成分                        大量帧、高维数据
==============  ===================================  ==================

============================================================
设计原则
============================================================

1. 所有方法均为 ``@staticmethod``，无状态，可独立调用
2. 输入输出统一使用 ``numpy.ndarray``，便于与设备层衔接
3. 提供 ``auto_denoise`` 作为推荐的一站式入口
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter as ndi_median_filter
from scipy.signal import savgol_filter as sp_savgol_filter

# ---- 可选依赖：PyWavelets（小波去噪）----
try:
    import pywt
    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _HAS_PYWT = False

# ---- 可选依赖：scikit-learn（PCA 去噪）----
try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


class Denoiser:
    """光谱去噪器。

    使用示例::

        from spectrometer_driver import Denoiser
        import numpy as np

        # 假设 frames 形状为 (100, 2048)
        frames = np.random.randn(100, 2048)

        # 帧平均
        avg = Denoiser.frame_average(frames)

        # 帧平均 + SG 平滑
        smooth = Denoiser.savgol_filter(avg, window_length=11, polyorder=3)

        # 一站式入口
        result = Denoiser.auto_denoise(frames, method="median", kernel_size=5)
    """

    # ==================================================================
    # 1. 帧平均
    # ==================================================================
    @staticmethod
    def frame_average(frames: np.ndarray) -> np.ndarray:
        """帧平均：对 N 帧数据逐像素求均值。

        ============================================================
        原理
        ============================================================

        光谱采集最常用的去噪方法：

        - 真实信号在帧间**一致** → 求平均后保留
        - 随机噪声在帧间**不相关** → 求平均后相互抵消

        理论上信噪比提升 √N 倍：10 帧提升 3.16 倍，
        100 帧提升 10 倍。

        ============================================================
        边界情况
        ============================================================

        - 输入必须是 2D（N, num_points），否则报错
        - N = 1 时相当于直接返回该帧

        :param frames: 形状为 ``(N, num_points)`` 的二维数组
        :return:       平均后的单帧数据，形状 ``(num_points,)``
        :raises ValueError: 输入不是二维数组
        """
        if frames.ndim != 2:
            raise ValueError(
                f"frames 必须是二维数组 (N, num_points)，当前维度为 {frames.ndim}"
            )
        # 沿轴 0（帧维度）求均值
        return np.mean(frames, axis=0)

    # ==================================================================
    # 2. 中值滤波
    # ==================================================================
    @staticmethod
    def median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """中值滤波：去除脉冲噪声（如宇宙射线导致的尖峰）。

        ============================================================
        原理
        ============================================================

        用邻域像素的中值替代当前像素值。相比均值滤波，
        中值滤波**不会因为尖峰而"污染"邻域**。

        例如：邻域 = [10, 12, 11, 5000, 13]
        - 均值 = 1009（被尖峰拖累）
        - 中值 = 12（正确）

        ============================================================
        参数选择
        ============================================================

        - ``kernel_size`` = 3：轻度去尖峰
        - ``kernel_size`` = 5：常用值
        - ``kernel_size`` = 7：更强去尖峰，但会平滑窄峰

        :param data:        一维光谱数据
        :param kernel_size: 滤波窗口大小（会被强制为奇数）
        :return:            滤波后的数据
        """
        # 强制奇数窗口（偶数窗口在 scipy 中会自动偏移，不直观）
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        return ndi_median_filter(data.astype(np.float64), size=kernel_size)

    # ==================================================================
    # 3. Savitzky-Golay 滤波
    # ==================================================================
    @staticmethod
    def savgol_filter(
        data: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> np.ndarray:
        """Savitzky-Golay 平滑滤波。

        ============================================================
        原理
        ============================================================

        在滑动窗口内对数据做低阶多项式最小二乘拟合，
        取窗口中心点的拟合值。

        相比简单的移动平均，SG 滤波能**更好地保留光谱的
        峰高和峰宽**——因为它在窗口内拟合了一个多项式，
        相当于"预知"了峰的曲率。

        ============================================================
        参数选择建议
        ============================================================

        - ``window_length``：通常取峰宽的 0.5~1 倍
          - 太小 → 平滑不足
          - 太大 → 过度平滑、削峰

        - ``polyorder``：2~4 阶即可
          - 太小 → 拟合不足
          - 太大 → 引入振铃（oscillation）

        必须满足：``window_length > polyorder``

        :param data:          一维光谱数据
        :param window_length: 窗口长度（会被强制为奇数）
        :param polyorder:     多项式阶数，必须小于 window_length
        :return:              平滑后的数据
        """
        # 保证 window_length 是奇数
        if window_length % 2 == 0:
            window_length += 1

        # 保证 window_length > polyorder（否则 scipy 报错）
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

        return sp_savgol_filter(
            data.astype(np.float64),
            window_length=window_length,
            polyorder=polyorder,
        )

    # ==================================================================
    # 4. 小波去噪
    # ==================================================================
    @staticmethod
    def wavelet_denoise(
        data: np.ndarray,
        wavelet: str = "db8",
        level: int = 3,
        threshold_mode: str = "soft",
    ) -> np.ndarray:
        """小波阈值去噪。

        ============================================================
        流程
        ============================================================

        1. **多层小波分解**：得到近似系数 cA 与各层细节系数 cD1, cD2, ...
        2. **估计噪声标准差**：σ = MAD(cD1) / 0.6745
        3. **计算通用阈值**：λ = σ √(2 ln N)
        4. **阈值收缩**：对各层细节系数做软/硬阈值处理
        5. **小波重构**：得到去噪后的信号

        ============================================================
        通用阈值公式（Universal Threshold）
        ============================================================

        .. math::

            \\lambda = \\sigma \\sqrt{2 \\ln N}, \\quad
            \\sigma = \\frac{\\text{MAD}(cD_1)}{0.6745}

        其中 ``MAD`` 为中位数绝对偏差，``0.6745`` 是正态分布
        的 0.75 分位点对应的 σ 值。

        ============================================================
        软阈值 vs 硬阈值
        ============================================================

        - **硬阈值**：|x| < λ 时置 0，否则保留 x。简单，但可能
          产生不连续跳跃。
        - **软阈值**：|x| < λ 时置 0，否则收缩为 sign(x)(|x| - λ)。
          更平滑，推荐用于光谱。

        :param data:           一维光谱数据
        :param wavelet:        小波基名称（``'db8'``、``'sym6'``、``'coif5'``）
        :param level:          分解层数，一般 3~5
        :param threshold_mode: 阈值模式（``'soft'`` 或 ``'hard'``）
        :return:               去噪后的数据（长度与输入一致）
        :raises ImportError:   未安装 PyWavelets
        """
        if not _HAS_PYWT:
            raise ImportError("小波去噪需要安装 PyWavelets: pip install PyWavelets")

        x = data.astype(np.float64)

        # ---- 1. 多层小波分解 ----
        # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # 第一个元素是近似系数，其余是细节系数（从粗到细）
        coeffs = pywt.wavedec(x, wavelet, level=level)

        # ---- 2. 用 MAD 估计噪声标准差 ----
        # 最细尺度的细节系数 cD1 主要由噪声构成
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        # 避免全零信号导致阈值为 0（相当于不去噪）
        if sigma == 0:
            return x.copy()

        # ---- 3. 计算通用阈值 ----
        threshold = sigma * np.sqrt(2.0 * np.log(len(x)))

        # ---- 4. 对细节系数做阈值收缩 ----
        # 近似系数 cA 保持不变（保留信号的主要成分）
        coeffs_thresh = [coeffs[0]]
        for c in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(c, threshold, mode=threshold_mode))

        # ---- 5. 小波重构 ----
        reconstructed = pywt.waverec(coeffs_thresh, wavelet)
        # 重构后长度可能与原信号略有偏差，做截断
        return reconstructed[: len(x)]

    # ==================================================================
    # 5. PCA 去噪
    # ==================================================================
    @staticmethod
    def pca_denoise(frames: np.ndarray, n_components: int = 5) -> np.ndarray:
        """PCA 去噪：对多帧数据做主成分分析，保留前 K 个主成分重构。

        ============================================================
        原理
        ============================================================

        - **真实信号**在帧间高度相关 → 集中在少数主成分
        - **随机噪声**在帧间不相关 → 分散在众多次要成分中

        因此只保留前 K 个主成分即可显著去噪。对于 N ≫ K 的
        多帧数据，效果常优于简单的帧平均。

        ============================================================
        注意事项
        ============================================================

        - ``n_components`` 必须小于 ``min(N, num_points)``
        - 帧数太少（如 N < 10）时估计不稳定
        - 计算量大，适合离线分析；实时场景推荐帧平均

        :param frames:       形状为 ``(N, num_points)`` 的多帧数据
        :param n_components: 保留的主成分个数
        :return:             去噪后平均光谱，形状 ``(num_points,)``
        :raises ImportError: 未安装 scikit-learn
        :raises ValueError:  输入维度或参数非法
        """
        if not _HAS_SKLEARN:
            raise ImportError("PCA 去噪需要安装 scikit-learn: pip install scikit-learn")
        if frames.ndim != 2:
            raise ValueError("frames 必须是二维数组 (N, num_points)")

        n_samples, n_features = frames.shape
        max_components = min(n_samples, n_features)
        if not 1 <= n_components <= max_components:
            raise ValueError(
                f"n_components 必须在 1~{max_components} 之间，当前为 {n_components}"
            )

        # ---- 1. 拟合 PCA ----
        # 使用 randomized SVD 求解器加速大矩阵
        # random_state=0 保证结果可复现
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
        scores = pca.fit_transform(frames.astype(np.float64))

        # ---- 2. 用前 K 个主成分重构 ----
        reconstructed = pca.inverse_transform(scores)

        # ---- 3. 平均得到最终光谱 ----
        return np.mean(reconstructed, axis=0)

    # ==================================================================
    # 6. 自动化组合去噪
    # ==================================================================
    @staticmethod
    def auto_denoise(
        frames: np.ndarray,
        method: str = "frame_average",
        **kwargs,
    ) -> np.ndarray:
        """一站式去噪入口，按方法名分派。

        :param frames: 形状为 ``(N, num_points)`` 的多帧数据
        :param method: 去噪方法名，可选：

            - ``"frame_average"``: 帧平均（默认）
            - ``"median"``:        逐帧中值滤波后再帧平均
            - ``"savgol"``:        帧平均后 SG 平滑
            - ``"wavelet"``:       帧平均后小波去噪
            - ``"pca"``:           PCA 去噪

        :param kwargs: 传递给具体去噪算法的参数
        :return:       去噪后的单帧数据 ``(num_points,)``
        :raises ValueError: 不支持的去噪方法

        使用示例::

            # 帧平均
            avg = Denoiser.auto_denoise(frames)

            # 中值滤波 + 帧平均
            avg = Denoiser.auto_denoise(frames, method="median", kernel_size=5)

            # 帧平均 + SG 平滑
            avg = Denoiser.auto_denoise(frames, method="savgol",
                                        window_length=11, polyorder=3)

            # PCA
            avg = Denoiser.auto_denoise(frames, method="pca", n_components=5)
        """
        # ---- 分支 1：单纯帧平均 ----
        if method == "frame_average":
            return Denoiser.frame_average(frames)

        # ---- 分支 2：逐帧中值滤波 + 帧平均 ----
        # 先对每帧独立去脉冲噪声，再帧平均
        if method == "median":
            kernel = kwargs.get("kernel_size", 5)
            cleaned = np.stack(
                [Denoiser.median_filter(f, kernel_size=kernel) for f in frames]
            )
            return Denoiser.frame_average(cleaned)

        # ---- 分支 3：帧平均 + SG 平滑 ----
        # 先帧平均降随机噪声，再 SG 平滑
        if method == "savgol":
            avg = Denoiser.frame_average(frames)
            return Denoiser.savgol_filter(
                avg,
                window_length=kwargs.get("window_length", 11),
                polyorder=kwargs.get("polyorder", 3),
            )

        # ---- 分支 4：帧平均 + 小波去噪 ----
        if method == "wavelet":
            avg = Denoiser.frame_average(frames)
            return Denoiser.wavelet_denoise(
                avg,
                wavelet=kwargs.get("wavelet", "db8"),
                level=kwargs.get("level", 3),
                threshold_mode=kwargs.get("threshold_mode", "soft"),
            )

        # ---- 分支 5：PCA 去噪 ----
        if method == "pca":
            return Denoiser.pca_denoise(
                frames, n_components=kwargs.get("n_components", 5)
            )

        # ---- 未知方法 ----
        raise ValueError(f"不支持的去噪方法: {method}")