"""示例 3：对比 5 种去噪方法的效果。

需要安装 matplotlib: pip install matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrometer_driver import Spectrometer, Denoiser

PORT = "COM5"
N_FRAMES = 100


def main() -> None:
    with Spectrometer(PORT) as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 30_000

        # 采集 N 帧原始数据
        print(f"采集 {N_FRAMES} 帧原始数据...")
        data = spec.acquire_averaged(n_frames=N_FRAMES)

        # 拿到多帧矩阵、波长数组、系数
        frames = data.frames                    # (N, num_points)
        wl = data.wavelengths
        coeff = 10.0 ** data.spectrum_coefficient

        # 单帧作为基线
        single = frames[0].astype(np.float64) / coeff

        # 五种去噪方法
        results = {
            "单帧 (基线)": single,
            "帧平均 (N=100)": Denoiser.auto_denoise(frames, "frame_average") / coeff,
            "中值 + 帧平均": Denoiser.auto_denoise(frames, "median", kernel_size=5) / coeff,
            "帧平均 + SG": Denoiser.auto_denoise(
                frames, "savgol", window_length=11, polyorder=3
            ) / coeff,
            "帧平均 + 小波": Denoiser.auto_denoise(
                frames, "wavelet", wavelet="db8", level=3
            ) / coeff,
        }

        # 绘制对比图
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 10), sharex=True)
        for ax, (name, y) in zip(axes, results.items()):
            if wl is not None:
                ax.plot(wl, y, lw=0.8)
                ax.set_ylabel(name, fontsize=9)
            else:
                ax.plot(y, lw=0.8)
                ax.set_ylabel(name, fontsize=9)
        axes[-1].set_xlabel("Wavelength (nm)" if wl is not None else "Pixel index")
        plt.tight_layout()
        plt.show()

        # 定量评估：计算平坦区域的标准差
        print("\n平坦区标准差对比（越小越好）:")
        for name, y in results.items():
            print(f"  {name:<20}: {y[:100].std():.4f}")


if __name__ == "__main__":
    main()