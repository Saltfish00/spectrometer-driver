"""示例 5：单帧信号质量评价。

运行流程：
1. 遮光采集暗帧（估计暗噪声、杂散光）
2. 打开光源采集一帧信号
3. 输出 SNR、饱和情况、杂散光比例
"""

from spectrometer_driver import Spectrometer, evaluate_single_frame

PORT = "COM5"


def main() -> None:
    with Spectrometer(PORT) as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 100_000  # 100ms

        # ---------- 1. 采集暗帧 ----------
        input("请遮光，然后按 Enter 采集暗帧...")
        dark = spec.acquire_averaged(n_frames=50, denoise_method="frame_average")
        wavelengths = dark.wavelengths
        print(f"暗帧均值 : {dark.intensity.mean():.4f}")
        print(f"暗帧标准差: {dark.intensity.std():.4f}")

        # ---------- 2. 采集信号帧 ----------
        input("请打开光源，然后按 Enter 采集信号帧...")
        signal = spec.acquire_averaged(n_frames=50, denoise_method="frame_average")

        # ---------- 3. 评价 ----------
        metrics = evaluate_single_frame(
            intensity=signal.intensity,
            wavelengths=wavelengths,
            dark=dark.intensity,
            band=(650.0, 800.0),
            adc_max=65535,               # 16 位 ADC
            saturation_threshold=0.95,
        )

        print("\n===== 单帧信号质量报告 =====")
        print(f"  暗噪声标准差      : {metrics.dark_noise:.4f}")
        print(f"  噪声估计（差分法）: {metrics.noise_estimate:.4f}")
        print(f"  650-800 nm 均值   : {metrics.band_signal_mean:.2f}")
        print(f"  信噪比 SNR        : {metrics.snr:.1f}")
        print(f"  饱和像素数        : {metrics.saturation_count} "
              f"({metrics.saturation_ratio * 100:.4f}%)")
        print(f"  杂散光/环境光比例 : {metrics.stray_light_ratio}")


if __name__ == "__main__":
    main()