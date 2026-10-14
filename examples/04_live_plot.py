"""示例 4：实时绘图显示光谱（连续采集 + 动态刷新）。

需要安装 matplotlib: pip install matplotlib
按 Ctrl+C 退出。
"""

import matplotlib.pyplot as plt

from spectrometer_driver import Spectrometer

PORT = "COM5"
REFRESH_EVERY = 5  # 每 5 帧刷新一次画面，降低 UI 开销


def main() -> None:
    with Spectrometer(PORT) as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 30_000

        # 打开交互模式
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        (line,) = ax.plot([], [], lw=0.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title("实时光谱（Ctrl+C 退出）")
        fig.tight_layout()

        try:
            for i, frame in enumerate(spec.iter_continuous()):
                # 每 REFRESH_EVERY 帧刷新一次，减少 UI 抖动
                if i % REFRESH_EVERY == 0 and frame.wavelengths is not None:
                    line.set_data(frame.wavelengths, frame.intensity)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)
        except KeyboardInterrupt:
            print("\n用户中断，退出。")
        finally:
            plt.ioff()
            plt.close(fig)


if __name__ == "__main__":
    main()