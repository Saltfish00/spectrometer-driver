"""示例 2：连续采集 N 帧并做基本统计。

演示 `iter_continuous` 迭代器的用法。
"""

import numpy as np

from spectrometer_driver import Spectrometer

PORT = "COM5"
N_FRAMES = 200


def main() -> None:
    with Spectrometer(PORT) as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 50_000  # 50 ms

        print(f"开始连续采集 {N_FRAMES} 帧...")
        all_frames = []

        # iter_continuous 是生成器；退出时会自动 stop_continuous
        for i, frame in enumerate(spec.iter_continuous(count=N_FRAMES)):
            all_frames.append(frame.raw_data)
            if (i + 1) % 20 == 0:
                print(f"  已采集 {i + 1}/{N_FRAMES} 帧")

        # 堆叠为 (N, num_points)
        frames = np.stack(all_frames, axis=0)
        print(f"数据形状: {frames.shape}")
        print(f"理论 SNR 增益: sqrt({N_FRAMES}) = {np.sqrt(N_FRAMES):.2f} 倍")

        # 帧间统计
        print(f"像素方差均值 : {frames.var(axis=0).mean():.2f}")
        avg = frames.mean(axis=0)
        print(f"平均光谱最大值: {avg.max():.0f}")


if __name__ == "__main__":
    main()