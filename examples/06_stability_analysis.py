"""示例 6：光源稳定性分析（滑动窗口 + 稳定时间判定）。

流程：
1. 预热 1 秒
2. 连续采集 5 秒
3. 用 StabilityAnalyzer 逐帧喂入
4. 输出每个窗口的稳定性指标，并给出 t_stable
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from spectrometer_driver import (
    Spectrometer,
    StabilityAnalyzer,
    StabilityCriteria,
)

PORT = "COM5"
DURATION_SEC = 5.0          # 采集总时长
WINDOW_SEC = 1.0            # 滑动窗口长度


def main() -> None:
    # 更严格的判据可根据需要调整
    criteria = StabilityCriteria(
        cv_max=0.01,               # CV < 1%
        correlation_min=0.999,     # r > 0.999
        delta_uv_max=0.002,        # Δu'v' < 0.002
        centroid_drift_max=0.2,    # 重心漂移 < 0.2 nm
    )

    with Spectrometer(PORT) as spec:
        spec.exposure_mode = "manual"
        spec.exposure_time = 30_000  # 30ms

        # 预热 1 秒（光源本身也需要稳定时间）
        print("预热 1 秒...")
        time.sleep(1.0)

        # 拿到波长数组
        probe = spec.get_spectrum()
        wavelengths = probe.wavelengths

        analyzer = StabilityAnalyzer(
            wavelengths=wavelengths,
            criteria=criteria,
            window_seconds=WINDOW_SEC,
            band=(650.0, 800.0),
            numerator_band=(650.0, 800.0),
            denominator_band=(400.0, 500.0),
        )

        # 记录时间序列，便于后续绘图
        log = {
            "t": [],
            "integral": [],
            "cv": [],
            "r_min": [],
            "delta_uv": [],
            "centroid": [],
            "cct": [],
            "stable": [],
        }

        print(f"\n开始连续采集 {DURATION_SEC} 秒...")
        t_start = time.time()

        # iter_continuous 是无限迭代器，我们在循环里主动 break
        for frame in spec.iter_continuous():
            now = time.time() - t_start
            if now > DURATION_SEC:
                break

            report = analyzer.add_frame(frame.intensity, timestamp=now)

            # 只记录窗口就绪后的数据
            if report.ready:
                log["t"].append(now)
                log["integral"].append(report.integral_mean)
                log["cv"].append(report.integral_cv)
                log["r_min"].append(report.correlation_min)
                log["delta_uv"].append(report.delta_uv_max)
                log["centroid"].append(report.centroid_mean)
                log["cct"].append(report.cct_mean)
                log["stable"].append(report.is_stable)

        # ---------- 结果汇总 ----------
        print("\n===== 稳定性分析汇总 =====")
        if analyzer.t_stable is not None:
            print(f"稳定时间 t_stable = {analyzer.t_stable:.3f} 秒")
        else:
            print("未能在采集时段内判定稳定")

        if log["t"]:
            print(f"积分强度 CV       : {np.mean(log['cv']) * 100:.3f}%")
            print(f"相关系数最小值    : {np.min(log['r_min']):.5f}")
            print(f"Δu'v' 最大值      : {np.max(log['delta_uv']):.5f}")
            print(f"色温均值          : {np.mean(log['cct']):.1f} K")
            print(f"色温波动范围      : "
                  f"{np.max(log['cct']) - np.min(log['cct']):.1f} K")

        # ---------- 可视化 ----------
        if log["t"]:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            axes[0].plot(log["t"], log["integral"], lw=0.8)
            axes[0].set_ylabel("积分强度 I(t)")

            axes[1].plot(log["t"], np.array(log["cv"]) * 100, label="CV (%)")
            axes[1].axhline(criteria.cv_max * 100, ls="--", c="r",
                            label=f"阈值 {criteria.cv_max * 100:.1f}%")
            axes[1].set_ylabel("CV (%)")
            axes[1].legend()

            axes[2].plot(log["t"], log["delta_uv"], label="Δu'v'")
            axes[2].axhline(criteria.delta_uv_max, ls="--", c="r",
                            label=f"阈值 {criteria.delta_uv_max}")
            axes[2].set_ylabel("Δu'v'")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()

            # 标记稳定时间
            if analyzer.t_stable is not None:
                for ax in axes:
                    ax.axvline(analyzer.t_stable, ls=":", c="g",
                               label=f"t_stable={analyzer.t_stable:.2f}s")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()