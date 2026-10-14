"""示例 1：单帧采集与设备信息读取。

运行前请把 PORT 改成实际的串口名称。
Linux:   "/dev/ttyUSB0"
Windows: "COM3"
"""

from spectrometer_driver import Spectrometer

PORT = "COM5"


def main() -> None:
    # 使用 with 语句确保串口一定被关闭
    with Spectrometer(PORT) as spec:
        print("=" * 60)
        print(f"设备信息 : {spec.device_info}")
        print(f"波长范围 : {spec.wavelength_range[0]:.1f} ~ "
              f"{spec.wavelength_range[1]:.1f} nm")
        print("=" * 60)

        # 设置手动曝光 + 100ms 曝光时间
        spec.exposure_mode = "manual"
        spec.exposure_time = 100_000        # 微秒
        print(f"当前曝光 : {spec.exposure_time} us")

        # 单帧采集
        data = spec.get_spectrum()
        print(f"光谱点数 : {data.num_points}")
        print(f"曝光状态 : {'正常' if data.is_normal else hex(data.exposure_status)}")
        print(f"光谱系数 : {data.spectrum_coefficient}")

        # 数据访问示例
        print(f"原始数据 (前5): {data.raw_data[:5]}")
        print(f"实际光强 (前5): {data.intensity[:5]}")
        if data.wavelengths is not None:
            print(f"波长 (前5)   : {data.wavelengths[:5]}")

        # 寻找峰值波长
        peak_idx = int(data.intensity.argmax())
        if data.wavelengths is not None:
            print(f"峰值位置 : {data.wavelengths[peak_idx]:.1f} nm "
                  f"(强度 {data.intensity[peak_idx]:.2f})")


if __name__ == "__main__":
    main()