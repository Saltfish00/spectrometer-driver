"""设备层：面向用户的高级 API。

============================================================
职责
============================================================

这一层把协议层的原始命令封装成符合直觉的方法与属性：

- 命令收发与响应类型校验
- 设备信息 / 波长范围 / 像素点数的缓存
- 采集数据的结构化封装（``SpectrumData``）
- 多帧采集与去噪的组合逻辑
- 线程安全与生命周期管理

============================================================
关键修复（v0.2.1）
============================================================

**Bug #2：``stop_continuous`` 遇脏帧中断**

  连续模式下缓冲区里有大量残留光谱帧，若某帧解析失败，
  原代码会直接退出循环，导致 STOP 响应被跳过。

  修复：遇脏帧跳过继续清理，循环结束后再次清空缓冲。

**Bug #3：波长范围查询失败无降级**

  若设备不支持 0x0F 命令，``_wavelength_start`` 一直为 None，
  所有依赖波长的功能（SNR、CCT 等）失效。

  修复：查询失败时降级使用默认 340~1000 nm。

**Bug #4：``_send_command`` 遇到脏响应无重试**

  串口缓冲区残留数据会导致响应类型不匹配。

  修复：自动清空缓冲并重试最多 3 次。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .denoising import Denoiser
from .exceptions import (
    ChecksumError,
    ConnectionError,
    DeviceError,
    ProtocolError,
    SpectrometerError,
    TimeoutError,
)
from .protocol import Command, build_command_frame
from .transport import SerialTransport


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class SpectrumData:
    """一次采集得到的光谱数据及其元信息。

    ============================================================
    字段说明
    ============================================================

    exposure_status : int
        曝光状态：0x00=正常, 0x01=过曝, 0x02=欠曝。

    exposure_time_us : int
        采集这一帧时设备使用的曝光时间（微秒）。

    spectrum_coefficient : int
        光谱系数 N。实际光强 = 原始值 / 10^N。

        例如：raw = 1300, N = 2 → intensity = 13.0。

    raw_data : numpy.ndarray
        原始光谱数据（uint16 数组）。

        - 单帧采集时形状 ``(num_points,)``
        - 多帧平均后形状 ``(num_points,)``

    wavelengths : numpy.ndarray | None
        每个像素对应的波长（nm），无法获取波长范围时为 None。

    frames : numpy.ndarray | None
        多帧采集时保存的所有原始帧，形状 ``(N, num_points)``；
        单帧采集时为 None。
    """

    exposure_status: int
    exposure_time_us: int
    spectrum_coefficient: int
    raw_data: np.ndarray
    wavelengths: Optional[np.ndarray] = None
    frames: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    @property
    def intensity(self) -> np.ndarray:
        """系数修正后的实际光强值（float64）。

        手册规定：实际光谱值 = 原始值 / 10^N。
        """
        return self.raw_data.astype(np.float64) / (10.0 ** self.spectrum_coefficient)

    @property
    def is_normal(self) -> bool:
        """曝光是否正常。"""
        return self.exposure_status == 0x00

    @property
    def is_overexposed(self) -> bool:
        """是否过曝。"""
        return self.exposure_status == 0x01

    @property
    def is_underexposed(self) -> bool:
        """是否欠曝。"""
        return self.exposure_status == 0x02

    @property
    def num_points(self) -> int:
        """光谱像素点数。"""
        return int(self.raw_data.size)

    @property
    def snr_gain(self) -> float:
        """多帧采集时的理论信噪比增益 = √N；单帧时为 1.0。"""
        if self.frames is None or self.frames.shape[0] <= 1:
            return 1.0
        return float(np.sqrt(self.frames.shape[0]))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpectrumData(num_points={self.num_points}, "
            f"exposure_time={self.exposure_time_us}us, "
            f"status=0x{self.exposure_status:02X}, "
            f"coeff={self.spectrum_coefficient}"
            + (f", frames={self.frames.shape[0]}" if self.frames is not None else "")
            + ")"
        )


# ============================================================================
# 主驱动类
# ============================================================================

class Spectrometer:
    """TLM 系列串口光谱仪驱动。

    ============================================================
    典型用法
    ============================================================

    单帧采集::

        with Spectrometer("/dev/ttyUSB0") as spec:
            print(spec.device_info)
            print(spec.wavelength_range)

            spec.exposure_mode = "manual"
            spec.exposure_time = 100_000        # 100 ms
            data = spec.get_spectrum()

    多帧平均::

        with Spectrometer("COM3") as spec:
            data = spec.acquire_averaged(n_frames=100)

    连续采集::

        with Spectrometer("COM3") as spec:
            for frame in spec.iter_continuous(count=100):
                process(frame)
    """

    #: 单条命令响应的默认超时（秒）
    DEFAULT_TIMEOUT: float = 2.0

    #: 连续模式下单帧读取的默认超时（秒）
    CONTINUOUS_TIMEOUT: float = 2.0

    #: 响应类型不匹配时的最大重试次数
    MAX_SEND_RETRIES: int = 3

    #: 波长范围查询失败时的降级默认值
    #: 参考手册 TLM 系列的常见范围
    DEFAULT_WAVELENGTH_START: float = 340.0
    DEFAULT_WAVELENGTH_END: float = 1000.0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """创建一个光谱仪驱动实例（不自动打开串口）。

        :param port:     串口名称，如 ``"/dev/ttyUSB0"`` 或 ``"COM3"``
        :param baudrate: 波特率，默认 921600
        :param timeout:  单次读写超时（秒）
        """
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

        self._transport: Optional[SerialTransport] = None
        self._lock = threading.RLock()

        # 设备信息缓存（首次 open 时填充）
        self._device_info: Optional[str] = None
        self._wavelength_start: Optional[float] = None
        self._wavelength_end: Optional[float] = None
        self._num_points: Optional[int] = None

        # 连续采集状态
        self._continuous = False

    def open(self) -> None:
        """打开串口并读取设备基本信息。

        该方法在 ``__enter__`` 中被自动调用。

        ============================================================
        流程
        ============================================================

        1. 创建 SerialTransport 并打开串口
        2. 延迟 100ms 让设备稳定
        3. 清空输入缓冲（丢弃上电瞬态数据）
        4. 查询设备信息（0x08）
        5. 查询波长范围（0x0F）；失败时降级到默认值

        :raises ConnectionError: 串口打开失败
        :raises TimeoutError:    设备无响应
        :raises ProtocolError:   协议错误
        """
        with self._lock:
            # 幂等：已打开则直接返回
            if self._transport is not None and self._transport.is_open:
                return

            self._transport = SerialTransport(self._port, self._baudrate, self._timeout)
            try:
                # ---- 修复：延迟 + 清缓冲 ----
                time.sleep(0.1)
                self._transport.reset_input_buffer()

                # ---- 查询设备信息 ----
                self._device_info = self._query_device_info()

                # ---- 修复 3：波长范围查询失败降级 ----
                try:
                    self._wavelength_start, self._wavelength_end = (
                        self._query_wavelength_range()
                    )
                except (TimeoutError, ProtocolError, ChecksumError) as exc:
                    # 部分设备型号可能不支持 0x0F 命令，此时降级到默认值
                    self._wavelength_start = self.DEFAULT_WAVELENGTH_START
                    self._wavelength_end = self.DEFAULT_WAVELENGTH_END
                    self._device_info = (
                        f"{self._device_info} "
                        f"[波长范围查询失败, 使用默认 "
                        f"{self.DEFAULT_WAVELENGTH_START}~"
                        f"{self.DEFAULT_WAVELENGTH_END}nm: {exc}]"
                    )
            except Exception:
                # 初始化失败时保证串口被释放，避免资源泄漏
                self._transport.close()
                self._transport = None
                raise

    def close(self) -> None:
        """关闭串口（幂等）。

        若正处于连续采集模式，会先发送停止命令并清空缓冲区。
        """
        with self._lock:
            if self._transport is None:
                return
            try:
                if self._continuous:
                    try:
                        self.stop_continuous()
                    except Exception:
                        # 关闭阶段尽力而为，不阻断释放
                        pass
            finally:
                self._transport.close()
                self._transport = None
                self._continuous = False

    # ---- 上下文管理器 ----
    def __enter__(self) -> "Spectrometer":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        state = "open" if self.is_open else "closed"
        return f"<Spectrometer port={self._port} state={state}>"

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        """串口是否已打开。"""
        return self._transport is not None and self._transport.is_open

    @property
    def device_info(self) -> Optional[str]:
        """设备信息字符串，如 ``'T3200000000FTAH-323-0000'``。"""
        return self._device_info

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """``(起始波长, 终止波长)``，单位 nm。"""
        if self._wavelength_start is None or self._wavelength_end is None:
            raise SpectrometerError("尚未获取波长范围，请先调用 open()")
        return self._wavelength_start, self._wavelength_end

    @property
    def num_points(self) -> Optional[int]:
        """光谱像素点数；未采集过任何一帧时为 None。"""
        return self._num_points

    @property
    def wavelengths(self) -> Optional[np.ndarray]:
        """均匀分布的波长数组（nm）；信息不全时返回 None。

        由于设备只返回起止波长，这里用 ``np.linspace`` 均匀划分。
        若实际像素分布不是均匀的，需要额外标定。

        :return: 波长数组，或 None（如果还没获取到点数/范围）
        """
        if (
            self._num_points is None
            or self._wavelength_start is None
            or self._wavelength_end is None
        ):
            return None
        return np.linspace(
            self._wavelength_start, self._wavelength_end, self._num_points
        )

    @property
    def is_continuous(self) -> bool:
        """是否处于连续采集模式。"""
        return self._continuous

    # ------------------------------------------------------------------
    # 内部：命令收发
    # ------------------------------------------------------------------
    def _require_open(self) -> SerialTransport:
        """确认串口已打开，返回传输层实例。"""
        if self._transport is None or not self._transport.is_open:
            raise ConnectionError("串口未打开，请先调用 open() 或使用 with 语句")
        return self._transport

    def _send_command(
        self,
        cmd_type: int,
        data: bytes = b"",
        expected_type: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """发送命令并等待响应，返回响应数据部分。

        ============================================================
        修复 4：响应类型不匹配自动重试
        ============================================================

        若响应类型不匹配（通常是缓冲区有脏数据），
        清空输入缓冲后重发命令，最多 ``MAX_SEND_RETRIES`` 次。

        这样可以吸收上一会话残留的响应，避免第一次命令
        就读到错误的响应。

        :param cmd_type:      要发送的命令类型
        :param data:          命令数据
        :param expected_type: 期望的响应类型；None 表示与请求相同
        :param timeout:       等待响应超时
        :return:              响应数据段
        :raises ProtocolError: 重试后仍无法匹配响应类型
        """
        transport = self._require_open()
        if expected_type is None:
            expected_type = cmd_type
        timeout = timeout if timeout is not None else self._timeout

        last_type: Optional[int] = None
        for attempt in range(self.MAX_SEND_RETRIES):
            with self._lock:
                # 重试前清空输入缓冲（第一次不清，保留正常握手）
                if attempt > 0:
                    transport.reset_input_buffer()

                transport.write(build_command_frame(cmd_type, data))
                resp_type, resp_data = transport.read_frame(timeout=timeout)

            # 类型匹配 → 成功返回
            if resp_type == expected_type:
                return resp_data

            # 类型不匹配 → 记录实际类型，继续重试
            last_type = resp_type

        # 重试用尽
        raise ProtocolError(
            f"响应类型不匹配: 期望 0x{expected_type:02X}, "
            f"实际 0x{last_type:02X}（已重试 {self.MAX_SEND_RETRIES} 次）"
        )

    # ------------------------------------------------------------------
    # 设备信息 / 波长范围
    # ------------------------------------------------------------------
    def _query_device_info(self) -> str:
        """获取设备信息（命令 0x08，命名数据 0x18 表示期望 24 字节）。

        :return: 解码后的 ASCII 字符串，如 ``'T32B5C102341TPD-113-0005'``
        """
        resp = self._send_command(
            Command.GET_DEVICE_INFO,
            data=bytes([0x18]),    # 0x18 = 24，期望返回 24 字节
            expected_type=Command.GET_DEVICE_INFO,
        )
        # 去掉尾部 \x00 与空白，按 ASCII 解码
        return resp.rstrip(b"\x00").decode("ascii", errors="replace").strip()

    def _query_wavelength_range(self) -> tuple[float, float]:
        """获取波长范围（命令 0x0F，返回 2 个 uint16 小端）。

        :return: ``(起始波长, 终止波长)``，单位 nm
        :raises ProtocolError: 响应数据长度不足
        """
        resp = self._send_command(
            Command.GET_WAVELENGTH_RANGE,
            expected_type=Command.GET_WAVELENGTH_RANGE,
        )
        if len(resp) < 4:
            raise ProtocolError(f"波长范围响应数据过短: {len(resp)} 字节")
        start = int.from_bytes(resp[0:2], "little")
        end = int.from_bytes(resp[2:4], "little")
        return float(start), float(end)

    # ------------------------------------------------------------------
    # 单帧 / 连续采集
    # ------------------------------------------------------------------
    def get_spectrum(self, timeout: Optional[float] = None) -> SpectrumData:
        """单帧获取一帧光谱数据（命令 0x02）。

        :param timeout: 等待超时；None 表示使用默认值
        :return:        :class:`SpectrumData` 实例
        """
        resp = self._send_command(
            Command.SINGLE_ACQUISITION,
            expected_type=Command.SINGLE_ACQUISITION,
            timeout=timeout,
        )
        return self._parse_spectrum(resp)

    def start_continuous(self) -> None:
        """开始连续获取光谱（命令 0x03）。

        设备收到后会自动持续推送光谱帧，每帧的命令类型为 0x03。
        """
        self._send_command(
            Command.START_CONTINUOUS,
            expected_type=Command.START_CONTINUOUS,
        )
        self._continuous = True

    def read_continuous(self, timeout: Optional[float] = None) -> SpectrumData:
        """读取连续模式下的一帧光谱。

        :param timeout: 单帧读取超时
        :return:        :class:`SpectrumData`
        :raises SpectrometerError: 未处于连续模式
        :raises ProtocolError:     收到非预期的帧类型
        """
        if not self._continuous:
            raise SpectrometerError("未处于连续采集模式，请先调用 start_continuous()")

        transport = self._require_open()
        timeout = timeout if timeout is not None else self.CONTINUOUS_TIMEOUT

        with self._lock:
            resp_type, resp_data = transport.read_frame(timeout=timeout)

        # 连续模式下的帧类型必须是 START_CONTINUOUS (0x03)
        if resp_type != Command.START_CONTINUOUS:
            raise ProtocolError(
                f"连续模式收到意外帧类型: 0x{resp_type:02X}"
            )
        return self._parse_spectrum(resp_data)

    def stop_continuous(self) -> None:
        """停止连续获取（命令 0x04）。

        ============================================================
        修复 2：容忍脏帧、清空残留
        ============================================================

        停止命令发出后，缓冲区中可能还残留若干光谱帧；
        本方法会：

        1. 一直读取直到收到 0x04 响应或超时
        2. **遇到脏帧时跳过**（不再中断）
        3. 最后再清一次输入缓冲

        避免下一次单帧采集读到脏数据。
        """
        if not self._continuous:
            return

        transport = self._require_open()
        try:
            with self._lock:
                # 给清理阶段留足时间（至少 3 秒）
                cleanup_deadline = time.time() + max(self._timeout * 2, 3.0)

                # 发送停止命令
                transport.write(build_command_frame(Command.STOP_CONTINUOUS))

                # 一直读到 STOP 响应或超时
                while time.time() < cleanup_deadline:
                    try:
                        resp_type, _ = transport.read_frame(
                            timeout=cleanup_deadline - time.time()
                        )
                    except TimeoutError:
                        # 超时说明缓冲区空了，可以退出
                        break
                    except (ProtocolError, ChecksumError):
                        # 读到脏帧，跳过继续清理（关键修复）
                        continue

                    # 收到 STOP 响应，完成
                    if resp_type == Command.STOP_CONTINUOUS:
                        break

                # 最后再清一次输入缓冲，确保无残留
                transport.reset_input_buffer()
        finally:
            self._continuous = False

    def iter_continuous(
        self,
        count: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[SpectrumData]:
        """连续采集的迭代器。

        典型用法::

            for frame in spec.iter_continuous(count=100):
                process(frame)

        迭代器退出（正常结束或被 break）时自动停止连续模式。

        :param count:   最多采集帧数；None 表示无限，由调用方 break 退出
        :param timeout: 单帧读取超时
        """
        self.start_continuous()
        try:
            n = 0
            while count is None or n < count:
                yield self.read_continuous(timeout=timeout)
                n += 1
        finally:
            # 无论正常结束还是异常/break，都保证停止
            self.stop_continuous()

    # ------------------------------------------------------------------
    # 多帧采集 + 去噪
    # ------------------------------------------------------------------
    def acquire_averaged(
        self,
        n_frames: int,
        denoise_method: str = "frame_average",
        **denoise_kwargs,
    ) -> SpectrumData:
        """连续采集 ``n_frames`` 帧，并根据指定方法合成去噪后的光谱。

        :param n_frames:       采集帧数（≥1）
        :param denoise_method: 去噪方法名，可选：

            - ``"frame_average"``: 帧平均（默认，√N 信噪比提升）
            - ``"median"``:        逐帧中值滤波 + 帧平均
            - ``"savgol"``:        帧平均 + SG 平滑
            - ``"wavelet"``:       帧平均 + 小波去噪
            - ``"pca"``:           PCA 去噪

        :param denoise_kwargs: 传递给具体去噪算法的参数
        :return:               :class:`SpectrumData`，其中 ``raw_data`` 为
                               去噪结果，``frames`` 为全部原始帧
        """
        if n_frames < 1:
            raise ValueError(f"n_frames 必须 >= 1，当前为 {n_frames}")

        # ---- 1. 采集 N 帧 ----
        frames_list: list[np.ndarray] = []
        first: Optional[SpectrumData] = None
        for _ in range(n_frames):
            frame = self.get_spectrum()
            if first is None:
                first = frame    # 记录第一帧的元信息
            frames_list.append(frame.raw_data)

        assert first is not None
        # 堆叠为 (N, num_points)
        frames = np.stack(frames_list, axis=0)

        # ---- 2. 去噪 ----
        denoised = Denoiser.auto_denoise(
            frames, method=denoise_method, **denoise_kwargs
        )

        # ---- 3. 打包结果 ----
        return SpectrumData(
            exposure_status=first.exposure_status,
            exposure_time_us=first.exposure_time_us,
            spectrum_coefficient=first.spectrum_coefficient,
            raw_data=denoised,
            wavelengths=first.wavelengths,
            frames=frames,
        )

    # ------------------------------------------------------------------
    # 曝光控制
    # ------------------------------------------------------------------
    def set_manual_exposure_mode(self) -> None:
        """切换为手动曝光模式（命令 0x0A）。"""
        self._check_ack(
            self._send_command(
                Command.SET_MANUAL_EXPOSURE_MODE,
                expected_type=Command.SET_MANUAL_EXPOSURE_MODE,
            ),
            "设置手动曝光模式",
        )

    def set_auto_exposure_mode(self) -> None:
        """切换为自动曝光模式（命令 0x0B）。"""
        self._check_ack(
            self._send_command(
                Command.SET_AUTO_EXPOSURE_MODE,
                expected_type=Command.SET_AUTO_EXPOSURE_MODE,
            ),
            "设置自动曝光模式",
        )

    @property
    def exposure_mode(self) -> str:
        """曝光模式（只写属性，读取会抛异常）。

        之所以设计成只写：手册没有提供"读取当前模式"的命令。
        """
        raise SpectrometerError(
            "曝光模式不支持读取，请直接赋值 'manual' 或 'auto'"
        )

    @exposure_mode.setter
    def exposure_mode(self, mode: str) -> None:
        """设置曝光模式。

        :param mode: ``"manual"`` 或 ``"auto"``
        """
        mode_lower = mode.lower()
        if mode_lower == "manual":
            self.set_manual_exposure_mode()
        elif mode_lower == "auto":
            self.set_auto_exposure_mode()
        else:
            raise ValueError("exposure_mode 只能为 'manual' 或 'auto'")

    def set_exposure_time(self, microseconds: int) -> None:
        """设置曝光时间（命令 0x0C）。

        :param microseconds: 曝光时间，单位微秒，范围 0 ~ 0xFFFFFFFF
        """
        if not 0 <= microseconds <= 0xFFFFFFFF:
            raise ValueError(
                f"曝光时间必须在 0 ~ 0xFFFFFFFF 微秒之间，当前为 {microseconds}"
            )
        # 4 字节小端
        payload = microseconds.to_bytes(4, "little")
        self._check_ack(
            self._send_command(
                Command.SET_EXPOSURE,
                data=payload,
                expected_type=Command.SET_EXPOSURE,
            ),
            "设置曝光时间",
        )

    def get_exposure_time(self) -> int:
        """读取当前曝光时间（命令 0x0D），单位微秒。"""
        resp = self._send_command(
            Command.GET_EXPOSURE,
            expected_type=Command.GET_EXPOSURE,
        )
        if len(resp) < 4:
            raise ProtocolError(f"曝光值响应过短: {len(resp)} 字节")
        return int.from_bytes(resp[0:4], "little")

    @property
    def exposure_time(self) -> int:
        """当前曝光时间（μs），可读可写。"""
        return self.get_exposure_time()

    @exposure_time.setter
    def exposure_time(self, microseconds: int) -> None:
        self.set_exposure_time(microseconds)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _check_ack(resp: bytes, action: str) -> None:
        """检查设置类命令的返回状态码。

        协议规定：返回数据首字节 0x00 表示成功，其他值表示失败。

        :param resp:   响应数据段
        :param action: 操作名称（用于错误信息）
        :raises ProtocolError: 响应数据为空
        :raises DeviceError:   设备返回非 0x00 状态码
        """
        if len(resp) < 1:
            raise ProtocolError(f"{action} 响应数据为空")
        if resp[0] != 0x00:
            raise DeviceError(resp[0], f"{action} 失败")

    def _parse_spectrum(self, data: bytes) -> SpectrumData:
        """解析单帧/连续模式的光谱数据段。

        数据段结构（来自手册）::

            曝光状态(1) + 曝光时间(4, LE) + 光谱系数(2, LE) + 光谱数据(2*N, LE)

        ============================================================
        解析流程
        ============================================================

        1. 前 7 字节：元信息（曝光状态、时间、系数）
        2. 剩余字节：光谱数据（uint16 数组）

        :param data: 响应数据段
        :return:     :class:`SpectrumData`
        :raises ProtocolError: 数据段长度不足或非偶数字节
        """
        # 至少 7 字节：1 + 4 + 2
        if len(data) < 7:
            raise ProtocolError(f"光谱数据段过短: {len(data)} 字节")

        # ---- 前 7 字节：元信息 ----
        exposure_status = data[0]
        exposure_time_us = int.from_bytes(data[1:5], "little")
        spectrum_coeff = int.from_bytes(data[5:7], "little")

        # ---- 剩余部分：光谱数据 ----
        spectrum_bytes = data[7:]
        if len(spectrum_bytes) % 2 != 0:
            raise ProtocolError(
                f"光谱数据长度不是偶数字节: {len(spectrum_bytes)}"
            )

        # uint16 小端序数组
        # copy() 保证数组可写且不共享底层缓冲
        raw = np.frombuffer(spectrum_bytes, dtype="<u2").copy()

        # 首次采集时记录点数，以便后续计算波长
        if self._num_points is None:
            self._num_points = int(raw.size)

        return SpectrumData(
            exposure_status=exposure_status,
            exposure_time_us=exposure_time_us,
            spectrum_coefficient=spectrum_coeff,
            raw_data=raw,
            wavelengths=self.wavelengths,   # 此时 wavelengths 可能为 None
            frames=None,
        )