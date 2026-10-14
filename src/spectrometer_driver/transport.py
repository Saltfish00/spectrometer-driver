"""传输层：只关心字节流的收发与帧同步，不关心命令语义。

============================================================
职责
============================================================

1. 打开 / 关闭串口（参数由手册固定为 921600-8-N-1）
2. 原始字节的读写
3. 从字节流中同步到合法帧头并拼装出完整帧
4. 线程安全（使用 ``RLock`` 保护）

============================================================
关键修复（v0.2.1）
============================================================

**Bug #1：光谱数据里偶然出现的 CC 81 会被误认为帧头**

  光谱数据段 1322 字节里出现 ``CC 81`` 的概率约 1%~3%
  （例如某像素值 = 0x81CC，小端存储为 CC 81）。
  简单的"找到 CC 81 就认为帧头"会导致误同步。

**修复方案**：使用 **5 字节滑动窗口**：
    - 检查前 2 字节是否 CC 81
    - 若匹配，再检查后 3 字节表示的总长是否在合理范围
    - 任意一项不满足则窗口右移 1 字节，继续查找

**Bug #5：空 chunk 时 CPU 空转**

  串口读超时时 pyserial 返回空 bytes，循环紧密重试会
  消耗大量 CPU。

**修复方案**：空 chunk 时 ``time.sleep(0.0001)``。

============================================================
超时设计
============================================================

使用统一的 ``deadline`` 而非每步单独计时：

- ``deadline = time.time() + timeout``（绝对截止时间）
- 每步读之前检查 ``remaining = deadline - now``
- 若 remaining <= 0 则抛 TimeoutError

这样即使中间多次循环，总耗时也不会超过 timeout。
"""

from __future__ import annotations

import threading
import time

import serial  # pyserial

from .exceptions import ChecksumError, ConnectionError, ProtocolError, TimeoutError
from .protocol import MIN_FRAME_LEN, parse_response_frame


class SerialTransport:
    """串口传输层实现。

    典型用法::

        with SerialTransport("/dev/ttyUSB0") as transport:
            transport.write(b"...")
            cmd_type, data = transport.read_frame()

    或者手动管理::

        transport = SerialTransport("/dev/ttyUSB0")
        try:
            ...
        finally:
            transport.close()
    """

    #: 单帧最大合理字节数
    #:
    #: 光谱帧 = 曝光状态(1) + 时间(4) + 系数(2) + 数据(N×2)。
    #: 661 点时总长约 1338 字节。这里给足余量到 20000，
    #: 用于拒绝明显不合理的"伪帧头"（数据里的 CC 81）。
    MAX_FRAME_LEN: int = 20000

    #: 一次 ``read_frame`` 最多尝试几次同步
    #:
    #: 5 次足够：一次真实数据读取的误同步概率 < 1e-6，
    #: 5 次连续误同步的概率几乎为 0。
    MAX_SYNC_RETRIES: int = 5

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 2.0,
    ) -> None:
        """打开串口。

        :param port:     串口名称，如 ``"/dev/ttyUSB0"`` 或 ``"COM3"``
        :param baudrate: 波特率，默认 921600（手册规定值）
        :param timeout:  单次读写超时（秒）
        :raises ConnectionError: 串口打开失败
        """
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,       # 数据位 8
                parity=serial.PARITY_NONE,       # 校验位 NONE
                stopbits=serial.STOPBITS_ONE,    # 停止位 1
                timeout=timeout,                 # 单次 read 超时
                write_timeout=timeout,           # 单次 write 超时
            )
        except serial.SerialException as exc:
            raise ConnectionError(f"打开串口 {port} 失败: {exc}") from exc

        self._port = port
        self._timeout = timeout

        # 使用 RLock 而非 Lock：
        # RLock 允许同一线程多次获取同一把锁（可重入）。
        # 例如：close() 中调用 stop_continuous()，两者都在
        # 各自的 with self._lock 里 —— 若是普通 Lock 会死锁。
        self._lock = threading.RLock()

        # ---- 修复 1：打开后立即清空缓冲 ----
        # 上次会话未读走的响应会留在缓冲区里，导致本次第一个
        # 命令读到错误的响应类型。这里先清一次。
        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except Exception:
            # 某些虚拟串口不支持 reset，忽略
            pass

    # ------------------------------------------------------------------
    # 属性 / 生命周期
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        """串口是否处于打开状态。"""
        return self._serial.is_open

    @property
    def port(self) -> str:
        """串口名称。"""
        return self._port

    def close(self) -> None:
        """关闭串口（幂等操作）。"""
        with self._lock:
            if self._serial.is_open:
                self._serial.close()

    def reset_input_buffer(self) -> None:
        """清空输入缓冲区（丢弃所有未读字节）。

        用于：

        - 打开串口后清理残留数据
        - 命令响应类型错误后清理脏数据
        - 连续模式停止后清理残留帧
        """
        with self._lock:
            try:
                self._serial.reset_input_buffer()
            except Exception:
                pass

    # 上下文管理器支持：with SerialTransport(...) as t: ...
    def __enter__(self) -> "SerialTransport":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 底层 IO
    # ------------------------------------------------------------------
    def write(self, data: bytes) -> None:
        """向串口写入字节并等待发送完成。

        :param data: 要发送的字节序列
        :raises ConnectionError: 串口未打开或写入失败
        """
        with self._lock:
            if not self._serial.is_open:
                raise ConnectionError("串口未打开")
            try:
                self._serial.write(data)
                # flush 确保数据全部发出（对于 USB 转串口尤其重要）
                self._serial.flush()
            except serial.SerialException as exc:
                raise ConnectionError(f"串口写入失败: {exc}") from exc

    def _read_exact(self, n: int, deadline: float) -> bytes:
        """精确读取 ``n`` 个字节；超过 deadline 则抛 TimeoutError。

        ============================================================
        动态调整 pyserial timeout 的原因
        ============================================================

        pyserial 的 ``timeout`` 是**单次 read 的超时**，不能直接
        表达 deadline。例如期望读 100 字节，read 返回 50 字节后
        再次 read 又会重新计时 timeout。

        为了让整体耗时不超过 deadline，我们在每次 read 前
        动态把 ``self._serial.timeout`` 设为剩余时间。

        ============================================================
        修复 5：空 chunk 时微休眠
        ============================================================

        read 超时时返回空 bytes，若立即重试会紧密循环消耗 CPU。
        加入 0.1ms 睡眠，几乎不影响读取速度但能让出 CPU。

        :param n:        需要读取的字节数
        :param deadline: 绝对截止时间（time.time() 度量）
        :return:         长度为 n 的字节串
        """
        buf = bytearray()
        while len(buf) < n:
            # 计算剩余时间
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"读取超时: 期望 {n} 字节, 已读到 {len(buf)} 字节"
                )

            # 动态设置本次 read 的超时
            self._serial.timeout = remaining

            try:
                chunk = self._serial.read(n - len(buf))
            except serial.SerialException as exc:
                raise ConnectionError(f"串口读取失败: {exc}") from exc

            if chunk:
                buf += chunk
            else:
                # ---- 修复 5 ----
                # 空 chunk：让出 CPU 0.1ms，避免紧密循环
                time.sleep(0.0001)

        return bytes(buf)

    # ------------------------------------------------------------------
    # 帧同步（核心修复）
    # ------------------------------------------------------------------
    def _read_frame_header(self, deadline: float) -> tuple[bytes, int]:
        """寻找一个**合法帧头**并读取总包长。

        ============================================================
        5 字节滑动窗口算法
        ============================================================

        窗口为 ``[b0, b1, b2, b3, b4]``。判定合法性：

        - ``b0 == 0xCC and b1 == 0x81`` （帧头正确）
        - ``MIN_FRAME_LEN <= int(b2:b5) <= MAX_FRAME_LEN``（长度合理）

        若合法 → 返回 ``(5 字节头, 总长)``；否则窗口右移 1 字节
        再读入 1 字节，重复上述检查。

        ============================================================
        为什么这样能解决 Bug #1
        ============================================================

        光谱数据里偶然出现 ``CC 81`` 时，后面 3 字节大概率是
        随机数据，其小端值落在 9~20000 之间的概率约 0.03%。
        再叠加 5 次重试的容错，实际误判概率几乎为 0。

        :param deadline: 绝对截止时间
        :return:         ``(5 字节头部, 总包长)``
        :raises TimeoutError: 在 deadline 前未找到合法帧头
        """
        # 先填满 5 字节窗口
        window = bytearray()
        while len(window) < 5:
            if time.time() > deadline:
                raise TimeoutError("等待帧头超时")
            window.append(self._read_exact(1, deadline)[0])

        # 滑动检查
        while True:
            # 检查帧头
            if window[0] == 0xCC and window[1] == 0x81:
                total_len = int.from_bytes(window[2:5], "little")
                # 长度合理性
                if MIN_FRAME_LEN <= total_len <= self.MAX_FRAME_LEN:
                    return bytes(window), total_len

            # 不合法：右移一位
            if time.time() > deadline:
                raise TimeoutError("等待帧头超时")
            del window[0]
            window.append(self._read_exact(1, deadline)[0])

    def _read_one_frame(self, deadline: float) -> tuple[int, bytes]:
        """单次尝试读取一帧（不做重试）。

        :param deadline: 绝对截止时间
        :return:         ``(命令类型, 数据段)``
        """
        # ---- 1. 寻找合法帧头 + 读总长 ----
        header, total_len = self._read_frame_header(deadline)

        # ---- 2. 读剩余字节 ----
        # total_len - len(header) = total_len - 5
        rest = self._read_exact(total_len - len(header), deadline)
        frame = header + rest

        # ---- 3. 交给协议层解析 ----
        return parse_response_frame(frame)

    def read_frame(self, timeout: float | None = None) -> tuple[int, bytes]:
        """读取一个完整的返回帧并解析。

        ============================================================
        重试机制
        ============================================================

        单帧解析失败（帧结构不合法 / 校验和不通过）时，
        不立即向上抛异常，而是**跳过这次**，继续尝试读取
        下一帧。最多尝试 ``MAX_SYNC_RETRIES`` 次。

        这样即使偶尔出现残留脏帧，也能被自动吸收。

        :param timeout: 超时秒数；None 表示使用构造时的默认值
        :return:        ``(命令类型, 返回数据)``
        :raises TimeoutError:  等待超时
        :raises ProtocolError: 重试次数耗尽仍无法解析
        :raises ChecksumError: 校验和不通过（已重试后）
        """
        with self._lock:
            # 计算绝对截止时间
            deadline = time.time() + (
                timeout if timeout is not None else self._timeout
            )

            last_exc: Exception | None = None

            for _ in range(self.MAX_SYNC_RETRIES):
                try:
                    return self._read_one_frame(deadline)
                except (ProtocolError, ChecksumError) as exc:
                    # 记录异常，跳过这个"伪帧头"，继续找下一个
                    last_exc = exc
                    continue

            # 重试用尽
            if last_exc is not None:
                raise ProtocolError(
                    f"连续 {self.MAX_SYNC_RETRIES} 次同步失败，"
                    f"最后一次错误: {last_exc}"
                )
            raise TimeoutError("读取帧超时（重试耗尽）")