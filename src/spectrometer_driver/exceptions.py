"""驱动包的自定义异常类型。

============================================================
设计原则
============================================================

1. **单一根异常**：所有异常继承 :class:`SpectrometerError`，
   调用方可以用 ``except SpectrometerError`` 一网打尽。

2. **语义细分**：按"连接 / 协议 / 校验 / 超时 / 设备"五类细分，
   便于上层根据场景做差异化处理（重试、报警、复位设备等）。

3. **携带上下文**：异常对象尽量携带足够信息（错误码、期望值、
   实际值等），方便日志与调试。

============================================================
捕获策略示例
============================================================

只关心"是否发生驱动错误"::

    try:
        data = spec.get_spectrum()
    except SpectrometerError:
        log("驱动出错")

按类型差异化处理::

    try:
        data = spec.get_spectrum()
    except TimeoutError:
        retry()
    except ChecksumError:
        log("通信质量差，检查串口线")
    except DeviceError as e:
        log(f"设备拒绝: 0x{e.code:02X}")
"""


class SpectrometerError(Exception):
    """所有驱动异常的基类。

    若上层只关心"是否发生了驱动错误"，捕获本类即可。
    """


class ConnectionError(SpectrometerError):
    """串口相关错误。

    涵盖以下场景：

    - 串口打开失败（端口不存在、被占用、权限不足）
    - 串口在通信过程中断开
    - 写入失败

    注意：这与 Python 内建的 ``ConnectionError`` 同名，
    但用途完全不同。在本包中请从 ``spectrometer_driver`` 导入。
    """


class ProtocolError(SpectrometerError):
    """协议层错误：帧结构不合法。

    包括：

    - 包头不正确（不是 ``0xCC 0x81``）
    - 声明的总包长与实际长度不一致
    - 包尾不正确（不是 ``0x0D 0x0A``）
    - 响应类型与期望不符
    - 重试后仍未匹配到期望响应
    """


class ChecksumError(ProtocolError):
    """校验和不匹配。

    继承自 :class:`ProtocolError`，因为校验失败本质上也是
    协议错误。单独设类是为了让上层在"只关心校验问题"时
    精准捕获（例如：校验失败常意味着串口线质量差）。

    常见原因：

    - 波特率不匹配
    - 串口线质量差导致字节损坏
    - 缓冲区中混入了脏数据
    """


class TimeoutError(SpectrometerError):
    """等待设备响应超时。

    通常意味着：

    - 设备未上电或未连接
    - 命令不被设备支持（设备不响应）
    - 曝光时间过长导致采集延迟超过 timeout
    """


class DeviceError(SpectrometerError):
    """设备返回了失败状态码。

    例如设置曝光时间时设备返回 ``0x15``，表明参数越界或
    设备拒绝执行。

    Attributes
    ----------
    code : int
        设备返回的原始错误码。

    Examples
    --------
    ::

        try:
            spec.set_exposure_time(10**9)
        except DeviceError as e:
            print(f"设备拒绝: 0x{e.code:02X}")
    """

    def __init__(self, code: int, message: str = "") -> None:
        """构造异常。

        :param code:    设备返回的状态码
        :param message: 额外的描述信息（例如"设置曝光时间失败"）
        """
        self.code = code
        # 格式化输出：同时包含错误码和附加信息
        text = f"设备返回错误码 0x{code:02X}"
        if message:
            text += f" ({message})"
        super().__init__(text)