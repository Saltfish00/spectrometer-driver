"""协议层：帧的构建与解析，与具体传输方式完全解耦。

============================================================
协议要点（TLM 手册 V1.2.1）
============================================================

**命名包（主机 → 设备）**::

    偏移   字节数   含义
    ----   ------   -----------------------------------------
    0      2        包头固定为 0xCC 0x01
    2      3        总包长（小端 uint24），含整帧字节数
    5      1        命令类型
    6      N        命名数据
    N+6    1        校验位 = 之前所有字节之和 & 0xFF
    N+7    2        包尾固定为 0x0D 0x0A

**返回包（设备 → 主机）**::

    结构与命名包一致，仅包头为 0xCC 0x81。

============================================================
设计要点
============================================================

- 本模块只做**字节序列的编解码**，不涉及串口 IO，
  因此可以脱离硬件进行单元测试。
- 所有多字节数值均采用**小端序**（低位在前）。
- 校验位规则简单：累加之前所有字节并取低 8 位。
"""

from __future__ import annotations

from enum import IntEnum

from .exceptions import ChecksumError, ProtocolError

# ============================================================================
# 协议常量
# ============================================================================

#: 命名包（主机 → 设备）的包头
#: 两个固定字节，所有主机下发的命令都必须以此开头
CMD_HEADER: bytes = b"\xCC\x01"

#: 返回包（设备 → 主机）的包头
#: 与命名包区分：0x81 vs 0x01
RESP_HEADER: bytes = b"\xCC\x81"

#: 帧尾
#: 固定为 0x0D 0x0A（即 CR LF，回车换行）
FRAME_END: bytes = b"\x0D\x0A"

#: 除数据外的固定开销
#: 头(2) + 长(3) + 类型(1) + 校验(1) + 尾(2) = 9 字节
FRAME_OVERHEAD: int = 9

#: 一帧最少字节数（无数据时）
#: 与 FRAME_OVERHEAD 数值相同，但语义不同：这个用于长度校验
MIN_FRAME_LEN: int = 9

#: 总包长字段占用的字节数
LENGTH_FIELD_SIZE: int = 3


# ============================================================================
# 命令类型枚举
# ============================================================================

class Command(IntEnum):
    """协议命令类型枚举。

    使用 ``IntEnum`` 而非普通 ``Enum`` 的好处：

    - 可以像整数一样参与位运算与比较
    - 打印时仍显示有意义的名称
    - 可直接传入 ``build_command_frame(cmd_type=...)``
    """

    SINGLE_ACQUISITION = 0x02        #: 单帧获取光谱数据
    START_CONTINUOUS = 0x03          #: 开始连续获取光谱数据
    STOP_CONTINUOUS = 0x04           #: 停止获取光谱数据
    GET_DEVICE_INFO = 0x08           #: 获取设备信息（命名数据 = 0x18）
    SET_MANUAL_EXPOSURE_MODE = 0x0A  #: 设置手动曝光模式
    SET_AUTO_EXPOSURE_MODE = 0x0B    #: 设置自动曝光模式
    SET_EXPOSURE = 0x0C              #: 设置曝光值（uint32 微秒）
    GET_EXPOSURE = 0x0D              #: 获取曝光值
    GET_WAVELENGTH_RANGE = 0x0F      #: 获取光谱起止波长


# ============================================================================
# 校验和
# ============================================================================

def calc_checksum(data: bytes) -> int:
    """计算校验位：之前所有字节之和，取低 8 位。

    :param data: 校验位之前的全部字节
    :return:     0~255 之间的整数

    示例::

        >>> calc_checksum(b"")
        0
        >>> calc_checksum(b"\\x01")
        1
        >>> calc_checksum(b"\\xff\\x02")   # 0xFF + 0x02 = 0x101 → 0x01
        1
    """
    # sum() 累加所有字节值，& 0xFF 等价于 "% 256"
    return sum(data) & 0xFF


# ============================================================================
# 帧构建
# ============================================================================

def build_command_frame(cmd_type: int, data: bytes = b"") -> bytes:
    """构建一个命名帧（主机 → 设备）。

    :param cmd_type: 命令类型，取值参见 :class:`Command`
    :param data:     命令携带的数据（可为空）
    :return:         完整的字节序列，可直接写入串口
    :raises ValueError: 命令类型越界，或数据长度超出 3 字节表示范围

    示例::

        >>> build_command_frame(Command.SINGLE_ACQUISITION)
        b'\\xcc\\x01\\x09\\x00\\x00\\x02\\xd8\\r\\n'

        >>> # 设置 100ms 曝光
        >>> payload = (100_000).to_bytes(4, "little")
        >>> build_command_frame(Command.SET_EXPOSURE, payload)
    """
    # 命令类型必须是单字节（0~255）
    if not 0 <= cmd_type <= 0xFF:
        raise ValueError(f"命令类型必须在 0~255 之间，当前为 {cmd_type}")

    # 总包长 = 固定开销 + 数据长度
    # 3 字节小端可表示的最大值 = 0xFFFFFF
    total_len = FRAME_OVERHEAD + len(data)
    if total_len > 0xFFFFFF:
        raise ValueError(f"数据过长，总包长 {total_len} 超出 3 字节表示范围")

    # ---- 逐段拼装 ----
    frame = bytearray()
    frame += CMD_HEADER                                    # 包头（2B）
    frame += total_len.to_bytes(LENGTH_FIELD_SIZE, "little")  # 总长（3B，LE）
    frame.append(cmd_type)                                 # 类型（1B）
    frame += data                                          # 数据（NB）
    # 校验：此时 frame 内不含校验位和包尾
    frame.append(calc_checksum(frame))
    frame += FRAME_END                                     # 包尾（2B）

    return bytes(frame)


# ============================================================================
# 帧解析
# ============================================================================

def parse_response_frame(frame: bytes) -> tuple[int, bytes]:
    """解析一个完整的返回帧。

    ============================================================
    解析步骤
    ============================================================

    1. 检查长度是否达到最小值
    2. 检查包头是否为 ``0xCC 0x81``
    3. 检查声明的总包长是否合理
    4. 检查包尾是否为 ``0x0D 0x0A``
    5. 计算并验证校验和
    6. 提取命令类型与数据段

    ============================================================
    修复点
    ============================================================

    长度检查放宽——若输入比声明的总长更长，
    只解析前 total_len 字节，忽略尾部多余数据。
    这样既能兼容传输层"多读了几个字节"的场景，
    也方便单元测试构造数据。

    :param frame: 从串口读到的完整帧字节
    :return:      ``(命令类型, 返回数据)``
    :raises ProtocolError: 帧结构不合法
    :raises ChecksumError: 校验和不通过
    """
    # ---- 1. 长度检查 ----
    if len(frame) < MIN_FRAME_LEN:
        raise ProtocolError(f"帧长度过短: {len(frame)} 字节（最少 {MIN_FRAME_LEN}）")

    # ---- 2. 包头检查 ----
    if frame[:2] != RESP_HEADER:
        raise ProtocolError(f"非法的返回帧包头: {frame[:2].hex(' ')}")

    # ---- 3. 总包长检查 ----
    total_len = int.from_bytes(frame[2:5], "little")
    if total_len < MIN_FRAME_LEN:
        raise ProtocolError(f"总包长过小: {total_len}")
    if len(frame) < total_len:
        raise ProtocolError(
            f"帧长度不足: 声明 {total_len} 字节, 实际 {len(frame)} 字节"
        )
    # 修复：允许输入比声明的更长，多余部分直接忽略
    if len(frame) > total_len:
        frame = frame[:total_len]

    # ---- 4. 包尾检查 ----
    if frame[-2:] != FRAME_END:
        raise ProtocolError(f"非法的帧尾: {frame[-2:].hex(' ')}")

    # ---- 5. 校验和检查 ----
    # body = 去掉校验位(1) + 尾(2) 后的全部字节
    body = frame[:-3]
    received_checksum = frame[-3]
    expected_checksum = calc_checksum(body)
    if received_checksum != expected_checksum:
        raise ChecksumError(
            f"校验和不匹配: 收到 0x{received_checksum:02X}, "
            f"计算得到 0x{expected_checksum:02X}"
        )

    # ---- 6. 提取命令类型与数据 ----
    cmd_type = frame[5]                     # 第 6 字节（索引 5）是命令类型
    payload = frame[6:-3]                   # 索引 6 到 -3（不含校验位与包尾）

    return cmd_type, payload