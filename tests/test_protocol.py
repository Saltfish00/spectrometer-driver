"""协议层单元测试。

验证帧构建、解析、校验和逻辑的正确性。
"""

import pytest

from spectrometer_driver.protocol import (
    CMD_HEADER,
    FRAME_END,
    RESP_HEADER,
    Command,
    build_command_frame,
    calc_checksum,
    parse_response_frame,
)
from spectrometer_driver.exceptions import ChecksumError, ProtocolError


# ============================================================================
# 校验和
# ============================================================================

def test_checksum_simple():
    """校验和 = 字节求和取低 8 位。"""
    assert calc_checksum(b"") == 0
    assert calc_checksum(b"\x01") == 1
    # 0xFF + 0x02 = 0x101 → 0x01
    assert calc_checksum(b"\xff\x02") == 0x01


# ============================================================================
# 帧构建
# ============================================================================

def test_build_command_frame_no_data():
    """验证单帧采集命名帧的字节结构。"""
    frame = build_command_frame(Command.SINGLE_ACQUISITION)
    # 头 + 长(3) + 类型 + 校验 + 尾
    assert frame[:2] == CMD_HEADER
    assert frame[2:5] == b"\x09\x00\x00"    # 总包长 = 9
    assert frame[5] == 0x02                 # 命令类型
    assert frame[-2:] == FRAME_END
    # 校验位 = 之前所有字节之和 & 0xFF
    assert frame[6] == calc_checksum(frame[:-3])


def test_build_command_frame_with_data():
    """验证带数据的命令帧。"""
    payload = (100_000).to_bytes(4, "little")
    frame = build_command_frame(Command.SET_EXPOSURE, payload)
    assert frame[:2] == CMD_HEADER
    assert frame[5] == Command.SET_EXPOSURE
    # 总包长 = 9 + 4 = 13
    assert int.from_bytes(frame[2:5], "little") == 13


def test_build_command_frame_invalid_type():
    """命令类型越界应抛 ValueError。"""
    with pytest.raises(ValueError):
        build_command_frame(0x100)


# ============================================================================
# 帧解析
# ============================================================================

def _make_response(cmd_type: int, payload: bytes) -> bytes:
    """辅助：构造合法返回帧用于测试。"""
    total_len = 9 + len(payload)
    frame = bytearray()
    frame += RESP_HEADER
    frame += total_len.to_bytes(3, "little")
    frame.append(cmd_type)
    frame += payload
    frame.append(calc_checksum(frame))
    frame += FRAME_END
    return bytes(frame)


def test_parse_response_frame_ok():
    """正常解析。"""
    payload = b"\x00\x10\x27\x00\x00\x02"
    frame = _make_response(Command.GET_EXPOSURE, payload)
    cmd_type, data = parse_response_frame(frame)
    assert cmd_type == Command.GET_EXPOSURE
    assert data == payload


def test_parse_response_frame_bad_header():
    """包头错误应抛 ProtocolError。"""
    frame = _make_response(Command.GET_EXPOSURE, b"\x00")
    bad = b"\xCC\x82" + frame[2:]
    with pytest.raises(ProtocolError):
        parse_response_frame(bad)


def test_parse_response_frame_bad_checksum():
    """校验错误应抛 ChecksumError。"""
    frame = bytearray(_make_response(Command.GET_EXPOSURE, b"\x00"))
    frame[-3] ^= 0xFF
    with pytest.raises(ChecksumError):
        parse_response_frame(bytes(frame))


def test_parse_response_frame_bad_tail():
    """包尾错误应抛 ProtocolError。"""
    frame = bytearray(_make_response(Command.GET_EXPOSURE, b"\x00"))
    frame[-1] = 0xFF
    with pytest.raises(ProtocolError):
        parse_response_frame(bytes(frame))


def test_parse_response_frame_extra_bytes_ignored():
    """比声明更长时，多余字节应被忽略。"""
    frame = _make_response(Command.GET_EXPOSURE, b"\x00")
    frame = frame + b"\x00\x00\x00"    # 追加垃圾
    cmd_type, data = parse_response_frame(frame)
    assert cmd_type == Command.GET_EXPOSURE