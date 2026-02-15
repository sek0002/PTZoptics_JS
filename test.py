#!/usr/bin/env python3
"""
Virtual VISCA over IP "camera" (UDP/52381).

Implements:
- VISCA-over-IP framing (8-byte header):
  - bytes 0-1: payload type
  - bytes 2-3: payload length (1..16 typically; many devices use <=16)
  - bytes 4-7: sequence number
- Payload types (common):
  - 0x0100: VISCA command
  - 0x0110: VISCA inquiry
  - 0x0111: VISCA reply
  - 0x0200: control command
  - 0x0201: control reply

This is a control-plane emulator only (no video streaming).

Tested design goal:
- Be useful for controller integration tests, simulators, and CI.

Run:
  python3 virtual_visca_camera.py --bind 0.0.0.0 --port 52381 --verbose

Send example (linux netcat):
  printf '\x01\x00\x00\x06\x00\x00\x00\x01\x81\x01\x04\x00\x02\xFF' | nc -u -w1 127.0.0.1 52381

That example is "Power On" for many VISCA cameras (payload length 6).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import socket
import struct
import time
from typing import Dict, Optional, Tuple


# -------- VISCA-over-IP constants --------
PT_VISCA_COMMAND = 0x0100
PT_VISCA_INQUIRY = 0x0110
PT_VISCA_REPLY   = 0x0111
PT_CONTROL_CMD   = 0x0200
PT_CONTROL_REPLY = 0x0201

DEFAULT_PORT = 52381


# -------- Embedded PTZOptics VISCA command catalog --------
# (CommandName, ActionOrMode) -> CommandHex
#
# This removes any runtime dependency on CSV/pandas. You can run this script in
# "send" mode to emit VISCA-over-IP packets, or in "serve" mode (default) to
# emulate a camera.
COMMANDS: Dict[Tuple[str, str], str] = {
    ('ACK', 'ACK'): '90 4y FF',
    ('BGain', 'Direct'): '81 01 04 44 00 00 0p 0q FF',
    ('BGain', 'Down'): '81 01 04 04 03 FF',
    ('BGain', 'Reset'): '81 01 04 04 00 FF',
    ('BGain', 'Up'): '81 01 04 04 02 FF',
    ('Completion', 'Completion'): '90 5y FF',
    ('Error', 'Buffer Full'): '90 60 03 FF',
    ('Error', 'Command Canceled'): '90 6y 04 FF',
    ('Error', 'No Socket'): '90 6y 05 FF',
    ('Error', 'Not Executable'): '90 6y 41 FF',
    ('Error', 'Syntax Error'): '90 60 02 FF',
    ('Exposure', 'Bright'): '81 01 04 39 0D FF',
    ('Exposure', 'Full Auto'): '81 01 04 39 00 FF',
    ('Exposure', 'Iris Priority'): '81 01 04 39 0B FF',
    ('Exposure', 'Manual'): '81 01 04 39 03 FF',
    ('Exposure', 'Shutter Priority'): '81 01 04 39 0A FF',
    ('Focus', 'Auto'): '81 01 04 38 02 FF',
    ('Focus', 'Auto/Manual Toggle'): '81 01 04 38 10 FF',
    ('Focus', 'Direct'): '81 01 04 48 0p 0q 0r 0s FF',
    ('Focus', 'Far Std'): '81 01 04 08 02 FF',
    ('Focus', 'Far Var'): '81 01 04 08 2p FF',
    ('Focus', 'Manual'): '81 01 04 38 03 FF',
    ('Focus', 'Near Std'): '81 01 04 08 03 FF',
    ('Focus', 'Near Var'): '81 01 04 08 3p FF',
    ('Focus', 'One Push Trigger'): '81 01 04 18 01 FF',
    ('Gain', 'Direct'): '81 01 04 4C 00 00 0p 0q FF',
    ('Gain', 'Down'): '81 01 04 0C 03 FF',
    ('Gain', 'Reset'): '81 01 04 0C 00 FF',
    ('Gain', 'Up'): '81 01 04 0C 02 FF',
    ('Inquiry', 'BGain'): '81 09 04 44 FF',
    ('Inquiry', 'Exposure'): '81 09 04 39 FF',
    ('Inquiry', 'Focus'): '81 09 04 38 FF',
    ('Inquiry', 'Focus Mode'): '81 09 04 38 FF',
    ('Inquiry', 'Gain'): '81 09 04 4C FF',
    ('Inquiry', 'Iris'): '81 09 04 4B FF',
    ('Inquiry', 'Power'): '81 09 04 00 FF',
    ('Inquiry', 'RGain'): '81 09 04 43 FF',
    ('Inquiry', 'Shutter'): '81 09 04 4A FF',
    ('Inquiry', 'WB Mode'): '81 09 04 35 FF',
    ('Inquiry', 'Zoom'): '81 09 04 47 FF',
    ('Iris', 'Direct'): '81 01 04 4B 00 00 0p 0q FF',
    ('Iris', 'Down'): '81 01 04 0B 03 FF',
    ('Iris', 'Reset'): '81 01 04 0B 00 FF',
    ('Iris', 'Up'): '81 01 04 0B 02 FF',
    ('Power', 'Off'): '81 01 04 00 03 FF',
    ('Power', 'On'): '81 01 04 00 02 FF',
    ('RGain', 'Direct'): '81 01 04 43 00 00 0p 0q FF',
    ('RGain', 'Down'): '81 01 04 03 03 FF',
    ('RGain', 'Reset'): '81 01 04 03 00 FF',
    ('RGain', 'Up'): '81 01 04 03 02 FF',
    ('Shutter', 'Direct'): '81 01 04 4A 00 00 0p 0q FF',
    ('Shutter', 'Down'): '81 01 04 0A 03 FF',
    ('Shutter', 'Reset'): '81 01 04 0A 00 FF',
    ('Shutter', 'Up'): '81 01 04 0A 02 FF',
    ('WB', 'Auto'): '81 01 04 35 00 FF',
    ('WB', 'Indoor'): '81 01 04 35 01 FF',
    ('WB', 'Manual'): '81 01 04 35 05 FF',
    ('WB', 'One Push'): '81 01 04 35 03 FF',
    ('WB', 'Outdoor'): '81 01 04 35 02 FF',
    ('WB', 'One Push Trigger'): '81 01 04 10 05 FF',
    ('Zoom', 'Direct'): '81 01 04 47 0p 0q 0r 0s FF',
    ('Zoom', 'Tele Std'): '81 01 04 07 02 FF',
    ('Zoom', 'Tele Var'): '81 01 04 07 2p FF',
    ('Zoom', 'Wide Std'): '81 01 04 07 03 FF',
    ('Zoom', 'Wide Var'): '81 01 04 07 3p FF',
    ('Zoom', 'Stop'): '81 01 04 07 00 FF',
    ('Pan_TiltDrive','UpLeft'): '81 01 06 01 VV WW 01 01 FF',
}


# -------- Helpers --------
def hexdump(b: bytes) -> str:
    return " ".join(f"{x:02X}" for x in b)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# -------- VISCA command templating / sender --------
def _parse_nibbles(nibble_args) -> Dict[str, int]:
    """Parse nibble assignments like ['p=10','q=2'] into {'p':10,'q':2}."""
    nibbles: Dict[str, int] = {}
    for item in nibble_args or []:
        if "=" not in item:
            raise SystemExit(f"Bad --nibble '{item}', expected like p=10")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(k) != 1 or not k.isalpha():
            raise SystemExit(f"Nibble key must be a single letter, got '{k}'")
        val = int(v, 0)
        if not (0 <= val <= 0xF):
            raise SystemExit(f"Nibble '{k}' out of range (0..15): {val}")
        nibbles[k] = val
    return nibbles


def _token_to_byte(tok: str, nibbles: Dict[str, int]) -> int:
    tok = tok.strip()
    if not tok:
        raise ValueError("empty token")

    # pure byte like "81" or "FF"
    if re.fullmatch(r"[0-9A-Fa-f]{2}", tok):
        return int(tok, 16)

    # pattern like 0p, 2p, 4y, 6y => high nibble fixed, low nibble variable
    m = re.fullmatch(r"([0-9A-Fa-f])([A-Za-z])", tok)
    if m:
        hi = int(m.group(1), 16)
        var = m.group(2)
        if var not in nibbles:
            raise ValueError(f"Missing nibble value for '{var}' (needed for token '{tok}')")
        return (hi << 4) | (nibbles[var] & 0xF)

    raise ValueError(f"Unrecognized token '{tok}'")


def hex_template_to_bytes(template: str, nibbles: Dict[str, int]) -> bytes:
    parts = template.strip().split()
    out = bytearray()
    for p in parts:
        out.append(_token_to_byte(p, nibbles))
    return bytes(out)


def list_embedded_commands() -> None:
    pairs = sorted(COMMANDS.keys(), key=lambda x: (x[0].casefold(), x[1].casefold()))
    for name, mode in pairs:
        print(f"{name:12s}  {mode}")


def send_visca_from_catalog(
    host: str,
    port: int,
    name: str,
    mode: str,
    *,
    seq: int = 1,
    inquiry: bool = False,
    nibbles: Optional[Dict[str, int]] = None,
    timeout_s: float = 0.5,
    expect_reply: bool = True,
    verbose: bool = False,
) -> int:
    """Send one embedded command; returns number of replies received."""
    key = (name.strip(), mode.strip())
    if key not in COMMANDS:
        name_cf = name.casefold()
        suggestions = [k for k in COMMANDS.keys() if name_cf in k[0].casefold()]
        msg = f"No exact match for {key}.\n"
        if suggestions:
            msg += "Close matches:\n" + "\n".join([f"  {k[0]} / {k[1]}" for k in suggestions[:20]])
        raise SystemExit(msg)

    nibbles = nibbles or {}
    payload = hex_template_to_bytes(COMMANDS[key], nibbles)
    payload_type = PT_VISCA_INQUIRY if inquiry else PT_VISCA_COMMAND
    pkt = build_voip_packet(payload_type, seq, payload)

    if verbose:
        print(f"Selected: {key[0]} / {key[1]}")
        print(f"VISCA payload: {hexdump(payload)}")
        print(f"VOIP packet : {hexdump(pkt)}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout_s)
        sock.sendto(pkt, (host, port))

        if not expect_reply:
            return 0

        replies = 0
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            replies += 1
            if verbose:
                print(f"RX from {addr}: {hexdump(data)}")
        return replies
    finally:
        sock.close()


# -------- Virtual camera state --------
@dataclasses.dataclass
class CameraState:
    power_on: bool = True
    pan: int = 0        # arbitrary units
    tilt: int = 0       # arbitrary units
    zoom: int = 0       # 0..0x4000-ish typical; we clamp to 0..16384 here
    focus: int = 0      # 0..16384
    presets: Dict[int, Tuple[int, int, int, int, bool]] = dataclasses.field(default_factory=dict)

    def store_preset(self, n: int) -> None:
        self.presets[n] = (self.pan, self.tilt, self.zoom, self.focus, self.power_on)

    def recall_preset(self, n: int) -> bool:
        if n not in self.presets:
            return False
        self.pan, self.tilt, self.zoom, self.focus, self.power_on = self.presets[n]
        return True


# -------- VISCA message building --------
def visca_ack(socket_no: int = 1, device_addr: int = 0x90) -> bytes:
    # Many VISCA docs describe ACK as: y0 4z FF (z=socket)
    # We'll use device_addr 0x90 (address 1 in VISCA-over-IP contexts) and socket 1.
    return bytes([device_addr, 0x40 | (socket_no & 0x0F), 0xFF])

def visca_completion(socket_no: int = 1, device_addr: int = 0x90) -> bytes:
    # Completion: y0 5z FF
    return bytes([device_addr, 0x50 | (socket_no & 0x0F), 0xFF])

def visca_error(err: int = 0x02, socket_no: int = 1, device_addr: int = 0x90) -> bytes:
    # Common pattern: y0 60 0e FF (varies by implementation).
    # We'll implement a generic "syntax error" style response:
    # y0 60 02 FF  (02 often used as "syntax error" / "invalid parameter" in vendor docs)
    return bytes([device_addr, 0x60 | (socket_no & 0x0F), err & 0xFF, 0xFF])

def visca_inquiry_reply(data: bytes, device_addr: int = 0x90) -> bytes:
    # Many inquiry replies use: y0 50 ... FF (note: not completion notice)
    # We'll send: y0 50 <data...> FF
    return bytes([device_addr, 0x50]) + data + bytes([0xFF])


# -------- VISCA-over-IP framing --------
def parse_voip_packet(pkt: bytes) -> Tuple[int, int, int, bytes]:
    if len(pkt) < 8:
        raise ValueError("packet too short for VISCA-over-IP header")
    payload_type, payload_len = struct.unpack(">HH", pkt[0:4])   # big-endian
    seq = struct.unpack(">I", pkt[4:8])[0]
    payload = pkt[8:]
    if payload_len != len(payload):
        # Some senders may pad; we enforce equality for clean simulation.
        raise ValueError(f"payload_len header={payload_len} != actual={len(payload)}")
    return payload_type, payload_len, seq, payload

def build_voip_packet(payload_type: int, seq: int, payload: bytes) -> bytes:
    hdr = struct.pack(">HHI", payload_type & 0xFFFF, len(payload) & 0xFFFF, seq & 0xFFFFFFFF)
    return hdr + payload


# -------- VISCA parsing (minimal) --------
class ViscaEmulator:
    """
    Minimal VISCA parser sufficient for integration tests:
    - Power:      81 01 04 00 02 FF (On), 81 01 04 00 03 FF (Off)
    - Pan/Tilt:   81 01 06 01 VV WW 03 DD FF  (drive; we treat DD as direction)
    - Stop:       81 01 06 01 VV WW 03 03 FF
    - Zoom:       81 01 04 07 2p FF (tele), 81 01 04 07 3p FF (wide), 00 stop
    - Focus:      81 01 04 08 2p/3p/00 FF
    - Preset set: 81 01 04 3F 01 nn FF
    - Preset call:81 01 04 3F 02 nn FF

    Inquiries:
    - PowerInq:   81 09 04 00 FF  -> returns 90 50 02 FF (on) or 90 50 03 FF (off)
    - ZoomPosInq: 81 09 04 47 FF  -> returns 90 50 0p 0q 0r 0s FF (nibbles)
    - FocusPosInq:81 09 04 48 FF  -> returns 90 50 0p 0q 0r 0s FF
    """
    def __init__(self, state: CameraState, verbose: bool = False):
        self.state = state
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _is_visca(msg: bytes) -> bool:
        # VISCA messages typically end with 0xFF
        return len(msg) >= 2 and msg[-1] == 0xFF

    @staticmethod
    def _nibbles_16bit(v: int) -> bytes:
        # VISCA position replies often use 4 nibbles as bytes: 0p 0q 0r 0s
        v = clamp(v, 0, 0xFFFF)
        return bytes([
            (v >> 12) & 0x0F,
            (v >> 8) & 0x0F,
            (v >> 4) & 0x0F,
            (v >> 0) & 0x0F,
        ])

    def handle_visca(self, payload_type: int, payload: bytes) -> Tuple[bytes, ...]:
        """
        Returns a tuple of VISCA payloads to send back (each will be wrapped as VOIP reply type).
        Usually: ACK then Completion, or inquiry reply (no ACK/Completion in many implementations).
        """
        if not self._is_visca(payload):
            return (visca_error(),)

        # Basic framing: [dest][...][FF]
        # We only implement "camera 1" style addressing and ignore dest variations.
        # Common command prefix: 81 01 ...
        if len(payload) < 4:
            return (visca_error(),)

        # Normalize: accept dest 0x81 and ignore it
        dest = payload[0]
        if dest not in (0x81, 0x88, 0x80, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F):
            # Still try to parse; many controllers always use 0x81
            pass

        # Inquiry messages often start with 81 09 ...
        if payload_type == PT_VISCA_INQUIRY or (len(payload) >= 2 and payload[1] == 0x09):
            return self._handle_inquiry(payload)

        # Commands usually: 81 01 ...
        if len(payload) >= 2 and payload[1] == 0x01:
            return self._handle_command(payload)

        return (visca_error(),)

    def _handle_command(self, cmd: bytes) -> Tuple[bytes, ...]:
        # Patterns:
        # Power: 81 01 04 00 02/03 FF
        if cmd[:4] == bytes([0x81, 0x01, 0x04, 0x00]) and len(cmd) == 6 and cmd[-1] == 0xFF:
            val = cmd[4]
            if val == 0x02:
                self.state.power_on = True
                self._log("Power -> ON")
                return (visca_ack(), visca_completion())
            if val == 0x03:
                self.state.power_on = False
                self._log("Power -> OFF")
                return (visca_ack(), visca_completion())
            return (visca_error(),)

        # Zoom: 81 01 04 07 xx FF
        if cmd[:4] == bytes([0x81, 0x01, 0x04, 0x07]) and len(cmd) == 6:
            mode = cmd[4]
            # 0x2p tele, 0x3p wide, 0x00 stop
            if mode == 0x00:
                self._log("Zoom -> STOP")
            elif (mode & 0xF0) == 0x20:
                speed = mode & 0x0F
                self.state.zoom = clamp(self.state.zoom + (50 * max(1, speed)), 0, 16384)
                self._log(f"Zoom -> TELE speed={speed} zoom={self.state.zoom}")
            elif (mode & 0xF0) == 0x30:
                speed = mode & 0x0F
                self.state.zoom = clamp(self.state.zoom - (50 * max(1, speed)), 0, 16384)
                self._log(f"Zoom -> WIDE speed={speed} zoom={self.state.zoom}")
            else:
                return (visca_error(),)
            return (visca_ack(), visca_completion())

        # Focus: 81 01 04 08 xx FF
        if cmd[:4] == bytes([0x81, 0x01, 0x04, 0x08]) and len(cmd) == 6:
            mode = cmd[4]
            if mode == 0x00:
                self._log("Focus -> STOP")
            elif (mode & 0xF0) == 0x20:
                speed = mode & 0x0F
                self.state.focus = clamp(self.state.focus + (50 * max(1, speed)), 0, 16384)
                self._log(f"Focus -> FAR speed={speed} focus={self.state.focus}")
            elif (mode & 0xF0) == 0x30:
                speed = mode & 0x0F
                self.state.focus = clamp(self.state.focus - (50 * max(1, speed)), 0, 16384)
                self._log(f"Focus -> NEAR speed={speed} focus={self.state.focus}")
            else:
                return (visca_error(),)
            return (visca_ack(), visca_completion())

        # Pan/Tilt drive:
        # 81 01 06 01 VV WW 03 DD FF
        if len(cmd) == 9 and cmd[:4] == bytes([0x81, 0x01, 0x06, 0x01]) and cmd[6] == 0x03 and cmd[-1] == 0xFF:
            vv = cmd[4]
            ww = cmd[5]
            dd = cmd[7]
            # dd: 01 up, 02 down, 03 stop, 04 left, 05 up-left ... varies;
            # we implement a pragmatic subset:
            step_pan = max(1, vv)
            step_tilt = max(1, ww)

            if dd == 0x03:
                self._log("Pan/Tilt -> STOP")
            elif dd == 0x01:  # up
                self.state.tilt += step_tilt
                self._log(f"Tilt -> UP +{step_tilt} => {self.state.tilt}")
            elif dd == 0x02:  # down
                self.state.tilt -= step_tilt
                self._log(f"Tilt -> DOWN -{step_tilt} => {self.state.tilt}")
            elif dd == 0x04:  # left
                self.state.pan -= step_pan
                self._log(f"Pan -> LEFT -{step_pan} => {self.state.pan}")
            elif dd == 0x05:  # right (common on some models)
                self.state.pan += step_pan
                self._log(f"Pan -> RIGHT +{step_pan} => {self.state.pan}")
            else:
                self._log(f"Pan/Tilt -> unknown direction dd={dd:02X} (ignored)")
            return (visca_ack(), visca_completion())

        # Preset: 81 01 04 3F 01 nn FF (set),  ... 02 nn FF (recall)
        if len(cmd) == 7 and cmd[:4] == bytes([0x81, 0x01, 0x04, 0x3F]) and cmd[-1] == 0xFF:
            op = cmd[4]
            n = cmd[5]
            if op == 0x01:
                self.state.store_preset(n)
                self._log(f"Preset -> SET {n}")
                return (visca_ack(), visca_completion())
            if op == 0x02:
                ok = self.state.recall_preset(n)
                self._log(f"Preset -> RECALL {n} ok={ok}")
                return (visca_ack(), visca_completion() if ok else visca_error())
            return (visca_error(),)

        self._log(f"Unknown command: {hexdump(cmd)}")
        return (visca_error(),)

    def _handle_inquiry(self, inq: bytes) -> Tuple[bytes, ...]:
        # Power inquiry: 81 09 04 00 FF
        if inq == bytes([0x81, 0x09, 0x04, 0x00, 0xFF]):
            # Reply body commonly: 02 (on) or 03 (off)
            val = 0x02 if self.state.power_on else 0x03
            self._log(f"PowerInq -> {val:02X}")
            return (visca_inquiry_reply(bytes([val]), device_addr=0x90),)

        # Zoom position inquiry: 81 09 04 47 FF
        if inq == bytes([0x81, 0x09, 0x04, 0x47, 0xFF]):
            data = self._nibbles_16bit(self.state.zoom)
            self._log(f"ZoomPosInq -> zoom={self.state.zoom}")
            return (visca_inquiry_reply(data, device_addr=0x90),)

        # Focus position inquiry: 81 09 04 48 FF
        if inq == bytes([0x81, 0x09, 0x04, 0x48, 0xFF]):
            data = self._nibbles_16bit(self.state.focus)
            self._log(f"FocusPosInq -> focus={self.state.focus}")
            return (visca_inquiry_reply(data, device_addr=0x90),)

        self._log(f"Unknown inquiry: {hexdump(inq)}")
        return (visca_error(),)


# -------- UDP server --------
class ViscaUdpServer(asyncio.DatagramProtocol):
    def __init__(self, emulator: ViscaEmulator, verbose: bool = False):
        self.emulator = emulator
        self.verbose = verbose
        self._raw_seq = 1  # seq counter for unframed VISCA packets
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        if self.verbose:
            sockname = self.transport.get_extra_info("sockname")
            print(f"Listening on {sockname}")

    def datagram_received(self, data: bytes, addr) -> None:
        if self.verbose:
            print(f"\nRX from {addr}: {hexdump(data)}")

        # Some controller apps send *raw* VISCA bytes over UDP (no 8-byte VISCA-over-IP header).
        # Detect that case (starts with 0x8? and ends with 0xFF) and treat it as a VISCA command.
        if len(data) >= 3 and (data[0] & 0xF0) == 0x80 and data[-1] == 0xFF:
            payload_type = PT_VISCA_COMMAND
            payload = data
            seq = self._raw_seq
            self._raw_seq = (self._raw_seq + 1) & 0xFFFFFFFF
        else:
            try:
                payload_type, payload_len, seq, payload = parse_voip_packet(data)
            except Exception as e:
                if self.verbose:
                    print(f"Parse error: {e}")
                return

        # Handle control commands (e.g., RESET sequence number)
        if payload_type == PT_CONTROL_CMD:
            # Very small emulator: if payload starts with 0x01 => RESET
            # Reply with CONTROL_REPLY, same seq, payload 0x01.
            if len(payload) >= 1 and payload[0] == 0x01:
                reply = build_voip_packet(PT_CONTROL_REPLY, seq, bytes([0x01]))
                self._send(reply, addr)
                return
            reply = build_voip_packet(PT_CONTROL_REPLY, seq, bytes([0xFF]))
            self._send(reply, addr)
            return

        # VISCA command/inquiry handling
        visca_payloads = self.emulator.handle_visca(payload_type, payload)

        # Wrap each VISCA response as payload type "VISCA reply" and reuse seq
        for vp in visca_payloads:
            pkt = build_voip_packet(PT_VISCA_REPLY, seq, vp)
            self._send(pkt, addr)

    def _send(self, pkt: bytes, addr) -> None:
        if self.transport is None:
            return
        if self.verbose:
            print(f"TX to   {addr}: {hexdump(pkt)}")
        self.transport.sendto(pkt, addr)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Virtual VISCA over IP camera (UDP/52381).")
    ap.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP port (default: {DEFAULT_PORT})")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging (hex RX/TX)")
    args = ap.parse_args()

    state = CameraState()
    emulator = ViscaEmulator(state, verbose=args.verbose)

    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: ViscaUdpServer(emulator, verbose=args.verbose),
        local_addr=(args.bind, args.port),
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()


if __name__ == "__main__":
    asyncio.run(main())

