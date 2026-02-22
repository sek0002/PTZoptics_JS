import sys
import math
import time as _time
import json
import os
import platform
from pathlib import Path


def _ptz_user_data_dir(app_name: str = "PTZoptics_JS") -> str:
    """
    Return a per-user writable data directory for the app.

    Preference order:
      1) PTZ_APPDATA_DIR env var (set by PyInstaller runtime hook)
      2) OS standard user data dir
    """
    env_dir = (os.environ.get("PTZ_APPDATA_DIR") or "").strip()
    if env_dir:
        p = os.path.abspath(env_dir)
        os.makedirs(p, exist_ok=True)
        return p

    home = str(Path.home())
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")

    p = os.path.join(base, app_name)
    os.makedirs(p, exist_ok=True)
    return p

from PySide6.QtCore import Qt, QTimer, QPointF, QUrl
from PySide6.QtGui import QFont, QColor, QBrush, QPen, QDesktopServices, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QTabWidget,
    QGraphicsSimpleTextItem,
    QMessageBox,
    QInputDialog,
    QStyle,
)
# QtMultimedia (optional; backend-dependent RTMP support)
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
except Exception:  # pragma: no cover
    QMediaPlayer = QAudioOutput = QVideoWidget = None  # type: ignore

# QtCharts (bundled with PySide6 as an optional module)
try:
    from PySide6.QtCharts import (
        QChart,
        QChartView,
        QLineSeries,
        QScatterSeries,
        QValueAxis,
    )
except Exception as e:  # pragma: no cover
    QChart = QChartView = QLineSeries = QScatterSeries = QValueAxis = None  # type: ignore
    _qtcharts_import_error = e

from camera import Camera
from exceptions import ViscaException, NoQueryResponse

# Optional controller backend (pygame)
try:
    import pygame  # type: ignore
except Exception:  # pragma: no cover
    pygame = None


# -----------------------------
# VISCA presets (label, payload, is_query)
# -----------------------------
# Local reference PDF (place next to this script)
VISCA_PDF_FILENAME = "PTZOptics-VISCA-over-IP-Rev-1_2-8-20.pdf"

VISCA_COMMANDS = [
    ("Power ON",  "04 00 02", False),
    ("Power OFF", "04 00 03", False),
    ("Zoom Stop", "04 07 00", False),
    ("Zoom Tele (standard)", "04 07 02", False),
    ("Zoom Wide (standard)", "04 07 03", False),
    ("Focus Auto",   "04 38 02", False),
    ("Focus Manual", "04 38 03", False),
    ("Pan Left (slow)",  "06 01 01 00 01 03", False),
    ("Pan Right (slow)", "06 01 01 00 02 03", False),
    ("Tilt Up (slow)",   "06 01 00 01 03 01", False),
    ("Tilt Down (slow)", "06 01 00 01 03 02", False),
    ("PT Stop",          "06 01 00 00 03 03", False),
    ("Exposure Manual", "04 39 03", False),
    ("Exposure Auto",   "04 39 00", False),
    ("WB Auto",    "04 35 00", False),
    ("WB Indoor",  "04 35 01", False),
    ("WB Outdoor", "04 35 02", False),
    ("WB Manual", "04 35 05", False),
    ("Preset 1 Recall", "04 3F 02 01", False),
    ("Preset 1 Save",   "04 3F 01 01", False),
    ("Query: Power",         "04 00", True),
    ("Query: Zoom Position", "04 47", True),
    ("Query: Focus Mode",    "04 38", True),
]


def sanitize_hex_payload(s: str) -> str:
    """
    Accepts:
      - "04 00 02"
      - "040002"
      - "04,00,02"
    Returns spaced hex pairs like "04 00 02".
    """
    s = (s or "").strip().lower()
    if not s:
        raise ValueError("Empty payload")

    filtered = []
    for ch in s:
        if ch in "0123456789abcdef":
            filtered.append(ch)
    hexchars = "".join(filtered)

    if len(hexchars) % 2 != 0:
        raise ValueError("Hex payload must have an even number of hex digits")

    pairs = [hexchars[i : i + 2] for i in range(0, len(hexchars), 2)]
    return " ".join(pairs)


def parse_visca_full_frame_hex(s: str) -> tuple[str, bool] | None:
    """Return (normalized_frame_hex, is_query) if s looks like a full VISCA frame (81 .. FF)."""
    norm = sanitize_hex_payload(s)
    parts = [p for p in norm.split() if p]
    if len(parts) < 3:
        return None
    if parts[0] != "81" or parts[-1] != "ff":
        return None
    is_query = (parts[1] == "09")
    return norm.upper(), bool(is_query)


def apply_always_dark_theme(app: QApplication) -> None:
    """Force an always-dark UI theme (Fusion + dark palette)."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(20, 20, 20))
    pal.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    pal.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(45, 45, 45))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.BrightText, QColor(255, 0, 0))
    pal.setColor(QPalette.Highlight, QColor(42, 130, 218))
    pal.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    try:
        app.setPalette(pal)
    except Exception:
        pass

    # Small stylesheet to unify widget backgrounds across platforms.
    try:
        app.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #dcdcdc; }
            QGroupBox { border: 1px solid #3a3a3a; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListView {
                background-color: #141414; border: 1px solid #3a3a3a;
            }
            QPushButton, QToolButton { background-color: #2d2d2d; border: 1px solid #3a3a3a; padding: 4px 10px; }
            QPushButton:hover, QToolButton:hover { background-color: #3a3a3a; }
            QTabWidget::pane { border: 1px solid #3a3a3a; }
            QTabBar::tab { background: #2d2d2d; border: 1px solid #3a3a3a; padding: 6px 10px; }
            QTabBar::tab:selected { background: #1e1e1e; }
            QMenu { background-color: #1e1e1e; border: 1px solid #3a3a3a; }
            QMenu::item:selected { background-color: #2a82da; color: #000000; }
            QToolTip { background-color: #2d2d2d; color: #dcdcdc; border: 1px solid #3a3a3a; }
        """)
    except Exception:
        pass

class LogBuffer:
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.lines: list[str] = []

    def add(self, msg: str) -> str:
        ts = _time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.lines.append(line)
        if len(self.lines) > self.capacity:
            self.lines = self.lines[-self.capacity :]
        return line


# -----------------------------
# Response curve model (per axis)
# -----------------------------
class AxisCurveParams:
    """
    Symmetric curve around 0:
      - deadzone (x range around 0 that outputs 0)
      - expo (applies after deadzone remap)
      - min_thresh/max_thresh output clamp
    Also supports an editor control point at x_mid to set expo interactively.
    """
    def __init__(self):
        self.deadzone: float = 0.05
        self.expo: float = 1.0
        self.min_thresh: float = -1.0
        self.max_thresh: float = 1.0
        self.invert: bool = False

        self.zero_based: bool = False  # treat input as [0..1] instead of [-1..1]

        # Quantization / speed bins (optional)
        # When enabled, shaped output is snapped to discrete bins, and can be mapped to an integer VISCA speed.
        self.quantize_bins: bool = False
        self.bin_count: int = 8          # number of discrete non-zero levels
        self.speed_min: int = 0          # inclusive
        self.speed_max: int = 7          # inclusive
        # editor: use y at x_mid to derive expo; keep in sync
        self.x_mid: float = 0.5
        self.y_mid: float = 0.5  # for expo=1.0

    def sync_y_mid_from_expo(self):
        x = max(1e-6, min(0.999999, self.x_mid))
        e = max(0.2, min(8.0, self.expo))
        self.y_mid = x ** e

    def sync_expo_from_y_mid(self):
        x = max(1e-6, min(0.999999, self.x_mid))
        y = max(1e-6, min(0.999999, self.y_mid))
        self.expo = math.log(y) / math.log(x)
        self.expo = max(0.2, min(8.0, self.expo))

    def shape(self, x: float) -> float:
        """Shape raw axis input into a normalized output.

        Modes:
          - symmetric (default): raw x in [-1..+1]
          - zero_based: raw x in [0..+1] (e.g., triggers)

        Inversion:
          - symmetric: invert mirrors the output sign
          - zero_based: invert keeps 0 anchored and drives negative values (0..-1)
        """

        # Note: invert is applied to the SHAPED output (post-curve), not to the raw input.

        if bool(getattr(self, 'zero_based', False)):
            # Clamp to [0..1]
            try:
                x = float(x)
            except Exception:
                x = 0.0
            x = max(0.0, min(1.0, x))

            # Deadzone from 0 upward
            dz = max(0.0, min(0.95, float(getattr(self, 'deadzone', 0.0))))
            if x <= dz:
                x = 0.0
            else:
                x = (x - dz) / (1.0 - dz)

            # Expo
            e = max(0.2, min(8.0, float(getattr(self, 'expo', 1.0))))
            if x != 0.0 and abs(e - 1.0) > 1e-6:
                x = x ** e

            # Clamp outputs
            mn = max(-1.0, min(1.0, float(getattr(self, 'min_thresh', -1.0))))
            mx = max(-1.0, min(1.0, float(getattr(self, 'max_thresh', 1.0))))
            if mn > mx:
                mn, mx = mx, mn
            if x < mn:
                x = mn
            if x > mx:
                x = mx

            # Invert shaped output (post-curve): 0 remains 0, +1 becomes -1
            if bool(getattr(self, 'invert', False)):
                x = -x
            return float(x)

        # --- Symmetric mode (original behaviour) ---
        dz = max(0.0, min(0.5, self.deadzone))
        if abs(x) <= dz:
            x = 0.0
        else:
            x = (abs(x) - dz) / (1.0 - dz) * (1.0 if x >= 0 else -1.0)

        e = max(0.2, min(8.0, self.expo))
        if x != 0.0 and abs(e - 1.0) > 1e-6:
            x = (abs(x) ** e) * (1.0 if x >= 0 else -1.0)

        # clamp outputs
        mn = max(-1.0, min(1.0, self.min_thresh))
        mx = max(-1.0, min(1.0, self.max_thresh))
        if mn > mx:
            mn, mx = mx, mn

        if x < mn:
            x = mn
        if x > mx:
            x = mx

        # Invert shaped output (post-curve)
        if self.invert:
            x = -x
        return x

    def _clamp_int(self, v: int, lo: int, hi: int) -> int:
        try:
            v = int(v)
        except Exception:
            v = lo
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(hi, v))

    def speed_from_shaped(self, shaped: float) -> int:
        """
        Map |shaped| in [0..1] to an integer speed using bins.

        Conventions:
          - shaped == 0 -> speed 0 (stop / no movement intent)
          - otherwise -> speed in [speed_min..speed_max], distributed across bin_count levels.
        """
        if shaped is None:
            return 0

        mag = abs(float(shaped))
        mag = max(0.0, min(1.0, mag))

        if mag <= 1e-9:
            return 0

        bins = self._clamp_int(getattr(self, 'bin_count', 8), 1, 256)
        smin = self._clamp_int(getattr(self, 'speed_min', 0), -1024, 1024)
        smax = self._clamp_int(getattr(self, 'speed_max', 7), -1024, 1024)
        if smin > smax:
            smin, smax = smax, smin

        # Choose a discrete level 1..bins (0 reserved for stop)
        level = int(math.ceil(mag * bins - 1e-12))
        level = max(1, min(bins, level))

        if bins == 1:
            return int(smax)

        # Distribute levels across [smin..smax]
        frac = (level - 1) / float(bins - 1)
        speed = int(round(smin + frac * (smax - smin)))
        return int(speed)

    def shaped_with_bins(self, raw: float) -> tuple[float, int]:
        """
        Returns (shaped_value_for_curve, mapped_speed_int).

        If quantize_bins is enabled, shaped output is snapped to the center of the selected bin,
        and speed is computed from the snapped magnitude.
        """
        shaped = float(self.shape(raw))
        if not bool(getattr(self, 'quantize_bins', False)):
            return shaped, self.speed_from_shaped(shaped)

        # Quantize magnitude into bins and rebuild signed shaped value at bin center
        bins = self._clamp_int(getattr(self, 'bin_count', 8), 1, 256)
        mag = abs(shaped)
        if mag <= 1e-9:
            return 0.0, 0

        level = int(math.ceil(mag * bins - 1e-12))
        level = max(1, min(bins, level))

        # Bin center in (0..1]
        center = (level - 0.5) / float(bins)
        center = max(0.0, min(1.0, center))
        q = center * (1.0 if shaped >= 0 else -1.0)

        return q, self.speed_from_shaped(q)



# -----------------------------
# Interactive curve editor (one curve visible, axis selectable)
# -----------------------------
class CurveChartView(QChartView):
    """
    Adds simple draggable handles on the response curve:
      - Deadzone handle at (+dz, 0)
      - Expo handle at (x_mid, y_mid)
      - Max threshold handle at (1, max_thresh)
      - Min threshold handle at (-1, min_thresh)
    Also shows the live controller point (raw, shaped).

    Dragging constraints:
      - Deadzone: x in [0, 0.5], y locked to 0
      - Expo: x locked to x_mid, y in [0, 1]
      - Max/Min: x locked to +/-1, y in [-1, 1]
    """
    HANDLE_DZ = "dz"
    HANDLE_EXPO = "expo"
    HANDLE_MAX = "max"
    HANDLE_MIN = "min"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging: str | None = None
        self._handle_pick_radius_px = 14

        self.params: AxisCurveParams | None = None
        self.on_params_changed = None  # callback()

        self.curve_series: QLineSeries | None = None
        self.handle_dz: QScatterSeries | None = None
        self.handle_expo: QScatterSeries | None = None
        self.handle_max: QScatterSeries | None = None
        self.handle_min: QScatterSeries | None = None
        self.live_point: QScatterSeries | None = None


        # HUD label (top-right) - optional speed readout
        self.hud_text_item = None  # set by parent (ResponseCurveEditor)

    def _position_hud(self):
        """Anchor HUD text to the chart plot area's top-right corner."""
        try:
            chart = self.chart()
            item = getattr(self, "hud_text_item", None)
            if chart is None or item is None:
                return
            plot = chart.plotArea()
            margin = 10
            br = item.boundingRect()
            item.setPos(plot.right() - br.width() - margin, plot.bottom() - br.height() - margin)
        except Exception:
            pass

    def update_speed_hud(self, raw_x: float):
        """Update HUD with mapped speed for the given raw input x."""
        if self.params is None or QChart is None:
            return
        item = getattr(self, "hud_text_item", None)
        if item is None:
            return
        try:
            raw_x = float(raw_x)
            if self.params is not None and bool(getattr(self.params, 'zero_based', False)):
                raw_x = max(0.0, min(1.0, raw_x))
            else:
                raw_x = max(-1.0, min(1.0, raw_x))
            shaped, speed = self.params.shaped_with_bins(raw_x)
            item.setText(f"speed={int(speed)}")
            item.setVisible(True)
            self._position_hud()
        except Exception:
            pass

    def resizeEvent(self, event):
        # Keep HUD anchored on resize / layout changes.
        try:
            self._position_hud()
        except Exception:
            pass
        return super().resizeEvent(event)
    def _hit_test(self, series: QScatterSeries, pos_px) -> bool:
        if series.count() == 0:
            return False
        pt = series.at(0)
        chart = self.chart()
        sp = chart.mapToPosition(pt, series)
        dx = sp.x() - pos_px.x()
        dy = sp.y() - pos_px.y()
        return (dx * dx + dy * dy) ** 0.5 <= self._handle_pick_radius_px

    def mousePressEvent(self, event):
        if self.params is None or QChart is None:
            return super().mousePressEvent(event)

        pos = event.position()
        if self.handle_dz and self._hit_test(self.handle_dz, pos):
            self._dragging = self.HANDLE_DZ
            event.accept()
            return
        if self.handle_expo and self._hit_test(self.handle_expo, pos):
            self._dragging = self.HANDLE_EXPO
            event.accept()
            return
        if self.handle_max and self._hit_test(self.handle_max, pos):
            self._dragging = self.HANDLE_MAX
            event.accept()
            return
        if self.handle_min and self._hit_test(self.handle_min, pos):
            self._dragging = self.HANDLE_MIN
            event.accept()
            return

        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.params is None or QChart is None:
            return super().mouseMoveEvent(event)

        # Only reshape while a handle is being dragged
        if not self._dragging:
            return super().mouseMoveEvent(event)

        chart = self.chart()
        if chart is None:
            return super().mouseMoveEvent(event)

        # Convert mouse position (pixels) -> chart value coords
        pos_px = event.position()
        v = chart.mapToValue(pos_px)

        p = self.params

        if self._dragging == self.HANDLE_DZ:
            # deadzone: x in [0..0.5], y locked to 0
            p.deadzone = max(0.0, min(0.95 if bool(getattr(p, 'zero_based', False)) else 0.5, float(v.x())))

        elif self._dragging == self.HANDLE_EXPO:
            # expo handle: x locked to x_mid, y in [0..1]
            p.y_mid = max(1e-6, min(0.999999, float(v.y())))
            p.sync_expo_from_y_mid()

        elif self._dragging == self.HANDLE_MAX:
            # max threshold: x locked to +1, y in [-1..1]
            p.max_thresh = max(-1.0, min(1.0, float(v.y())))

        elif self._dragging == self.HANDLE_MIN:
            # min threshold: x locked to -1 (symmetric) or 0 (zero-based); y in [-1..1]
            p.min_thresh = max(-1.0, min(1.0, float(v.y())))

        if callable(self.on_params_changed):
            self.on_params_changed()

        event.accept()

    def leaveEvent(self, event):
        # Hide hover speed readout when the cursor leaves the chart.
        try:
            item = getattr(self, "hud_text_item", None)
            if item is not None:
                item.setVisible(False)
        except Exception:
            pass
        return super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging = None
        return super().mouseReleaseEvent(event)


class ResponseCurveEditor(QWidget):
    """
    Single response curve at a time + axis selector.

    - Select axis (0..N-1)
    - Shows curve and draggable control points
    - Shows live controller point on the curve
    """
    def __init__(self, label_font: QFont, mono: QFont, parent=None):
        super().__init__(parent)

        self.label_font = label_font
        self.mono = mono

        # per-axis params (sized by detected controller axis count)
        self.axis_params: list[AxisCurveParams] = []
        self.selected_axis = 0
        self.controls = ["Pan", "Tilt", "Zoom", "Focus"]
        self.NONE_LABEL = "Unassigned"
        self.selected_control = None
        # map: physical axis index -> logical control (None = Unassigned)
        self.axis_control_map: dict[int, str | None] = {}
        self._syncing_selectors = False
        self.set_axis_count(1)  # default until controller selected
        self.last_raw = 0.0

        if QChart is None:
            layout = QVBoxLayout()
            msg = QLabel(
                "QtCharts is not available in this environment.\n"
                "Install/enable PySide6.QtCharts and re-run."
            )
            msg.setFont(mono)
            layout.addWidget(msg)
            self.setLayout(layout)
            return

        # Axis / control selector row
        self.control_combo = QComboBox()
        self.control_combo.setFont(mono)
        self.control_combo.addItem(self.NONE_LABEL, None)
        for c in self.controls:
            self.control_combo.addItem(c, c)

        self.axis_combo = QComboBox()
        self.axis_combo.setFont(mono)

        self.invert_chk = QCheckBox("Invert")
        self.invert_chk.setFont(label_font)

        self.zero_chk = QCheckBox("Zero start")
        self.zero_chk.setFont(label_font)

        top = QHBoxLayout()
        top.addWidget(QLabel("Axis:"))
        top.addWidget(self.axis_combo)
        top.addSpacing(10)
        top.addWidget(QLabel("Control:"))
        top.addWidget(self.control_combo)
        top.addStretch(1)
        top.addWidget(self.zero_chk)
        top.addWidget(self.invert_chk)

# Chart + series
        self.curve_series = QLineSeries()
        self.curve_series.setName("response")

        self.handle_dz = QScatterSeries()
        self.handle_dz.setName("deadzone")
        self.handle_dz.setMarkerSize(10)

        self.handle_expo = QScatterSeries()
        self.handle_expo.setName("expo")
        self.handle_expo.setMarkerSize(10)

        self.handle_max = QScatterSeries()
        self.handle_max.setName("max thr")
        self.handle_max.setMarkerSize(10)

        self.handle_min = QScatterSeries()
        self.handle_min.setName("min thr")
        self.handle_min.setMarkerSize(10)

        self.live_point = QScatterSeries()
        self.live_point.setName("live")
        self.live_point.setMarkerSize(12)

        # Bin guide lines (drawn only when quantization is enabled)
        self.bin_guides: list[QLineSeries] = []

        # --- Series styling (dark theme friendly)
        self.curve_series.setPen(QPen(QColor(240, 208, 100), 3))

        def style_handle(s: QScatterSeries, fill: QColor, border: QColor, size: float, shape):
            s.setMarkerSize(size)
            s.setBrush(QBrush(fill))
            s.setPen(QPen(border, 2))
            try:
                s.setMarkerShape(shape)
            except Exception:
                pass

        # Handle shapes: circle for DZ, rectangle for thresholds, circle for expo
        style_handle(self.handle_dz, QColor(240, 208, 100), QColor(20, 22, 26), 14,
                   QScatterSeries.MarkerShapeCircle)
        style_handle(self.handle_expo, QColor(255, 235, 170), QColor(20, 22, 26), 14,
                   QScatterSeries.MarkerShapeCircle)
        style_handle(self.handle_max, QColor(120, 200, 255), QColor(20, 22, 26), 14,
                   QScatterSeries.MarkerShapeRectangle)
        style_handle(self.handle_min, QColor(120, 200, 255), QColor(20, 22, 26), 14,
                   QScatterSeries.MarkerShapeRectangle)

        # Live point: green with a white-ish outline
        self.live_point.setBrush(QBrush(QColor(120, 235, 160)))
        self.live_point.setPen(QPen(QColor(235, 235, 235), 2))
        try:
            self.live_point.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        except Exception:
            pass

        chart = QChart()
        # Text label for mapped speed near the live point (shown only when quantizing)
        self._speed_text_item = QGraphicsSimpleTextItem()
        self._speed_text_item.setBrush(QBrush(QColor(235,235,235)))
        self._speed_text_item.setZValue(10)
        self._speed_text_item.setVisible(False)
        try:
            chart.scene().addItem(self._speed_text_item)
        except Exception:
            pass

        chart.legend().setVisible(False)  # hide key/labels
        # --- Dark theme styling
        chart.setBackgroundBrush(QBrush(QColor(20, 22, 26)))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QBrush(QColor(14, 16, 19)))
        chart.setTitleBrush(QBrush(QColor(230, 230, 230)))
        chart.setBackgroundRoundness(8)

        chart.addSeries(self.curve_series)
        chart.addSeries(self.handle_dz)
        chart.addSeries(self.handle_expo)
        chart.addSeries(self.handle_max)
        chart.addSeries(self.handle_min)
        chart.addSeries(self.live_point)

        ax_x = QValueAxis()
        ax_x.setRange(-1.0, 1.0)
        ax_x.setTitleText("input (raw)")
        ax_x.setLabelsColor(QColor(200, 200, 200))
        ax_x.setTitleBrush(QBrush(QColor(200, 200, 200)))
        ax_x.setGridLineColor(QColor(45, 48, 55))
        ax_x.setMinorGridLineColor(QColor(30, 32, 38))
        ax_x.setLinePen(QPen(QColor(80, 85, 95)))
        ax_x.setMinorTickCount(2)
        ax_y = QValueAxis()
        ax_y.setRange(-1.05, 1.05)
        ax_y.setTitleText("output (shaped)")
        ax_y.setLabelsColor(QColor(200, 200, 200))
        ax_y.setTitleBrush(QBrush(QColor(200, 200, 200)))
        ax_y.setGridLineColor(QColor(45, 48, 55))
        ax_y.setMinorGridLineColor(QColor(30, 32, 38))
        ax_y.setLinePen(QPen(QColor(80, 85, 95)))
        ax_y.setMinorTickCount(2)

        chart.addAxis(ax_x, Qt.AlignBottom)
        chart.addAxis(ax_y, Qt.AlignLeft)


        # Keep references for later range updates (zero-based mode)
        self.ax_x = ax_x
        self.ax_y = ax_y

        for s in (self.curve_series, self.handle_dz, self.handle_expo, self.handle_max, self.handle_min, self.live_point):
            s.attachAxis(ax_x)
            s.attachAxis(ax_y)

        chart.setTitle("Response curve")

        self.view = CurveChartView()
        self.view.setChart(chart)
        # Ensure HUD text item is actually in a scene (chart.scene() is often None here)
        try:
            sc = self.view.scene()
            if sc is not None and self._speed_text_item.scene() is None:
                sc.addItem(self._speed_text_item)
        except Exception:
            pass
        try:
            self._speed_text_item.setFont(self.mono)
        except Exception:
            pass

        # Hover readout (speed) text item
        self.view.hud_text_item = self._speed_text_item
        self.view.setMinimumHeight(420)
        self.view.setStyleSheet("background: rgb(14,16,19); border: 0px;")

        # connect editor plumbing
        self.view.params = self.axis_params[self.selected_axis]
        try:
            self._sync_bins_widgets_from_params()
        except Exception:
            pass
        try:
            self._sync_bins_widgets_from_params()
        except Exception:
            pass

        self.view.curve_series = self.curve_series
        self.view.handle_dz = self.handle_dz
        self.view.handle_expo = self.handle_expo
        self.view.handle_max = self.handle_max
        self.view.handle_min = self.handle_min
        self.view.live_point = self.live_point
        self.view.on_params_changed = self._redraw

        # Small numeric readout (optional but useful)
        self.readout = QLabel("")
        self.readout.setFont(mono)

        # --- Quantization / speed bins controls (optional)
        self.quantize_chk = QCheckBox("Quantize (bins)")
        self.quantize_chk.setFont(label_font)

        self.speed_min_spin = QSpinBox()
        self.speed_min_spin.setFont(mono)
        self.speed_min_spin.setRange(0, 255)
        self.speed_min_spin.setValue(0)
        self.speed_min_spin.setToolTip("Min speed value (inclusive)")

        self.speed_max_spin = QSpinBox()
        self.speed_max_spin.setFont(mono)
        self.speed_max_spin.setRange(0, 255)
        self.speed_max_spin.setValue(7)
        self.speed_max_spin.setToolTip("Max speed value (inclusive)")

        self.speed_preview = QLabel("")
        self.speed_preview.setFont(mono)
        self.speed_preview.setStyleSheet("color: rgb(200,200,200);")

        # wiring
        self.control_combo.currentIndexChanged.connect(self._on_control_changed)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.invert_chk.stateChanged.connect(self._on_invert_changed)
        self.zero_chk.stateChanged.connect(self._on_zero_changed)
        self.quantize_chk.stateChanged.connect(self._on_bins_changed)        
        self.speed_min_spin.valueChanged.connect(self._on_bins_changed)
        self.speed_max_spin.valueChanged.connect(self._on_bins_changed)

        # layout
        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.view, stretch=1)

        bins_row = QHBoxLayout()
        bins_row.setSpacing(8)
        bins_row.addWidget(self.quantize_chk)
        bins_row.addSpacing(10)        
        bins_row.addWidget(QLabel("Speed min:"))
        bins_row.addWidget(self.speed_min_spin)
        bins_row.addWidget(QLabel("max:"))
        bins_row.addWidget(self.speed_max_spin)
        bins_row.addStretch(1)

        root.addLayout(bins_row)
        root.addWidget(self.readout)
        root.addWidget(self.speed_preview)
        self.setLayout(root)
        self._redraw()

    def set_axis_count(self, axis_count: int):
        axis_count = int(max(1, axis_count))

        # resize params
        cur = len(self.axis_params)
        if axis_count > cur:
            for _ in range(axis_count - cur):
                p = AxisCurveParams()
                p.sync_y_mid_from_expo()
                self.axis_params.append(p)
        elif axis_count < cur:
            self.axis_params = self.axis_params[:axis_count]

        # clamp selected axis
        if self.selected_axis < 0:
            self.selected_axis = 0
        if self.selected_axis >= axis_count:
            self.selected_axis = axis_count - 1

        # rebuild axis->control mapping (keep existing where possible; default Unassigned)
        old = getattr(self, "axis_control_map", {}) or {}
        new_map: dict[int, str | None] = {}
        for i in range(axis_count):
            ctrl = old.get(i)
            new_map[i] = ctrl if (ctrl in self.controls) else None
        self.axis_control_map = new_map
        self.selected_control = self.axis_control_map.get(self.selected_axis, None)

        # repopulate combo(s)
        if hasattr(self, 'axis_combo'):
            self._syncing_selectors = True
            try:
                self.axis_combo.blockSignals(True)
                self.axis_combo.clear()
                for i in range(axis_count):
                    ctrl = self.axis_control_map.get(i, None)
                    disp = ctrl if ctrl is not None else self.NONE_LABEL
                    self.axis_combo.addItem(f"Axis {i} → {disp}", i)
                self.axis_combo.setCurrentIndex(self.selected_axis)
                self.axis_combo.blockSignals(False)

                if hasattr(self, 'control_combo'):
                    self.control_combo.blockSignals(True)
                    ctrl = self.axis_control_map.get(self.selected_axis, None)
                    idx2 = 0 if ctrl is None else self.control_combo.findData(ctrl)
                    self.control_combo.setCurrentIndex(max(0, idx2))
                    self.control_combo.blockSignals(False)
            finally:
                self._syncing_selectors = False

        # update bound params/handles
        if hasattr(self, 'view') and self.view is not None:
            self.view.params = self.axis_params[self.selected_axis]
            self.invert_chk.blockSignals(True)
            self.invert_chk.setChecked(self.view.params.invert)
            self.invert_chk.blockSignals(False)
            if hasattr(self, "zero_chk"):
                self.zero_chk.blockSignals(True)
                self.zero_chk.setChecked(bool(getattr(self.view.params, "zero_based", False)))
                self.zero_chk.blockSignals(False)
            self._redraw()

    def _on_control_changed(self, idx: int):
        if self._syncing_selectors:
            return

        data = self.control_combo.currentData() if hasattr(self, 'control_combo') else None
        ctrl = data if (data in self.controls) else None

        # Axis -> Control mapping: assign to currently selected axis (None = Unassigned)
        self.selected_control = ctrl
        # Cap spinner maximums by control
        ctrl = getattr(self, "selected_control", None)
        if ctrl == "Pan":
            self.speed_max_spin.setMaximum(18)
        elif ctrl == "Tilt":
            self.speed_max_spin.setMaximum(14)
        elif ctrl in ("Zoom", "Focus"):
            self.speed_max_spin.setMaximum(8)
        else:
            self.speed_max_spin.setMaximum(7)

        self.axis_control_map[int(self.selected_axis)] = ctrl
        # Apply control-specific defaults for speed bin mapping
        try:
            self._apply_control_defaults_for_bins(ctrl)
        except Exception:
            pass

        # Reflect defaults in the bins UI immediately (this also enforces bins to cover the full integer range)
        try:
            self._sync_bins_widgets_from_params()
            self._on_bins_changed()
        except Exception:
            pass
        # Update the axis combo label (Axis N -> Control) without changing selection.
        self._syncing_selectors = True
        try:
            self.axis_combo.blockSignals(True)
            cur_idx = self.axis_combo.currentIndex()
            disp = ctrl if ctrl is not None else self.NONE_LABEL
            self.axis_combo.setItemText(cur_idx, f"Axis {self.selected_axis} → {disp}")
            self.axis_combo.blockSignals(False)
        finally:
            self._syncing_selectors = False

        # refresh curve display (params are per-axis)
        if hasattr(self, 'view') and self.view is not None:
            self.view.params = self.axis_params[self.selected_axis]
            self.invert_chk.blockSignals(True)
            self.invert_chk.setChecked(self.view.params.invert)
            self.invert_chk.blockSignals(False)
            self._redraw()

    def _on_axis_changed(self, idx: int):
        if getattr(self, "_syncing_selectors", False):
            return

        data = self.axis_combo.currentData()
        if data is None:
            return

        self.selected_axis = int(data)

        # Reflect axis assignment in the control selector (None -> Unassigned)
        ctrl = self.axis_control_map.get(self.selected_axis, None)
        self._syncing_selectors = True
        try:
            self.control_combo.blockSignals(True)
            idx2 = 0 if ctrl is None else self.control_combo.findData(ctrl)
            self.control_combo.setCurrentIndex(max(0, idx2))
            self.control_combo.blockSignals(False)
        finally:
            self._syncing_selectors = False

        self.selected_control = ctrl
        # Cap spinner maximums by control
        ctrl = getattr(self, "selected_control", None)
        if ctrl == "Pan":
            self.speed_max_spin.setMaximum(18)
        elif ctrl == "Tilt":
            self.speed_max_spin.setMaximum(14)
        elif ctrl in ("Zoom", "Focus"):
            self.speed_max_spin.setMaximum(8)
        else:
            self.speed_max_spin.setMaximum(7)
        # Preserve this axis' stored speed settings; just sync/clamp UI for the selected control.
        try:
            self._sync_bins_widgets_from_params()
        except Exception:
            pass
        self.view.params = self.axis_params[self.selected_axis]
        self.invert_chk.blockSignals(True)
        self.invert_chk.setChecked(self.view.params.invert)
        self.invert_chk.blockSignals(False)
        # Keep zero-start checkbox in sync with selected axis params
        self.zero_chk.blockSignals(True)
        self.zero_chk.setChecked(bool(getattr(self.view.params, "zero_based", False)))
        self.zero_chk.blockSignals(False)

        self._redraw()

    def get_active_axis(self) -> int:
        """Return the currently selected physical axis."""
        return int(self.selected_axis)


    def _apply_control_defaults_for_bins(self, ctrl: str | None):
        """Set sensible default speed ranges/bins when assigning a control."""
        p = self.axis_params[self.selected_axis]
        # Defaults per VISCA conventions:
        # - Zoom/Focus speed p: 0..7 (0=slow, 7=fast)   [stop is separate when shaped==0]
        # - Pan speed VV: 0x01..0x18 (1..24)
        # - Tilt speed WW: 0x01..0x14 (1..20)
        if ctrl == "Pan":
            p.speed_min, p.speed_max, p.bin_count = 1, 18, 18
        elif ctrl == "Tilt":
            p.speed_min, p.speed_max, p.bin_count = 1, 14, 14
        elif ctrl == "Zoom":
            # VISCA: signed_speed==0 is STOP, so motion speeds should be 1..7
            p.speed_min, p.speed_max, p.bin_count = 1, 8, 8
        elif ctrl == "Focus":
            # VISCA: signed_speed==0 is STOP, so motion speeds should be 1..7
            p.speed_min, p.speed_max, p.bin_count = 1, 8, 8
        else:
            p.speed_min, p.speed_max, p.bin_count = 0, 7, 7

    def _sync_bins_widgets_from_params(self):
        if not all(hasattr(self, n) for n in ("quantize_chk","speed_min_spin","speed_max_spin")):
            return
        p = self.axis_params[self.selected_axis]
        try:
            self.quantize_chk.blockSignals(True)            
            self.speed_min_spin.blockSignals(True)
            self.speed_max_spin.blockSignals(True)

            self.quantize_chk.setChecked(bool(getattr(p, "quantize_bins", False)))            
            self.speed_min_spin.setValue(int(getattr(p, "speed_min", 0)))
            self.speed_max_spin.setValue(int(getattr(p, "speed_max", 7)))

        finally:
            pass
            self.quantize_chk.blockSignals(False)            
            self.speed_min_spin.blockSignals(False)
            self.speed_max_spin.blockSignals(False)
        # Enforce bins to cover full integer range in UI
        try:
            self._on_bins_changed()
        except Exception:
            pass


    def _on_bins_changed(self, *_args):
        p = self.axis_params[self.selected_axis]
        p.quantize_bins = bool(self.quantize_chk.isChecked())

        # Read UI values
        speed_min = int(self.speed_min_spin.value())
        speed_max = int(self.speed_max_spin.value())
        ctrl = self.selected_control

        # Clamp per-control VISCA ranges
        if ctrl == "Pan":
            speed_max = min(speed_max, 18)
            speed_min = max(1, min(speed_min, speed_max))
        elif ctrl == "Tilt":
            speed_max = min(speed_max, 14)
            speed_min = max(1, min(speed_min, speed_max))
        elif ctrl in ("Zoom", "Focus"):
            # Zoom/Focus speed factor (app): 1..8, mapped to VISCA nibble p: 0..7
            speed_max = min(speed_max, 8)
            speed_min = max(1, min(speed_min, speed_max))
        else:
            speed_min = max(0, min(speed_min, speed_max))

        # Ensure ordering
        if speed_min > speed_max:
            speed_min = speed_max

        p.speed_min = speed_min
        p.speed_max = speed_max

        # Bin count should cover every integer in [speed_min..speed_max]
        p.bin_count = max(1, min(256, int(p.speed_max - p.speed_min + 1)))

        # Push any clamped values back into the UI (without recursion)
        try:
            self.speed_min_spin.blockSignals(True)
            self.speed_max_spin.blockSignals(True)
            if int(self.speed_min_spin.value()) != p.speed_min:
                self.speed_min_spin.setValue(p.speed_min)
            if int(self.speed_max_spin.value()) != p.speed_max:
                self.speed_max_spin.setValue(p.speed_max)
        finally:
            self.speed_min_spin.blockSignals(False)
            self.speed_max_spin.blockSignals(False)

        self._redraw()

    def _on_invert_changed(self, state: int):
        # Use isChecked() rather than comparing enum/int values (PySide versions differ).
        p = self.axis_params[self.selected_axis]
        p.invert = bool(self.invert_chk.isChecked())
        self._redraw()

    def _on_zero_changed(self, state: int):
        p = self.axis_params[self.selected_axis]
        p.zero_based = bool(self.zero_chk.isChecked())
        self._redraw()

    def set_controller_value(self, raw: float):
        p = self.axis_params[self.selected_axis]
        try:
            raw_f = float(raw)
        except Exception:
            raw_f = 0.0
        # In zero-based mode we interpret the *physical* axis as [-1..1] and remap to [0..1]
        if bool(getattr(p, "zero_based", False)):
            raw_f = (raw_f + 1.0) * 0.5
            raw_f = max(0.0, min(1.0, raw_f))
        else:
            raw_f = max(-1.0, min(1.0, raw_f))
        self.last_raw = raw_f
        self._update_live_point()

    def shaped_value(self) -> float:
        return self.axis_params[self.selected_axis].shape(self.last_raw)

    def _update_live_point(self):
        if QChart is None:
            return
        p = self.axis_params[self.selected_axis]
        shaped, _spd = p.shaped_with_bins(self.last_raw)
        self.live_point.clear()
        raw_plot = self.last_raw
        self.live_point.append(QPointF(raw_plot, shaped))
        # Update HUD speed readout (top-right)
        try:
            if hasattr(self, 'view') and self.view is not None:
                self.view.update_speed_hud(self.last_raw)
        except Exception:
            pass
        self._update_readout()

    def _update_readout(self):
        p = self.axis_params[self.selected_axis]
        shaped, spd = p.shaped_with_bins(self.last_raw)
        inv = 'ON' if bool(getattr(p, 'invert', False)) else 'OFF'
        zero = 'ON' if bool(getattr(p, 'zero_based', False)) else 'OFF'
        self.readout.setText(
            f"raw={self.last_raw:+.3f}  shaped={shaped:+.3f}  invert={inv}  zero={zero}   "
            f"dz={p.deadzone:.2f}  expo={p.expo:.2f}  min={p.min_thresh:.2f}  max={p.max_thresh:.2f}"
        )
        try:
            q = 'ON' if bool(getattr(p, 'quantize_bins', False)) else 'OFF'
            self.speed_preview.setText(
                f"bins={int(getattr(p,'bin_count',8))}  speed=[{int(getattr(p,'speed_min',0))}..{int(getattr(p,'speed_max',7))}]  quantize={q}  mapped_speed={int(spd)}"
            )
        except Exception:
            pass
    def _update_bin_guides(self):
        """Draw horizontal guide lines at quantization bin centers (decimated for readability)."""
        if QChart is None:
            return
        chart = self.view.chart() if hasattr(self, 'view') and self.view is not None else None
        if chart is None:
            return

        # Remove old guide series
        try:
            for s in getattr(self, 'bin_guides', []) or []:
                try:
                    chart.removeSeries(s)
                except Exception:
                    pass
        except Exception:
            pass
        self.bin_guides = []

        p = self.axis_params[self.selected_axis]
        if not bool(getattr(p, 'quantize_bins', False)):
            return

        bins = int(max(1, min(256, getattr(p, 'bin_count', 8))))
        # To avoid clutter, cap visible guides to ~32 per side
        step = max(1, bins // 32)

        # Build lines at bin centers: y = ±(k-0.5)/bins for k=1..bins
        try:
            from PySide6.QtGui import QPen, QColor
            pen = QPen(QColor(80, 85, 95, 110), 1)
        except Exception:
            pen = None

        # Axis references for attachment
        ax_x = chart.axes(Qt.Horizontal)[0] if chart.axes(Qt.Horizontal) else None
        ax_y = chart.axes(Qt.Vertical)[0] if chart.axes(Qt.Vertical) else None

        left = 0.0 if bool(getattr(p, 'zero_based', False)) else -1.0
        right = 1.0

        for k in range(1, bins + 1, step):
            y = (k - 0.5) / float(bins)
            for yy in (+y, -y):
                s = QLineSeries()
                try:
                    s.setUseOpenGL(False)
                except Exception:
                    pass
                if pen is not None:
                    s.setPen(pen)
                s.append(left, yy)
                s.append(right, yy)
                chart.addSeries(s)
                if ax_x is not None and ax_y is not None:
                    s.attachAxis(ax_x)
                    s.attachAxis(ax_y)
                self.bin_guides.append(s)

    def _redraw(self):
        if QChart is None:
            return

        p = self.axis_params[self.selected_axis]

        # Update x-axis range for zero-based mode
        try:
            if hasattr(self, "ax_x") and self.ax_x is not None:
                if bool(getattr(p, "zero_based", False)):
                    self.ax_x.setRange(0.0, 1.0)
                else:
                    self.ax_x.setRange(-1.0, 1.0)
        except Exception:
            pass
        # keep expo/y_mid consistent unless user is actively dragging expo handle
        p.sync_y_mid_from_expo()

        # bin guide lines (if enabled)
        self._update_bin_guides()        # redraw curve
        self.curve_series.clear()
        if bool(getattr(p, "zero_based", False)):
            for i in range(201):
                x = i / 200.0  # 0..1
                y, _spd = p.shaped_with_bins(x)
                self.curve_series.append(QPointF(x, y))
        else:
            for i in range(201):
                x = -1.0 + (2.0 * i / 200.0)
                y, _spd = p.shaped_with_bins(x)
                self.curve_series.append(QPointF(x, y))

        # redraw handles (single point each)
        self.handle_dz.clear()
        self.handle_dz.append(QPointF(+p.deadzone, 0.0))

        self.handle_expo.clear()
        self.handle_expo.append(QPointF(p.x_mid, p.y_mid))

        self.handle_max.clear()
        self.handle_max.append(QPointF(1.0, p.max_thresh))

        self.handle_min.clear()
        self.handle_min.append(QPointF((0.0 if bool(getattr(p, "zero_based", False)) else -1.0), p.min_thresh))

        # live point
        self._update_live_point()
    # -----------------------------
    # Persistence helpers (profiles)
    # -----------------------------
    def get_state(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of curves + axis/control mapping."""
        params_list: list[dict[str, object]] = []
        for p in getattr(self, "axis_params", []) or []:
            try:
                params_list.append({
                    "deadzone": float(getattr(p, "deadzone", 0.05)),
                    "expo": float(getattr(p, "expo", 1.0)),
                    "min_thresh": float(getattr(p, "min_thresh", -1.0)),
                    "max_thresh": float(getattr(p, "max_thresh", 1.0)),
                    "invert": bool(getattr(p, "invert", False)),
                    "zero_based": bool(getattr(p, "zero_based", False)),
                    "quantize_bins": bool(getattr(p, "quantize_bins", False)),
                    "bin_count": int(getattr(p, "bin_count", 8)),
                    "speed_min": int(getattr(p, "speed_min", 0)),
                    "speed_max": int(getattr(p, "speed_max", 7)),
                    "x_mid": float(getattr(p, "x_mid", 0.5)),
                    "y_mid": float(getattr(p, "y_mid", 0.5)),
                })
            except Exception:
                continue

        axis_map = {}
        try:
            for k, v in (getattr(self, "axis_control_map", {}) or {}).items():
                axis_map[str(int(k))] = v
        except Exception:
            axis_map = {}

        return {
            "axis_count": int(len(getattr(self, "axis_params", []) or [])),
            "selected_axis": int(getattr(self, "selected_axis", 0)),
            "axis_control_map": axis_map,
            "axis_params": params_list,
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        try:
            params_list = state.get("axis_params", []) or []
            axis_count = int(state.get("axis_count", len(params_list) or 1))
        except Exception:
            params_list = []
            axis_count = 1

        # Do not let a saved profile shrink the current live controller axis list.
        # Profiles may have been created with no controller connected (axis_count=1).
        try:
            axis_count = max(int(axis_count), int(len(getattr(self, "axis_params", []) or [])))
        except Exception:
            pass

        # Ensure we have enough axes
        try:
            self.set_axis_count(max(1, axis_count))
        except Exception:
            pass

        # Restore mapping axis->control
        try:
            amap_raw = state.get("axis_control_map", {}) or {}
            amap: dict[int, str | None] = {}
            for k, v in amap_raw.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                amap[kk] = v if (v in self.controls) else None
            # merge into current map (ensures all axes exist)
            cur_map = getattr(self, "axis_control_map", {}) or {}
            for i in range(max(1, axis_count)):
                if i in amap:
                    cur_map[i] = amap[i]
                elif i not in cur_map:
                    cur_map[i] = None
            self.axis_control_map = cur_map
        except Exception:
            pass

        # Restore per-axis params
        try:
            for i, pd in enumerate(params_list):
                if i >= len(self.axis_params):
                    break
                p = self.axis_params[i]
                for key in ("deadzone","expo","min_thresh","max_thresh","x_mid","y_mid"):
                    if key in pd:
                        try:
                            setattr(p, key, float(pd[key]))
                        except Exception:
                            pass
                for key in ("invert","zero_based","quantize_bins"):
                    if key in pd:
                        try:
                            setattr(p, key, bool(pd[key]))
                        except Exception:
                            pass
                for key in ("bin_count","speed_min","speed_max"):
                    if key in pd:
                        try:
                            setattr(p, key, int(pd[key]))
                        except Exception:
                            pass
                # keep derived values consistent
                try:
                    p.sync_y_mid_from_expo()
                except Exception:
                    pass
        except Exception:
            pass

        # Select axis and refresh widgets/curve
        try:
            sel = int(state.get("selected_axis", 0))
        except Exception:
            sel = 0
        sel = max(0, min(sel, len(self.axis_params) - 1))
        try:
            self._syncing_selectors = True
            self.axis_combo.setCurrentIndex(sel)
        except Exception:
            pass
        finally:
            try:
                self._syncing_selectors = False
            except Exception:
                pass

        try:
            self.selected_axis = sel
            self.selected_control = self.axis_control_map.get(self.selected_axis, None)
            # Update control selector to match map
            self.control_combo.blockSignals(True)
            try:
                idx = self.control_combo.findData(self.selected_control)
                if idx >= 0:
                    self.control_combo.setCurrentIndex(idx)
                else:
                    self.control_combo.setCurrentIndex(0)
            finally:
                self.control_combo.blockSignals(False)
            # Sync checkboxes + bins UI from params
            try:
                p = self.axis_params[self.selected_axis]
                self.invert_chk.setChecked(bool(getattr(p, "invert", False)))
                self.zero_chk.setChecked(bool(getattr(p, "zero_based", False)))
                self._sync_bins_widgets_from_params()
            except Exception:
                pass
            self._redraw()
        except Exception:
            pass


class VirtualControllerWidget(QWidget):
    """
    Virtual D-pad + buttons tab.
    - Each virtual control has a dropdown mapping to any VISCA_COMMANDS entry.
    - Press highlights and sends the mapped command immediately (if assigned).
    """
    def __init__(self, mono: QFont, label_font: QFont, send_payload_cb, parent=None):
        super().__init__(parent)
        self._send_payload = send_payload_cb
        self._mono = mono
        self._label_font = label_font
        # Command source for mapping dropdowns (can be extended at runtime)
        self._command_provider = lambda: VISCA_COMMANDS


        # Styles (match your dark UI)
        self._idle = (
            "QToolButton {"
            "  color: rgb(230,230,230);"
            "  background: rgb(28,31,36);"
            "  border: 1px solid rgba(255,255,255,30);"
            "  border-radius: 10px;"
            "  padding: 10px;"
            "  font-weight: 700;"
            "}"
        )
        self._down = (
            "QToolButton {"
            "  color: rgb(20,22,26);"
            "  background: rgb(240,208,100);"
            "  border: 1px solid rgba(0,0,0,120);"
            "  border-radius: 10px;"
            "  padding: 10px;"
            "  font-weight: 800;"
            "}"
        )

        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        title = QLabel("Virtual controller")
        title.setFont(label_font)
        title.setStyleSheet("color: rgb(235,235,235); font-weight: 600;")
        root.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(16)

        # ----------------
        # D-pad
        # ----------------
        dpad_box = QGroupBox("D-pad")
        dpad_box.setFont(label_font)
        dpad_box.setStyleSheet("QGroupBox { color: rgb(210,210,210); }")
        dpad_layout = QVBoxLayout()
        dpad_layout.setSpacing(8)

        dpad_grid = QGridLayout()
        dpad_grid.setHorizontalSpacing(8)
        dpad_grid.setVerticalSpacing(8)

        self.btn_up = self._make_button("↑")
        self.btn_down = self._make_button("↓")
        self.btn_left = self._make_button("←")
        self.btn_right = self._make_button("→")
        self.btn_center = self._make_button("●")
        self.btn_center.setEnabled(False)

        dpad_grid.addWidget(self.btn_up, 0, 1)
        dpad_grid.addWidget(self.btn_left, 1, 0)
        dpad_grid.addWidget(self.btn_center, 1, 1)
        dpad_grid.addWidget(self.btn_right, 1, 2)
        dpad_grid.addWidget(self.btn_down, 2, 1)

        dpad_layout.addLayout(dpad_grid)
        dpad_layout.addLayout(self._mapping_row("Up", self.btn_up))
        dpad_layout.addLayout(self._mapping_row("Down", self.btn_down))
        dpad_layout.addLayout(self._mapping_row("Left", self.btn_left))
        dpad_layout.addLayout(self._mapping_row("Right", self.btn_right))
        dpad_box.setLayout(dpad_layout)

        # ----------------
        # Face buttons
        # ----------------
        btns_box = QGroupBox("Buttons")
        btns_box.setFont(label_font)
        btns_box.setStyleSheet("QGroupBox { color: rgb(210,210,210); }")
        btns_layout = QVBoxLayout()
        btns_layout.setSpacing(10)

        face = QGridLayout()
        face.setHorizontalSpacing(10)
        face.setVerticalSpacing(10)

        self.btn_y = self._make_button("Y")
        self.btn_x = self._make_button("X")
        self.btn_b = self._make_button("B")
        self.btn_a = self._make_button("A")

        face.addWidget(self.btn_y, 0, 1)
        face.addWidget(self.btn_x, 1, 0)
        face.addWidget(self.btn_b, 1, 2)
        face.addWidget(self.btn_a, 2, 1)

        btns_layout.addLayout(face)
        btns_layout.addLayout(self._mapping_row("A", self.btn_a))
        btns_layout.addLayout(self._mapping_row("B", self.btn_b))
        btns_layout.addLayout(self._mapping_row("X", self.btn_x))
        btns_layout.addLayout(self._mapping_row("Y", self.btn_y))
        btns_box.setLayout(btns_layout)

        row.addWidget(dpad_box, stretch=1)
        row.addWidget(btns_box, stretch=1)
        root.addLayout(row)
        root.addStretch(1)

        self.setLayout(root)

    def set_command_provider(self, provider_fn):
        """Set a callable that returns iterable[(label, payload, is_query)] for dropdowns."""
        self._command_provider = provider_fn or (lambda: VISCA_COMMANDS)
        self.refresh_command_lists()

    def _iter_commands(self):
        try:
            cmds = list(self._command_provider())
            # basic shape guard
            return [c for c in cmds if isinstance(c, (list, tuple)) and len(c) == 3]
        except Exception:
            return list(VISCA_COMMANDS)

    def refresh_command_lists(self):
        """Rebuild all mapping dropdowns to include any newly registered commands."""
        # Buttons might not exist yet if called very early; guard.
        names = [
            "btn_up", "btn_down", "btn_left", "btn_right",
            "btn_a", "btn_b", "btn_x", "btn_y",
        ]
        buttons = []
        for n in names:
            if hasattr(self, n):
                buttons.append(getattr(self, n))

        for button in buttons:
            combo = getattr(button, "_mapping_combo", None)
            if combo is None:
                continue

            # Preserve current selection by payload/is_query if possible.
            cur = combo.currentData()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem("Unassigned", None)
                for cmd_label, payload, is_query in self._iter_commands():
                    combo.addItem(cmd_label, (payload, is_query, cmd_label))

                if cur is not None:
                    # find matching payload + is_query
                    try:
                        cur_payload, cur_is_query, _cur_label = cur
                        for i in range(combo.count()):
                            data = combo.itemData(i)
                            if isinstance(data, (list, tuple)) and len(data) == 3:
                                p, q, _l = data
                                if p == cur_payload and bool(q) == bool(cur_is_query):
                                    combo.setCurrentIndex(i)
                                    break
                    except Exception:
                        pass
            finally:
                combo.blockSignals(False)

    def _make_button(self, text: str) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setFont(self._mono)
        b.setMinimumSize(56, 56)
        b.setAutoRepeat(False)
        b.setStyleSheet(self._idle)

        b.pressed.connect(lambda bt=b: self._on_pressed(bt))
        b.released.connect(lambda bt=b: self._on_released(bt))
        return b

    def _mapping_row(self, label: str, button: QToolButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lab = QLabel(label)
        lab.setFont(self._label_font)
        lab.setStyleSheet("color: rgb(200,200,200);")
        lab.setMinimumWidth(44)

        combo = QComboBox()
        combo.setFont(self._mono)
        combo.setFixedHeight(24)
        combo.setFixedWidth(150)
        combo.addItem("Unassigned", None)
        for cmd_label, payload, is_query in self._iter_commands():
            combo.addItem(cmd_label, (payload, is_query, cmd_label))
        combo.setStyleSheet(
            "QComboBox { background: rgb(28,31,36); color: rgb(235,235,235); padding: 2px 6px; font-size: 10pt; }"
            "QComboBox::drop-down { width: 18px; border: 0px; }"
            "QComboBox QAbstractItemView { background: rgb(24,26,30); color: rgb(235,235,235); }"
        )

        # attach mapping combo to the button
        button._mapping_combo = combo  # type: ignore[attr-defined]

        row.addWidget(lab)
        row.addWidget(combo)
        return row

    def _on_pressed(self, button: QToolButton):
        button.setStyleSheet(self._down)

        combo = getattr(button, "_mapping_combo", None)
        if combo is None:
            return
        data = combo.currentData()
        if data is None:
            return

        payload, is_query, label = data
        self._send_payload(payload, bool(is_query), f"[Virtual {label}]")

    def _on_released(self, button: QToolButton):
        button.setStyleSheet(self._idle)


    # -----------------------------
    # Persistence helpers (profiles)
    # -----------------------------
    def get_mapping_state(self) -> dict[str, object]:
        """Return a JSON-serializable mapping for all virtual buttons."""
        state: dict[str, object] = {}
        names = [
            "btn_up", "btn_down", "btn_left", "btn_right",
            "btn_a", "btn_b", "btn_x", "btn_y",
        ]
        for n in names:
            btn = getattr(self, n, None)
            if btn is None:
                continue
            combo = getattr(btn, "_mapping_combo", None)
            if combo is None:
                continue
            data = combo.currentData()
            # data is (payload, is_query, label) or None
            if data is None:
                state[n] = None
            else:
                try:
                    payload, is_query, label = data
                    state[n] = {"payload": str(payload), "is_query": bool(is_query), "label": str(label)}
                except Exception:
                    state[n] = None
        return state

    def apply_mapping_state(self, state: dict[str, object] | None) -> None:
        """Apply a mapping state created by get_mapping_state()."""
        if not state:
            return
        for key, val in state.items():
            btn = getattr(self, key, None)
            if btn is None:
                continue
            combo = getattr(btn, "_mapping_combo", None)
            if combo is None:
                continue
            combo.blockSignals(True)
            try:
                if val is None:
                    combo.setCurrentIndex(0)
                    continue
                payload = str(getattr(val, "get", lambda _k, _d=None: None)("payload", "")) if hasattr(val, "get") else str(val.get("payload",""))
                is_query = bool(val.get("is_query", False)) if hasattr(val, "get") else False

                # find matching payload+is_query
                found = False
                for i in range(combo.count()):
                    d = combo.itemData(i)
                    if isinstance(d, (list, tuple)) and len(d) == 3:
                        p, q, _l = d
                        if str(p) == payload and bool(q) == bool(is_query):
                            combo.setCurrentIndex(i)
                            found = True
                            break
                if not found:
                    combo.setCurrentIndex(0)
            except Exception:
                try:
                    combo.setCurrentIndex(0)
                except Exception:
                    pass
            finally:
                combo.blockSignals(False)




class HexCommandsListWidget(QWidget):
    """
    Editable list of hex commands.
    - Columns: Label + Hex payload + Full hex frame (optional)
    - Send selected row via MainWindow.send_payload
    """
    def __init__(self, mono: QFont, label_font: QFont, send_payload_cb, register_cmd_cb=None, parent=None):
        super().__init__(parent)
        self._mono = mono
        self._label_font = label_font
        self._send_payload = send_payload_cb
        self._register_cmd = register_cmd_cb

        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Hex command list")
        title.setFont(label_font)
        title.setStyleSheet("color: rgb(235,235,235); font-weight: 600;")
        root.addWidget(title)

        # Link to local VISCA command reference PDF (if present alongside this script)
        self._pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VISCA_PDF_FILENAME)

        self._pdf_link = QLabel()
        self._pdf_link.setFont(mono)
        self._pdf_link.setStyleSheet("color: rgb(180,200,255);")
        self._pdf_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._pdf_link.setOpenExternalLinks(False)
        if os.path.exists(self._pdf_path):
            self._pdf_link.setText(
                f'<a href="open_visca_pdf">Open VISCA-over-IP PDF</a>  '
                f'<span style="color: rgb(160,160,160);">({VISCA_PDF_FILENAME})</span>'
            )
        else:
            self._pdf_link.setText(
                f'<span style="color: rgb(160,160,160);">VISCA PDF not found next to script: {VISCA_PDF_FILENAME}</span>'
            )
        self._pdf_link.linkActivated.connect(self._open_visca_pdf)
        root.addWidget(self._pdf_link)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "Hex payload", "Full hex frame (optional)"])
        self.table.setFont(mono)
        self.table.setStyleSheet(
            "QTableWidget { background: rgb(14,16,19); color: rgb(235,235,235); gridline-color: rgb(45,48,55); }"
            "QHeaderView::section { background: rgb(28,31,36); color: rgb(235,235,235); padding: 6px; border: 0px; }"
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        self.table.setEditTriggers(self.table.EditTrigger.DoubleClicked | self.table.EditTrigger.EditKeyPressed)
        self.table.setHorizontalScrollMode(self.table.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setTextElideMode(Qt.ElideNone)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 360)
        self.table.setColumnWidth(2, 520)
        self.table.horizontalHeader().setStretchLastSection(False)

        root.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.add_btn = QPushButton("Add")
        self.add_btn.setFont(mono)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setFont(mono)
        self.send_btn = QPushButton("Send selected")
        self.send_btn.setFont(mono)

        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.send_btn)
        root.addLayout(btn_row)

        self.setLayout(root)

        self.add_btn.clicked.connect(self.add_row)
        self.del_btn.clicked.connect(self.delete_selected)
        self.send_btn.clicked.connect(self.send_selected)

        # Seed from existing VISCA_COMMANDS for convenience (edit freely)
        self._suppress_register = True
        self._seed_defaults()

        # Live registration into the virtual pad dropdowns
        self._suppress_register = False
        self.table.itemChanged.connect(self._on_item_changed)
    def _open_visca_pdf(self, _link: str | None = None):
        """Open the local VISCA-over-IP PDF in the system PDF viewer."""
        try:
            path = getattr(self, "_pdf_path", None)
            if not path or not os.path.exists(path):
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception:
            pass



    def _seed_defaults(self):
        for (label, payload, _is_query) in VISCA_COMMANDS:
            self._append_row(label, payload)

    def _resolve_row_command(self, r: int) -> tuple[str, str, bool] | None:
        label_item = self.table.item(r, 0)
        payload_item = self.table.item(r, 1)
        full_item = self.table.item(r, 2)
        if label_item is None:
            return None

        label = (label_item.text() or "").strip()
        payload_raw = (payload_item.text() if payload_item is not None else "") or ""
        full_raw = (full_item.text() if full_item is not None else "") or ""
        payload_raw = payload_raw.strip()
        full_raw = full_raw.strip()

        if not label:
            return None

        # Full frame takes precedence when provided.
        if full_raw:
            try:
                full_norm, is_query = parse_visca_full_frame_hex(full_raw) or (None, None)
            except Exception:
                full_norm, is_query = (None, None)
            if full_norm:
                return label, str(full_norm), bool(is_query)
            # If the optional full-frame column is incomplete/invalid while editing,
            # fall back to the payload column (if present).

        if not payload_raw:
            return None
        try:
            payload = sanitize_hex_payload(payload_raw)
        except Exception:
            return None
        return label, payload, False

    def _append_row(self, label: str, payload: str, full_hex: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it_label = QTableWidgetItem(label)
        it_hex = QTableWidgetItem(payload)
        it_full = QTableWidgetItem(full_hex)
        it_label.setToolTip(label)
        it_hex.setToolTip(payload)
        it_full.setToolTip(full_hex)
        it_label.setFlags(it_label.flags() | Qt.ItemIsEditable)
        it_hex.setFlags(it_hex.flags() | Qt.ItemIsEditable)
        it_full.setFlags(it_full.flags() | Qt.ItemIsEditable)
        self.table.setItem(r, 0, it_label)
        self.table.setItem(r, 1, it_hex)
        self.table.setItem(r, 2, it_full)

    def _on_item_changed(self, _item):
        if getattr(self, "_suppress_register", False):
            return
        try:
            # If the optional full-frame column is used, clear the payload column for this row.
            if int(getattr(_item, "column", lambda: -1)()) == 2:
                txt = (_item.text() or "").strip()
                if txt:
                    r0 = int(getattr(_item, "row", lambda: -1)())
                    if r0 >= 0:
                        payload_item = self.table.item(r0, 1)
                        if payload_item is not None and (payload_item.text() or "").strip():
                            self.table.blockSignals(True)
                            try:
                                payload_item.setText("")
                                payload_item.setToolTip("")
                            finally:
                                self.table.blockSignals(False)
        except Exception:
            pass
        try:
            _item.setToolTip((_item.text() or "").strip())
        except Exception:
            pass
        if not callable(getattr(self, "_register_cmd", None)):
            return

        r = self.table.currentRow()
        # itemChanged fires even when currentRow isn't the changed row; use item.row() if available
        try:
            r = int(_item.row())
        except Exception:
            pass

        if r < 0:
            return
        resolved = self._resolve_row_command(r)
        if resolved is None:
            return
        label, payload, is_query = resolved

        try:
            self._register_cmd(label, payload, bool(is_query))
        except Exception:
            pass

    def add_row(self):
        self._append_row("New command", "04 00 02")
        r = self.table.rowCount() - 1
        self.table.setCurrentCell(r, 0)
        self.table.editItem(self.table.item(r, 0))

    def delete_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def send_selected(self):
        r = self.table.currentRow()
        if r < 0:
            return
        resolved = self._resolve_row_command(r)
        if resolved is None:
            return
        label, payload, is_query = resolved
        label = label or "Hex"

        if callable(getattr(self, "_register_cmd", None)):
            try:
                self._register_cmd(label, payload, bool(is_query))
            except Exception:
                pass
        self._send_payload(payload, bool(is_query), f"[Hex list {label}]")

# -----------------------------
# Main window
# -----------------------------


    # -----------------------------
    # Persistence helpers (profiles)
    # -----------------------------
    def get_rows_state(self) -> list[dict[str, str]]:
        """Return list of {label,payload,full_hex} for the table (JSON-serializable)."""
        rows: list[dict[str, str]] = []
        try:
            rc = int(self.table.rowCount())
        except Exception:
            rc = 0
        for r in range(rc):
            it_label = self.table.item(r, 0)
            it_hex = self.table.item(r, 1)
            it_full = self.table.item(r, 2)
            label = (it_label.text() if it_label is not None else "") or ""
            payload_raw = (it_hex.text() if it_hex is not None else "") or ""
            full_raw = (it_full.text() if it_full is not None else "") or ""
            label = label.strip()
            payload_raw = payload_raw.strip()
            full_raw = full_raw.strip()
            if not label and not payload_raw and not full_raw:
                continue
            # store a sanitized payload if possible, otherwise store raw text
            payload = payload_raw
            try:
                if payload_raw:
                    payload = sanitize_hex_payload(payload_raw)
            except Exception:
                payload = payload_raw
            full_hex = full_raw
            try:
                if full_raw:
                    parsed = parse_visca_full_frame_hex(full_raw)
                    if parsed is not None:
                        full_hex = parsed[0]
                    else:
                        full_hex = sanitize_hex_payload(full_raw)
            except Exception:
                full_hex = full_raw
            rows.append({"label": label, "payload": payload, "full_hex": full_hex})
        return rows

    def apply_rows_state(self, rows: list[dict[str, str]] | None) -> None:
        """Replace the table contents with the provided rows."""
        if rows is None:
            return
        try:
            self.table.blockSignals(True)
            self._suppress_register = True
            self.table.setRowCount(0)
            for row in rows:
                try:
                    label = str(row.get("label", "")).strip()
                    payload = str(row.get("payload", "")).strip()
                    full_hex = str(row.get("full_hex", "")).strip()
                except Exception:
                    continue
                if not label and not payload and not full_hex:
                    continue
                self._append_row(label or "Hex", payload, full_hex)
        finally:
            try:
                self._suppress_register = False
                self.table.blockSignals(False)
            except Exception:
                pass

        # Re-register commands into dropdowns
        if callable(getattr(self, "_register_cmd", None)):
            try:
                for row in self.get_rows_state():
                    try:
                        label = str(row.get("label", "")).strip()
                        full_hex = str(row.get("full_hex", "")).strip()
                        payload = str(row.get("payload", "")).strip()
                        if full_hex:
                            parsed = parse_visca_full_frame_hex(full_hex)
                            if parsed is not None:
                                cmd_hex, is_query = parsed
                            else:
                                cmd_hex, is_query = sanitize_hex_payload(full_hex), False
                        else:
                            cmd_hex, is_query = sanitize_hex_payload(payload), False
                        if label and cmd_hex:
                            self._register_cmd(label, cmd_hex, bool(is_query))
                    except Exception:
                        pass
            except Exception:
                pass


class RtmpViewerWidget(QWidget):
    """
    RTMP viewer tab.
    - Auto-follows the currently selected target's IP + RTMP port.
    - Builds URL: rtmp://<ip>:<port>/<path>
    - Uses QtMultimedia if available; otherwise shows a fallback message.
    """
    def __init__(self, mono: QFont, label_font: QFont, parent=None):
        super().__init__(parent)
        self._mono = mono
        self._label_font = label_font
        self._ip: str | None = None
        self._rtmp_port: int | None = None

        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("RTMP preview")
        title.setFont(label_font)
        title.setStyleSheet("color: rgb(235,235,235); font-weight: 600;")
        root.addWidget(title)

        # Controls row
        row = QHBoxLayout()
        row.setSpacing(8)

        self.path_edit = QLineEdit("live")
        self.path_edit.setFont(mono)
        self.path_edit.setPlaceholderText("path (e.g. live)")
        self.path_edit.setStyleSheet(
            "QLineEdit { background: rgb(28,31,36); color: rgb(235,235,235); padding: 6px; }"
        )

        self.play_btn = QPushButton("Play")
        self.play_btn.setFont(mono)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFont(mono)

        row.addWidget(QLabel("Path:"))
        row.addWidget(self.path_edit, stretch=1)
        row.addWidget(self.play_btn)
        row.addWidget(self.stop_btn)
        root.addLayout(row)

        # URL / status
        self.url_label = QLabel("URL: (no target selected)")
        self.url_label.setFont(mono)
        self.url_label.setStyleSheet("color: rgb(200,200,200);")
        self.url_label.setWordWrap(True)
        root.addWidget(self.url_label)

        self.status_label = QLabel("")
        self.status_label.setFont(mono)
        self.status_label.setStyleSheet("color: rgb(200,200,200);")
        root.addWidget(self.status_label)

        # Video area or fallback
        if QMediaPlayer is None or QVideoWidget is None or QUrl is None:
            msg = QLabel(
                "QtMultimedia is not available in this environment.\n"
                "Install/enable PySide6 QtMultimedia (+ a backend) to preview RTMP."
            )
            msg.setFont(mono)
            msg.setStyleSheet("color: rgb(200,200,200);")
            msg.setWordWrap(True)
            root.addWidget(msg, stretch=1)
            self._player = None
            self._video = None
        else:
            self._video = QVideoWidget()
            self._video.setStyleSheet("background: rgb(14,16,19); border: 0px;")
            self._video.setMinimumHeight(320)
            root.addWidget(self._video, stretch=1)

            self._player = QMediaPlayer(self)
            # Some platforms require an audio output object even if muted/unused
            try:
                self._audio = QAudioOutput(self)
                self._player.setAudioOutput(self._audio)
            except Exception:
                self._audio = None
            self._player.setVideoOutput(self._video)

        self.setLayout(root)

        # Wiring
        self.path_edit.textChanged.connect(self._refresh_url_label)
        self.play_btn.clicked.connect(self.play)
        self.stop_btn.clicked.connect(self.stop)

        self._refresh_url_label()

    def set_target(self, ip: str | None, rtmp_port: int | None):
        self._ip = (ip or "").strip() or None
        try:
            self._rtmp_port = int(rtmp_port) if rtmp_port is not None else None
        except Exception:
            self._rtmp_port = None
        self._refresh_url_label()

    def _build_url(self) -> str | None:
        if not self._ip or not self._rtmp_port:
            return None
        path = (self.path_edit.text() or "").strip().lstrip("/")
        if not path:
            path = "live"
        return f"rtmp://{self._ip}:{int(self._rtmp_port)}/{path}"

    def _refresh_url_label(self):
        url = self._build_url()
        if url is None:
            if not self._ip:
                self.url_label.setText("URL: (no target selected)")
                self.status_label.setText("Select a target with an RTMP port to preview video.")
            else:
                self.url_label.setText(f"URL: (missing RTMP port for {self._ip})")
                self.status_label.setText("Set RTMP port on the target to enable preview.")
            return
        self.url_label.setText(f"URL: {url}")
        self.status_label.setText("")

    def play(self):
        url = self._build_url()
        if url is None:
            self._refresh_url_label()
            return
        if self._player is None or QUrl is None:
            self.status_label.setText("Playback unavailable (QtMultimedia not loaded).")
            return
        try:
            self._player.setSource(QUrl(url))
            self._player.play()
            self.status_label.setText("Playing…")
        except Exception as e:
            self.status_label.setText(f"Play failed: {type(e).__name__}: {e}")

    def stop(self):
        if self._player is None:
            return
        try:
            self._player.stop()
            self.status_label.setText("Stopped.")
        except Exception:
            pass

class CameraStatusPanel(QGroupBox):
    """Bottom-right status box showing key camera parameters from VISCA inquiries."""

    def __init__(self, mono: QFont, label_font: QFont, parent=None):
        super().__init__("Camera status", parent)
        self.setFont(label_font)
        self.setStyleSheet(
            "QGroupBox { color: rgb(210,210,210); }"
            "QLabel { color: rgb(235,235,235); }"
        )

        self._mono = mono

        form = QFormLayout()
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        def _mk_val(default: str = "—") -> QLabel:
            lbl = QLabel(default)
            lbl.setFont(mono)
            lbl.setStyleSheet("color: rgb(200,200,200);")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return lbl

        # Lens block
        self.zoom_pos = _mk_val()
        self.focus_pos = _mk_val()
        self.focus_mode = _mk_val()

        # Camera block
        self.r_gain = _mk_val()
        self.b_gain = _mk_val()
        self.wb_mode = _mk_val()
        self.aperture = _mk_val()
        self.ae_mode = _mk_val()
        self.backlight_bit = _mk_val()
        self.exposure_comp_bit = _mk_val()
        self.shutter_pos = _mk_val()
        self.iris_pos = _mk_val()
        self.bright_pos = _mk_val()
        self.exposure_comp_pos = _mk_val()

        # Separate inquiries
        self.backlight_mode = _mk_val()
        self.af_sensitivity = _mk_val()

        form.addRow("Zoom position", self.zoom_pos)
        form.addRow("Focus position", self.focus_pos)
        form.addRow("Focus mode", self.focus_mode)
        form.addRow("R_Gain", self.r_gain)
        form.addRow("B_Gain", self.b_gain)
        form.addRow("WB mode", self.wb_mode)
        form.addRow("Aperture", self.aperture)
        form.addRow("AE mode", self.ae_mode)
        form.addRow("Backlight (bit)", self.backlight_bit)
        form.addRow("ExposureComp (bit)", self.exposure_comp_bit)
        form.addRow("Shutter position", self.shutter_pos)
        form.addRow("Iris position", self.iris_pos)
        form.addRow("Bright position", self.bright_pos)
        form.addRow("ExposureComp position", self.exposure_comp_pos)
        form.addRow("Backlight mode", self.backlight_mode)
        form.addRow("AF sensitivity", self.af_sensitivity)

        # Actions
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFont(mono)
        self.refresh_btn.setToolTip("Send inquiry commands and update values")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch(1)
        form.addRow(btn_row)

        self.setLayout(form)

    # --- update helpers
    def set_zoom_focus(self, zoom_u16: int | None, focus_u16: int | None, focus_mode_auto: bool | None):
        if zoom_u16 is None:
            self.zoom_pos.setText("—")
        else:
            self.zoom_pos.setText(f"0x{zoom_u16:04X} ({zoom_u16})")

        if focus_u16 is None:
            self.focus_pos.setText("—")
        else:
            self.focus_pos.setText(f"0x{focus_u16:04X} ({focus_u16})")

        if focus_mode_auto is None:
            self.focus_mode.setText("—")
        else:
            self.focus_mode.setText("Auto" if focus_mode_auto else "Manual")

    def set_camera_block(
        self,
        r_gain: int | None,
        b_gain: int | None,
        wb_mode: int | None,
        aperture: int | None,
        ae_mode: int | None,
        backlight_bit: bool | None,
        exposure_comp_bit: bool | None,
        shutter: int | None,
        iris: int | None,
        bright: int | None,
        exp_comp_pos: int | None,
    ):
        self.r_gain.setText("—" if r_gain is None else f"0x{r_gain:02X} ({r_gain})")
        self.b_gain.setText("—" if b_gain is None else f"0x{b_gain:02X} ({b_gain})")
        self.wb_mode.setText("—" if wb_mode is None else f"0x{wb_mode:01X} ({wb_mode})")
        self.aperture.setText("—" if aperture is None else f"0x{aperture:01X} ({aperture})")
        self.ae_mode.setText("—" if ae_mode is None else f"0x{ae_mode:02X} ({ae_mode})")

        if backlight_bit is None:
            self.backlight_bit.setText("—")
        else:
            self.backlight_bit.setText("On" if backlight_bit else "Off")

        if exposure_comp_bit is None:
            self.exposure_comp_bit.setText("—")
        else:
            self.exposure_comp_bit.setText("On" if exposure_comp_bit else "Off")

        self.shutter_pos.setText("—" if shutter is None else f"0x{shutter:02X} ({shutter})")
        self.iris_pos.setText("—" if iris is None else f"0x{iris:02X} ({iris})")
        self.bright_pos.setText("—" if bright is None else f"0x{bright:02X} ({bright})")
        self.exposure_comp_pos.setText("—" if exp_comp_pos is None else f"0x{exp_comp_pos:02X} ({exp_comp_pos})")

    def set_backlight_mode(self, on: bool | None):
        if on is None:
            self.backlight_mode.setText("—")
        else:
            self.backlight_mode.setText("On" if on else "Off")

    def set_af_sensitivity(self, level: str | None):
        self.af_sensitivity.setText(level or "—")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1500, 900)
        self.setWindowTitle("VISCA-over-IP (PTZoptics)")

        # Saved VISCA targets (IP/port slots)
        self._targets: dict[int, dict[str, object]] = {}
        self._targets_max_slots = 16
        self._targets_path = os.path.join(os.path.dirname(__file__), "visca_targets.json")
        self._load_targets()

        # -----------------------------
        # Profiles (save/load all UI/controller settings)
        # -----------------------------
        # Profiles directory (must be writable in packaged builds)
        _env_profiles = (os.environ.get("PTZ_PROFILES_DIR") or "").strip()
        if _env_profiles:
            self._profiles_dir = os.path.abspath(_env_profiles)
        else:
            self._profiles_dir = os.path.join(_ptz_user_data_dir(), "profiles")
        try:
            os.makedirs(self._profiles_dir, exist_ok=True)
        except Exception:
            pass
        self._profiles_meta_path = os.path.join(self._profiles_dir, "_meta.json")
        self._current_profile: str | None = None
        # pending state to apply once controller/mapping widgets exist
        self._pending_profile_state: dict[str, object] | None = None

        # Debounce repeated identical VISCA commands (prevents rapid duplicate sends)
        self._last_send: dict[tuple[str,int], tuple[str,float]] = {}
        self._send_debounce_sec = 0.05  # 50ms

        # Rate limit VISCA sends (Hz). 0 disables rate limiting.
        self._send_rate_limit_hz: int = 30
        self._send_rate_limit_sec: float = 1.0 / self._send_rate_limit_hz
        self._last_any_send_t: dict[tuple[str,int], float] = {}
        self._rate_pending: dict[tuple[str,int], tuple[str,str,bool]] = {}
        self._rate_timers: dict[tuple[str,int], QTimer] = {}

        # Targets UX state
        self._active_target_slot: int | None = None
        self._new_target_slot: int | None = None

        # Cached camera
        self._cam: Camera | None = None
        self._cam_ip: str | None = None
        self._cam_port: int | None = None

        # Controller state
        self._controller_ok = False
        self._active_joystick_id: int | None = None
        self._active_joystick = None
        self._active_controller_name: str | None = None
        self._axis_count = 0
        self._button_count = 0
        self._hat_count = 0
        # Per-controller button/hat assignments cached by a stable controller signature.
        self._button_mappings_by_controller: dict[str, object] = {}

        self.logbuf = LogBuffer(capacity=200)

        # User-registered commands (from the Hex list tab) to expose in mapping dropdowns
        self.user_commands: dict[str, tuple[str, bool]] = {}

        # Fonts
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)

        label_font = QFont()
        label_font.setPointSize(10)

        mono = QFont()
        mono.setPointSize(11)
        self._label_font = label_font
        self._mono_font = mono
        # --- Button indicator styles (for "B0", "B1", ...)
        self._btn_style_idle = (
            "color: rgb(230,230,230);"
            "background: transparent;"
            "border: 1px solid rgba(255,255,255,30);"
            "border-radius: 6px;"
            "padding: 2px 8px;"
        )
        self._btn_style_pressed = (
            "color: rgb(20,22,26);"
            "background: rgb(240,208,100);"  # warm yellow
            "border: 1px solid rgba(0,0,0,120);"
            "border-radius: 6px;"
            "padding: 2px 8px;"
        )


        # -----------------------------
        # LEFT: Controller panel
        # -----------------------------
        self.controller_title = QLabel("Controller Mapping")
        self.controller_title.setVisible(False)

        self.controller_status = QLabel("Controller backend: (not initialised)")
        self.controller_status.setFont(label_font)

        self.controller_combo = QComboBox()
        self.controller_combo.setFont(mono)
        self.controller_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.refresh_controllers_btn = QPushButton("Refresh controllers")
        self.refresh_controllers_btn.setFont(mono)
        self.refresh_controllers_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Max VISCA send rate limiter (Hz)
        self.rate_limit_spin = QSpinBox()
        self.rate_limit_spin.setFont(mono)
        self.rate_limit_spin.setRange(0, 120)
        self.rate_limit_spin.setValue(self._send_rate_limit_hz)
        self.rate_limit_spin.setSuffix(" Hz")
        self.rate_limit_spin.setToolTip("Limits non-query VISCA sends per target. 0 = unlimited. Stop commands are not limited.")
        self.rate_limit_lbl = QLabel("Send rate")
        self.rate_limit_lbl.setFont(label_font)
        # -----------------------------
        # Button mapping UI (dynamic)
        # -----------------------------
        self.button_map_combos: list[QComboBox] = []
        self.button_map_labels: list[QLabel] = []
        self.hat_map_indices: list[tuple[int, int]] = []  # (hat_index, dir_idx)

        mapping_box = QGroupBox("")
        mapping_box.setFont(label_font)

        self.mapping_grid = QGridLayout()
        self.mapping_grid.setHorizontalSpacing(10)
        self.mapping_grid.setVerticalSpacing(6)
        mapping_box.setLayout(self.mapping_grid)

        # Placeholder until a controller is selected
        self._rebuild_button_mapping_ui(0, 0, label_font=label_font, mono=mono)
        # --- Response curve editor (collapsible)
        # --- Tabs: Response curve + Virtual controller
        tabs_box = QGroupBox()
        tabs_box.setFont(label_font)
        tabs_box.setStyleSheet("QGroupBox { border: 0px; }")

        self.left_tabs = QTabWidget()
        self.left_tabs.setFont(label_font)
        self.left_tabs.setStyleSheet(
            "QTabWidget::pane { border: 0px; }"
            "QTabBar::tab { background: rgb(28,31,36); color: rgb(235,235,235); "
            "padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px; }"
            "QTabBar::tab:selected { background: rgb(14,16,19); font-weight: 700; }"
        )


        # Tab 1: Response curve editor
        self.curve_editor = ResponseCurveEditor(label_font=label_font, mono=mono)
        self.left_tabs.addTab(self.curve_editor, "Response curve")

        # Tab 2: RTMP viewer (follows selected target IP + RTMP port)
        self.rtmp_viewer = RtmpViewerWidget(mono=mono, label_font=label_font)
        self.left_tabs.addTab(self.rtmp_viewer, "RTMP")

        # Tab 3: Virtual pad/buttons
        self.virtual_controller = VirtualControllerWidget(
            mono=mono,
            label_font=label_font,
            send_payload_cb=self.send_payload,
        )
        self.virtual_controller.set_command_provider(self.get_all_commands)
        self.left_tabs.addTab(self.virtual_controller, "Virtual pad")
        # Tab 4: Editable hex list
        self.hex_list_tab = HexCommandsListWidget(
            mono=mono,
            label_font=label_font,
            send_payload_cb=self.send_payload,
            register_cmd_cb=self.register_user_command,
        )
        self.left_tabs.addTab(self.hex_list_tab, "Hex list")


        tabs_layout = QVBoxLayout()
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.addWidget(self.left_tabs)
        tabs_box.setLayout(tabs_layout)

        # curve_box = QGroupBox()
        # curve_box.setFont(label_font)
        # curve_box.setStyleSheet("QGroupBox { border: 0px; }")

        # self.curve_toggle = QToolButton()
        # self.curve_toggle.setText("Response curve editor")
        # self.curve_toggle.setCheckable(True)
        # self.curve_toggle.setChecked(True)
        # self.curve_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # self.curve_toggle.setArrowType(Qt.DownArrow)
        # self.curve_toggle.setStyleSheet(
        #     "QToolButton { color: rgb(235,235,235); font-weight: 600; padding: 6px; }"
        # )

        # self.curve_editor = ResponseCurveEditor(label_font=label_font, mono=mono)

        # def _on_curve_toggled(checked: bool):
        #     self.curve_editor.setVisible(checked)
        #     self.curve_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        # self.curve_toggle.toggled.connect(_on_curve_toggled)

        # curve_layout = QVBoxLayout()
        # curve_layout.setContentsMargins(0, 0, 0, 0)
        # curve_layout.addWidget(self.curve_toggle)
        # curve_layout.addWidget(self.curve_editor)
        # curve_box.setLayout(curve_layout)

        # -----------------------------
        # Profile UI (compact bar - placed next to Log toggle)
        # -----------------------------
        self.profile_combo = QComboBox()
        self.profile_combo.setFont(mono)
        self.profile_combo.setEditable(True)
        self.profile_combo.setInsertPolicy(QComboBox.NoInsert)
        self.profile_combo.setToolTip("Profile name (saved under ./profiles/)")
        self.profile_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.profile_combo.setMinimumWidth(170)

        self.profile_load_btn = QPushButton("Load")
        self.profile_load_btn.setFont(mono)
        self.profile_save_btn = QPushButton("Save")
        self.profile_save_btn.setFont(mono)
        self.profile_saveas_btn = QPushButton("Save as…")
        self.profile_saveas_btn.setFont(mono)
        self.profile_delete_btn = QPushButton("Delete")
        self.profile_delete_btn.setFont(mono)

        # Keep the profile controls as small as possible (inline bar)
        for _b in (self.profile_load_btn, self.profile_save_btn, self.profile_saveas_btn, self.profile_delete_btn):
            try:
                _b.setFixedHeight(24)
                _b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            except Exception:
                pass
        try:
            self.profile_load_btn.setMaximumWidth(60)
            self.profile_save_btn.setMaximumWidth(60)
            self.profile_saveas_btn.setMaximumWidth(90)
            self.profile_delete_btn.setMaximumWidth(70)
        except Exception:
            pass

        self._refresh_profile_list()

        # Inline profiles bar widget (added next to Log toggle)
        self.profiles_bar = QWidget()
        _pbar = QHBoxLayout()
        _pbar.setContentsMargins(0, 0, 0, 0)
        _pbar.setSpacing(6)
        _pbar.addWidget(self.profile_combo)
        _pbar.addWidget(self.profile_load_btn)
        _pbar.addWidget(self.profile_save_btn)
        _pbar.addWidget(self.profile_saveas_btn)
        _pbar.addWidget(self.profile_delete_btn)
        self.profiles_bar.setLayout(_pbar)
        self.profiles_bar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.controller_title)


        left_top_row = QHBoxLayout()
        left_top_row.addWidget(QLabel("Controller"))
        left_top_row.addWidget(self.controller_combo, stretch=1)
        left_top_row.addWidget(self.refresh_controllers_btn)
        left_top_row.addWidget(self.rate_limit_lbl)
        left_top_row.addWidget(self.rate_limit_spin)
        left_layout.addLayout(left_top_row)
        left_layout.addWidget(self.controller_status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setLayout(QVBoxLayout())
        scroll_content.layout().addWidget(mapping_box)
        scroll_content.layout().addWidget(tabs_box)
        scroll_content.layout().addStretch(1)
        scroll.setWidget(scroll_content)

        left_layout.addWidget(scroll, stretch=3)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # -----------------------------
        # RIGHT: VISCA UI
        # -----------------------------
        title = QLabel("VISCA-over-IP Command Sender")
        title.setVisible(False)
        title.setFont(title_font)

        # Hidden current target fields (used by send_payload/parse_ip_port)
        self.ip_edit = QLineEdit("")
        self.ip_edit.setFont(mono)
        self.ip_edit.setVisible(False)
        self.port_edit = QLineEdit("")
        self.port_edit.setFont(mono)
        self.port_edit.setVisible(False)

        # ip_lbl = QLabel("IP")
        # ip_lbl.setFont(label_font)
        # self.ip_edit = QLineEdit("192.168.0.100")
        # self.ip_edit.setFont(mono)
        # self.ip_edit.setPlaceholderText("IP address")

        # port_lbl = QLabel("VISCA Port (UDP)")
        # port_lbl.setFont(label_font)
        # self.port_edit = QLineEdit("52381")
        # self.port_edit.setFont(mono)
        # self.port_edit.setPlaceholderText("VISCA Port (UDP)")
        # self.port_edit.setMaximumWidth(120)

        preset_lbl = QLabel("Preset command (auto-send)")
        preset_lbl.setFont(label_font)
        self.preset_combo = QComboBox()
        self.preset_combo.setFont(mono)
        for label, payload, is_query in VISCA_COMMANDS:
            self.preset_combo.addItem(label, (payload, is_query))

        manual_lbl = QLabel("Manual hex")
        manual_lbl.setFont(label_font)
        self.cmd_edit = QLineEdit("04 00 02")
        self.cmd_edit.setFont(mono)
        self.cmd_edit.setPlaceholderText("Command hex")

        self.send_btn = QPushButton("Send Hex  (Enter)")
        self.send_btn.setFont(mono)
        self.send_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)


        # --- Collapsible log header (toggle)
        self.log_toggle = QToolButton()
        self.log_toggle.setText("Log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(False)
        self.log_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.log_toggle.setArrowType(Qt.DownArrow)
        self.log_toggle.setStyleSheet(
            "QToolButton { color: rgb(235,235,235); font-weight: 600; padding: 6px; }"
        )

        def _on_log_toggled(checked: bool):
            self.log_view.setVisible(checked)
            self.log_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            
        _on_log_toggled(False)
        self.log_toggle.toggled.connect(_on_log_toggled)

                # --- Header (left) + Targets column (right)
        # We build the main VISCA controls as a left column, and a targets column on the far right.
        visca_layout = QVBoxLayout()
        visca_layout.addWidget(title)

        # Targets column: discrete box per slot (hidden when empty), in a scroll area.
        self.targets_scroll = QScrollArea()
        self.targets_scroll.setWidgetResizable(True)
        self.targets_scroll.setFrameShape(QFrame.NoFrame)
        self.targets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.targets_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.targets_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.targets_container = QWidget()
        self.targets_container_layout = QVBoxLayout()
        self.targets_container_layout.setContentsMargins(6, 6, 6, 6)
        self.targets_container_layout.setSpacing(8)
        self.targets_container.setLayout(self.targets_container_layout)
        self.targets_scroll.setWidget(self.targets_container)
        self.targets_container.adjustSize()



        self.target_boxes = {}  # slot -> dict of widgets
        # Per-slot debounce timers for autosave (typing-based)
        self._target_autosave_timers: dict[int, QTimer] = {}

        for slot in range(1, self._targets_max_slots + 1):
            box = QGroupBox(f"Target #{slot}")
            box.setFont(label_font)
            box.setStyleSheet("QGroupBox { color: rgb(210,210,210); }")
            ip_edit = QLineEdit()
            ip_edit.setFont(mono)
            ip_edit.setPlaceholderText("IP")
            ip_edit.setMinimumWidth(170)

            # VISCA port (UDP)
            port_edit = QLineEdit()
            port_edit.setFont(mono)
            port_edit.setPlaceholderText("port")
            port_edit.setMaximumWidth(40)

            # RTMP port (stored per target; not used by VISCA sender)
            rtmp_edit = QLineEdit()
            rtmp_edit.setFont(mono)
            rtmp_edit.setPlaceholderText("port")
            rtmp_edit.setMaximumWidth(40)

            # Small red bin button (delete)
            bin_btn = QToolButton()
            bin_btn.setText("")
            bin_btn.setFont(mono)
            bin_btn.setToolTip("Delete target")
            bin_btn.setFixedSize(30, 30)
            try:
                icon = self.style().standardIcon(QStyle.SP_TrashIcon)
                if icon is not None and not icon.isNull():
                    bin_btn.setIcon(icon)
                else:
                    bin_btn.setText("X")
            except Exception:
                bin_btn.setText("X")
            bin_btn.setStyleSheet(
                "QToolButton {"
                "  color: rgb(235,235,235);"
                "  background: rgb(70,75,85);"
                "  border: 1px solid rgba(255,255,255,30);"
                "  border-radius: 8px;"
                "  font-weight: 900;"
                "}"
                "QToolButton:hover { background: rgb(90,95,105); }"
                "QToolButton:pressed { background: rgb(55,60,70); }"
            )
            # Layout inside box
            row1 = QHBoxLayout()
            row1.addWidget(QLabel("IP"))
            ip_edit.setFixedWidth(80)  
            row1.addWidget(ip_edit, stretch=1)
            row1.addWidget(QLabel("Port"))
            port_edit.setFixedWidth(50)  
            row1.addWidget(port_edit)
            row1.addSpacing(6)

            # Add (+) button (add another target)
            add_btn = QToolButton()
            add_btn.setText("+")
            add_btn.setFont(mono)
            add_btn.setToolTip("Add a new target")
            add_btn.setFixedSize(34, 34)
            add_btn.setStyleSheet(
                "QToolButton {"
                "  color: rgb(20,22,26);"
                "  background: rgb(240,208,100);"
                "  border: 1px solid rgba(0,0,0,120);"
                "  border-radius: 10px;"
                "  font-weight: 900;"
                "}"
                "QToolButton:hover { background: rgb(255,225,120); }"
                "QToolButton:pressed { background: rgb(210,180,90); }"
            )

            # Layout inside box (single row; + aligned with delete)
            row1 = QHBoxLayout()
            row1.addWidget(QLabel("IP"))
            row1.addWidget(ip_edit, stretch=1)
            row1.addWidget(QLabel("VISCA"))
            row1.addWidget(port_edit)
            row1.addWidget(QLabel("RTMP"))
            row1.addWidget(rtmp_edit)
            row1.addStretch(1)
            row1.addWidget(add_btn)
            row1.addWidget(bin_btn)
            box.setLayout(row1)

            # Click-to-select: capture clicks anywhere in the box (including children)
            box.setProperty("slot", slot)
            box.installEventFilter(self)
            ip_edit.setProperty("slot", slot)
            port_edit.setProperty("slot", slot)
            rtmp_edit.setProperty("slot", slot)
            bin_btn.setProperty("slot", slot)
            add_btn.setProperty("slot", slot)

            # Autosave debounce timer
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda s=slot: self._autosave_target_slot(int(s)))
            self._target_autosave_timers[slot] = t

            # Autosave on typing (text input)
            ip_edit.textChanged.connect(lambda _txt, s=slot: self._schedule_target_autosave(int(s)))
            port_edit.textChanged.connect(lambda _txt, s=slot: self._schedule_target_autosave(int(s)))
            rtmp_edit.textChanged.connect(lambda _txt, s=slot: self._schedule_target_autosave(int(s)))

            # Delete (bin) / Add (+)
            bin_btn.clicked.connect(lambda _=False, s=slot: self._on_target_box_clear(int(s)))
            add_btn.clicked.connect(self._on_add_target_clicked)

            self.targets_container_layout.addWidget(box)
            self.target_boxes[slot] = {
                "box": box,
                "ip": ip_edit,
                "port": port_edit,      # VISCA port (UDP)
                "rtmp": rtmp_edit,      # RTMP port (optional)
                "bin": bin_btn,
                "add": add_btn,
            }

        # Spacer so boxes pack to top
        self.targets_container_layout.addStretch(1)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # grid.addWidget(ip_lbl, 0, 0)
        # grid.addWidget(port_lbl, 0, 1)

        # grid.addWidget(self.ip_edit, 1, 0)
        # grid.addWidget(self.port_edit, 1, 1)

        # grid.addWidget(preset_lbl, 2, 0, 1, 2)
        # grid.addWidget(self.preset_combo, 3, 0, 1, 2)

        # manual_row = QHBoxLayout()
        # manual_left = QVBoxLayout()
        # # manual_left.addWidget(manual_lbl)
        # manual_left.addWidget(self.cmd_edit)

        # manual_row.addLayout(manual_left)
        # manual_row.addWidget(self.send_btn, alignment=Qt.AlignBottom)

        visca_layout.addLayout(grid)
        # Assemble right panel as two columns: main controls (left) + targets (right)
        visca_widget = QWidget()
        visca_widget.setLayout(visca_layout)

        # --- Place VISCA sender + log on the LEFT (controller/curve/log), keep Targets on the RIGHT
        left_layout.addWidget(visca_widget)

        # Log header row: Log toggle + compact Profiles bar (minimal space)
        _log_header = QHBoxLayout()
        _log_header.setContentsMargins(0, 0, 0, 0)
        _log_header.setSpacing(8)
        _log_header.addWidget(self.log_toggle)
        try:
            _log_header.addWidget(self.profiles_bar)
        except Exception:
            pass
        _log_header.addStretch(1)
        left_layout.addLayout(_log_header)

        left_layout.addWidget(self.log_view, stretch=1)
        # left_layout.addLayout(manual_row)

        targets_widget = QWidget()
        targets_col = QVBoxLayout()
        targets_col.setContentsMargins(0, 0, 0, 0)
        targets_col.setSpacing(6)

        targets_col.addWidget(self.targets_scroll, stretch=1)

        # Bottom-right status panel (camera inquiry decode)
        self.status_panel = CameraStatusPanel(mono=mono, label_font=label_font)
        self.status_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        targets_col.addWidget(self.status_panel, stretch=0)
        targets_widget.setLayout(targets_col)
        targets_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)


        right_panel_layout = QHBoxLayout()
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(12)
        right_panel_layout.addWidget(targets_widget)
        right_panel = QWidget()
        right_panel.setLayout(right_panel_layout)

        # -----------------------------
        # Root: Split view (50/50)
        # -----------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setHandleWidth(1)
        try:
            # Disable user resizing between columns
            splitter.handle(1).setDisabled(True)
        except Exception:
            pass

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        self.setCentralWidget(splitter)

        # ---- Behavior wiring
        self.send_btn.clicked.connect(self.send_current)
        self.cmd_edit.returnPressed.connect(self.send_current)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)
        self.rate_limit_spin.valueChanged.connect(self._on_rate_limit_changed)
        # Profiles wiring
        try:
            self.profile_load_btn.clicked.connect(self._on_profile_load_clicked)
            self.profile_save_btn.clicked.connect(self._on_profile_save_clicked)
            self.profile_saveas_btn.clicked.connect(self._on_profile_save_as_clicked)
            self.profile_delete_btn.clicked.connect(self._on_profile_delete_clicked)
        except Exception:
            pass
# Status panel wiring
        try:
            self.status_panel.refresh_btn.clicked.connect(self.refresh_camera_status)
        except Exception:
            pass
        # Targets wiring
        self._refresh_targets_ui()
        # Ensure at least one editable target box (with +) is visible at startup
        # when no targets are saved/active yet.
        if not self._targets and self._active_target_slot is None and self._new_target_slot is None:
            self._new_target_slot = 1
            self._refresh_targets_ui()

        # Controller wiring
        self.refresh_controllers_btn.clicked.connect(self.refresh_controllers)
        self.controller_combo.currentIndexChanged.connect(self.on_controller_selected)

        self.add_log("Select a command from dropdown (auto-sends), or type hex and press Enter / SEND.")
        self.init_controllers()
        # Load the previously-used profile (if any)
        try:
            self._load_last_profile_on_startup()
        except Exception:
            pass

        # Debounce for preset scrolling
        self._preset_debounce = QTimer(self)
        self._preset_debounce.setSingleShot(True)
        self._preset_debounce.timeout.connect(self._send_selected_preset)
        self._pending_preset: tuple[str, str, bool] | None = None

        # Poll controller axes at 60Hz for live point
        self._axis_timer = QTimer(self)
        self._axis_timer.timeout.connect(self._poll_controller_for_curve)
        self._axis_timer.start(16)
        self._last_input_states: list[int] = []

        # --- Continuous PTZ intent state (to avoid spamming identical commands)
        # Pan/Tilt are sent as a single VISCA command, so we track both.
        self._pt_intent = {
            "pan_speed": 0,
            "tilt_speed": 0,
            "pan_dir": 0x03,   # 0x01=left, 0x02=right, 0x03=stop
            "tilt_dir": 0x03,  # 0x01=up,   0x02=down,  0x03=stop
        }
        # Zoom/Focus track the last sent direction+speed byte (e.g. 0x2p / 0x3p / 0x00)
        self._zf_intent = {
            "zoom": 0x00,
            "focus": 0x00,
        }

        # --- Layout sizing: expand window to fit content (avoid scroll areas when possible)
        QTimer.singleShot(0, self._fit_to_contents)

    def closeEvent(self, event):
        try:
            if self._cam is not None:
                self._cam.close_connection()
        except Exception:
            pass

        try:
            if pygame is not None and self._controller_ok:
                pygame.joystick.quit()
                pygame.quit()
        except Exception:
            pass

        super().closeEvent(event)


    def _fit_to_contents(self):
        """Resize the main window to fit all widgets without scrolling when possible."""
        try:
            # Compute preferred size from the central widget.
            cw = self.centralWidget()
            if cw is None:
                return
            hint = cw.sizeHint()

            # Add a little headroom for window frame + spacing.
            target_w = int(hint.width() + 60)
            target_h = int(hint.height() + 80)

            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                # If it doesn't fit, maximize; otherwise resize to the hint.
                if target_w > avail.width() or target_h > avail.height():
                    self.showMaximized()
                    return
            self.resize(max(self.width()-300, target_w), max(self.height(), target_h))
            self.setMinimumSize(min(target_w, 1200), min(target_h, 800))
        except Exception:
            pass


    # -----------------------------
    # Targets (IP/port slots)
    # -----------------------------
    def _load_targets(self):
        """Load saved targets from JSON on startup."""
        self._targets = {}
        try:
            if os.path.exists(self._targets_path):
                with open(self._targets_path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                # stored as {"1": {"ip": "...", "port": 52381}, ...}
                for k, v in raw.items():
                    try:
                        slot = int(k)
                        ip = str(v.get("ip", "")).strip()
                        port = int(v.get("port", 0))
                        rtmp_raw = v.get("rtmp_port", v.get("rtmp", None))
                        rtmp_port = None
                        try:
                            if rtmp_raw not in (None, "", 0, "0"):
                                rtmp_port = int(rtmp_raw)
                                if not (1 <= rtmp_port <= 65535):
                                    rtmp_port = None
                        except Exception:
                            rtmp_port = None
                        if 1 <= slot <= self._targets_max_slots and ip and 1 <= port <= 65535:
                            entry = {"ip": ip, "port": port}
                            if rtmp_port is not None:
                                entry["rtmp_port"] = int(rtmp_port)
                            self._targets[slot] = entry
                    except Exception:
                        continue
        except Exception:
            # non-fatal; start empty
            self._targets = {}

    def _save_targets(self):
        try:
            out = {}
            for k, v in sorted(self._targets.items()):
                item = {"ip": v["ip"], "port": int(v["port"])}
                rp = v.get("rtmp_port")
                if rp is not None and str(rp).strip() != "":
                    try:
                        item["rtmp_port"] = int(rp)
                    except Exception:
                        pass
                out[str(k)] = item
            with open(self._targets_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            self.add_log(f"ERROR: Failed to save targets: {type(e).__name__}: {e}")

    def _format_target_label(self, slot: int) -> str:
        v = self._targets.get(slot)
        if not v:
            return f"#{slot} (empty)"
        base = f"#{slot} {v['ip']}:{int(v['port'])}"
        if v.get('rtmp_port') is not None:
            try:
                base += f"  | RTMP {int(v['rtmp_port'])}"
            except Exception:
                pass
        return base

    def resizeEvent(self, event):
        """
        Enforce 1:1 aspect ratio for entire window.
        Always keeps window square while resizing.
        """
        # s = min(self.width(), self.height())
        # self.resize(s, s)
        return super().resizeEvent(event)

    def eventFilter(self, obj, event):
        """Click-to-select on targets (group box *and* its child widgets).

        - Clicking anywhere inside a saved target (box, IP field, Port field, buttons) makes it active.
        - Clicking inside the visible "new target" editor attempts autosave; if saved, it becomes active.
        """
        try:
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                slot = obj.property("slot")
                if slot is None:
                    return super().eventFilter(obj, event)
                slot = int(slot)

                # Saved target: select it
                if slot in getattr(self, "_targets", {}):
                    self._active_target_slot = int(slot)
                    self._new_target_slot = None
                    self._apply_target_slot(slot)
                    self._set_selected_target_ui(slot)
                    self._refresh_targets_ui()
                    # Allow the click to continue (e.g., line edit focus)
                    return False

                # New target editor: try to autosave, then select if saved
                if getattr(self, "_new_target_slot", None) is not None and slot == int(self._new_target_slot):
                    self._autosave_target_slot(slot, force=True)
                    if slot in getattr(self, "_targets", {}):
                        self._active_target_slot = int(slot)
                        self._new_target_slot = None
                        self._apply_target_slot(slot)
                        self._set_selected_target_ui(slot)
                        self._refresh_targets_ui()
                    return False
        except Exception:
            pass

        return super().eventFilter(obj, event)




    def _valid_ip_port(self, ip: str, port_text: str, rtmp_text: str = "") -> tuple[bool, int, int | None]:
        ip = (ip or "").strip()
        if not ip:
            return (False, 0, None)

        # VISCA port (UDP) is required
        try:
            port = int((port_text or "").strip())
        except Exception:
            return (False, 0, None)
        if not (1 <= port <= 65535):
            return (False, 0, None)

        # RTMP port is optional, but if provided it must be valid
        rtmp_port: int | None = None
        rtxt = (rtmp_text or "").strip()
        if rtxt:
            try:
                rtmp_port = int(rtxt)
            except Exception:
                return (False, 0, None)
            if not (1 <= rtmp_port <= 65535):
                return (False, 0, None)

        return (True, port, rtmp_port)

    def _schedule_target_autosave(self, slot: int):
        """Debounce autosave while typing."""
        try:
            t = getattr(self, "_target_autosave_timers", {}).get(int(slot))
            if t is None:
                return
            # keep the UI responsive; modest debounce
            t.start(350)
        except Exception:
            pass

    def _autosave_target_slot(self, slot: int, force: bool = False):
        """Autosave slot when inputs are valid.

        Policy:
          - Creation is allowed ONLY for the slot currently opened via the (+) button (self._new_target_slot).
          - Updates are allowed for the active (in-use) slot, or when force=True (e.g., click-to-select).
        """
        slot = int(slot)
        w = getattr(self, "target_boxes", {}).get(slot)
        if not w:
            return

        ip = w["ip"].text()
        port_text = w["port"].text()
        rtmp_text = ""
        try:
            rtmp_text = w.get("rtmp").text() if w.get("rtmp") is not None else ""
        except Exception:
            rtmp_text = ""

        ok, port, rtmp_port = self._valid_ip_port(ip, port_text, rtmp_text)
        if not ok:
            return

        ip = ip.strip()
        new_entry = {"ip": ip, "port": int(port)}
        if rtmp_port is not None:
            new_entry["rtmp_port"] = int(rtmp_port)

        is_create = (
            (slot not in self._targets)
            and (getattr(self, "_new_target_slot", None) is not None)
            and (slot == int(self._new_target_slot))
        )

        is_update = (slot in self._targets) and (
            force
            or (
                (getattr(self, "_active_target_slot", None) is not None)
                and (slot == int(self._active_target_slot))
            )
        )

        if not (is_create or is_update):
            return

        prev = self._targets.get(slot)
        if prev == new_entry:
            return

        self._targets[slot] = new_entry
        self._save_targets()

        if is_create:
            msg = f"{ip}:{port}"
            if new_entry.get("rtmp_port") is not None:
                msg += f" | RTMP {int(new_entry.get('rtmp_port'))}"
            self.add_log(f"Saved target #{slot}: {msg}")
            # New target becomes active immediately; hide editor state
            self._new_target_slot = None
            self._active_target_slot = int(slot)
            self._apply_target_slot(int(slot))
        else:
            msg = f"{ip}:{port}"
            if new_entry.get("rtmp_port") is not None:
                msg += f" | RTMP {int(new_entry.get('rtmp_port'))}"
            self.add_log(f"Updated target #{slot}: {msg}")
            # Keep camera selection in sync if editing active slot
            if getattr(self, "_active_target_slot", None) is not None and int(self._active_target_slot) == slot:
                self._apply_target_slot(int(slot))

        self._refresh_targets_ui()



    
    def _set_selected_target_ui(self, slot: int, flash: bool = False):
        """Highlight the currently selected target box.

        If flash=True, briefly "pops" the selection to make button-driven target switches obvious.
        """
        slot = int(slot)

        # Lazily define styles (keeps edits localized; avoids touching __init__ styling)
        style_idle = "QGroupBox { color: rgb(210,210,210); }"
        style_selected = (
            "QGroupBox { color: rgb(210,210,210); border: 2px solid rgb(240,208,100); border-radius: 8px; }"
        )
        style_flash = (
            "QGroupBox { color: rgb(210,210,210); border: 3px solid rgb(240,208,100); border-radius: 8px; "
            "background: rgba(240,208,100,35); }"
        )

        for s, w in getattr(self, "target_boxes", {}).items():
            if not w["box"].isVisible():
                continue
            if s == slot:
                w["box"].setStyleSheet(style_flash if flash else style_selected)
            else:
                w["box"].setStyleSheet(style_idle)

        if flash:
            # If the user switches targets quickly, only revert the currently-active slot.
            def _revert():
                try:
                    if getattr(self, "_active_target_slot", None) is not None and int(self._active_target_slot) == slot:
                        for s, w in getattr(self, "target_boxes", {}).items():
                            if not w["box"].isVisible():
                                continue
                            if s == slot:
                                w["box"].setStyleSheet(style_selected)
                                break
                except Exception:
                    pass

            QTimer.singleShot(250, _revert)


    def _apply_target_slot(self, slot: int):
        """Select a saved slot as the current active target."""
        slot = int(slot)
        entry = self._targets.get(slot)
        if not entry:
            self.add_log(f"ERROR: Target #{slot} is empty")
            return

        self._active_target_slot = int(slot)
        self._new_target_slot = None

        ip = str(entry.get("ip", "")).strip()
        port = int(entry.get("port", 0))
        rtmp_port = entry.get("rtmp_port", None)

        self.ip_edit.setText(ip)
        self.port_edit.setText(str(port))
        # Keep RTMP port available for future features
        self._active_rtmp_port = rtmp_port
        # Update RTMP viewer tab (if present)
        try:
            if hasattr(self, "rtmp_viewer") and self.rtmp_viewer is not None:
                self.rtmp_viewer.set_target(ip, rtmp_port)
        except Exception:
            pass

        # Reset cached camera so next send uses the new target
        self._cam_ip = None
        self._cam_port = None

        msg = f"{ip}:{port}"
        if rtmp_port is not None:
            try:
                msg += f" | RTMP {int(rtmp_port)}"
            except Exception:
                pass
        self.add_log(f"Selected target #{slot}: {msg}")

        # Visual feedback: ensure the selected target box highlights even when switched via button mappings.
        try:
            if hasattr(self, "target_boxes"):
                self._set_selected_target_ui(int(slot), flash=True)
        except Exception:
            pass


    def _refresh_targets_ui(self):
        """Refresh the Targets column UI.

        UX policy:
          - Show all saved targets (persisted slots).
          - Show the "new target" editor slot only after user clicks (+), until it autosaves or is canceled.
          - Hide empty slots.
        """
        if not hasattr(self, "target_boxes"):
            return

        # Bootstrap visibility so the Targets column never starts empty:
        # - If saved targets exist but none is active, auto-select the first one.
        # - If no saved targets exist and no new slot is open, open slot #1 as "+ New target".
        if getattr(self, "_active_target_slot", None) is None and getattr(self, "_new_target_slot", None) is None:
            if getattr(self, "_targets", None):
                try:
                    first_slot = sorted(self._targets.keys())[0]
                    self._active_target_slot = int(first_slot)
                    self._apply_target_slot(int(first_slot))
                except Exception:
                    pass
            else:
                self._new_target_slot = 1

        active = getattr(self, "_active_target_slot", None)
        new_slot = getattr(self, "_new_target_slot", None)

        for slot, w in self.target_boxes.items():
            entry = self._targets.get(slot)

            should_show = False
            title = f"Target #{slot}"

            # Show all saved targets
            if entry:
                should_show = True
                ip = str(entry.get("ip", "")).strip()
                port = int(entry.get("port", 0))

                rtmp_port = entry.get("rtmp_port", None)

                # Do not clobber while user is typing in this slot
                rtmp_has_focus = False
                try:
                    rtmp_has_focus = bool(w.get("rtmp") is not None and w["rtmp"].hasFocus())
                except Exception:
                    rtmp_has_focus = False

                if not w["ip"].hasFocus() and not w["port"].hasFocus() and not rtmp_has_focus:
                    w["ip"].setText(ip)
                    w["port"].setText(str(port))
                    try:
                        if w.get("rtmp") is not None:
                            w["rtmp"].setText("" if rtmp_port is None else str(int(rtmp_port)))
                    except Exception:
                        pass

                title = f"Target #{slot}   {ip}:{port}" + (f" | RTMP {int(rtmp_port)}" if rtmp_port is not None else "")

            # Also show the unsaved "new target" editor slot (opened via +)
            elif (new_slot is not None) and slot == int(new_slot) and slot not in self._targets:
                should_show = True
                title = f"+ New target (#{slot})"

            # else: hide empty slots
            w["box"].setVisible(bool(should_show))
            w["box"].setTitle(title)

        # Highlight active slot if any
        if active is not None:
            try:
                self._set_selected_target_ui(int(active))
            except Exception:
                pass

        # Update mapping combos (labels include slot details)
        self._refresh_target_actions_in_mapping_combos()


    def _on_add_target_clicked(self):
        """Show a new editable target box (next available slot)."""
        # If already creating a new one, just focus it.
        if getattr(self, "_new_target_slot", None) is not None and int(self._new_target_slot) not in self._targets:
            w = self.target_boxes.get(int(self._new_target_slot))
            if w:
                w["box"].setVisible(True)
                w["ip"].setFocus()
            return

        # Find next empty slot
        slot = None
        for s in range(1, self._targets_max_slots + 1):
            if s not in self._targets:
                slot = s
                break

        if slot is None:
            self.add_log("No free target slots available.")
            return

        self._new_target_slot = int(slot)

        # Clear fields for clean entry
        w = self.target_boxes.get(int(slot))
        if w:
            w["ip"].setText("")
            w["port"].setText("")
            try:
                if w.get("rtmp") is not None:
                    w["rtmp"].setText("")
            except Exception:
                pass
        self._refresh_targets_ui()

        if w:
            w["ip"].setFocus()

    def _on_target_box_clear(self, slot: int):
        slot = int(slot)

        # If user is editing a new unsaved slot, allow canceling it
        if getattr(self, "_new_target_slot", None) is not None and slot == int(self._new_target_slot) and slot not in self._targets:
            self._new_target_slot = None
            self._refresh_targets_ui()
            return

        if slot in self._targets:
            old = self._targets.pop(slot)
            self._save_targets()

            if getattr(self, "_active_target_slot", None) is not None and slot == int(self._active_target_slot):
                self._active_target_slot = None

            self._refresh_targets_ui()
            self.add_log(f"Cleared target #{slot} (was {old.get('ip','')}:{old.get('port','')})")
        else:
            self.add_log(f"Target #{slot} already empty")


    def _refresh_target_actions_in_mapping_combos(self):
        """Update the 'Switch Target #n' items in all button mapping combos."""
        # NOTE: We don't want to clear/rebuild all combos in a way that loses user selections.
        # Instead, we rebuild each combo, preserving currentData.
        for cb in getattr(self, "button_map_combos", []):
            self._repopulate_mapping_combo_preserve(cb)

    def _repopulate_mapping_combo_preserve(self, cb: QComboBox):
        cur = cb.currentData()
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("Unassigned", None)

        # Built-in + user commands (Hex list commands are registered into self.user_commands)
        for label, payload, is_query in self.get_all_commands():
            cb.addItem(label, ("visca", payload, bool(is_query), label))

        # Target switches (only show saved / activated targets)
        cb.insertSeparator(cb.count())
        saved_slots = sorted(int(s) for s in self._targets.keys()) if getattr(self, "_targets", None) else []
        if not saved_slots:
            cb.addItem("(no saved targets)", ("target", None))
        else:
            for slot in saved_slots:
                v = self._targets.get(slot)
                if not v:
                    continue
                detail = f" ({v['ip']}:{int(v['port'])})"
                if v.get('rtmp_port') is not None:
                    try:
                        detail += f" | RTMP {int(v['rtmp_port'])}"
                    except Exception:
                        pass
                cb.addItem(f"Switch Target #{slot}{detail}", ("target", slot))

        # restore selection if possible
        if cur is not None:
            idx = cb.findData(cur)
            if idx >= 0:
                cb.setCurrentIndex(idx)

        cb.blockSignals(False)

    # -----------------------------
    # Controllers (left panel)
    # -----------------------------
    def init_controllers(self):
        if pygame is None:
            self.controller_status.setText("Controller backend: pygame not installed")
            self.controller_combo.clear()
            self.controller_combo.addItem("(pygame not available)")
            self.controller_combo.setEnabled(False)
            self.refresh_controllers_btn.setEnabled(True)
            return

        # Startup should be tolerant when no controller is connected (or SDL joystick backend
        # is not ready yet). Prefer a usable "no controllers" state over a hard init failure.
        try:
            pygame.init()
        except Exception:
            pass
        try:
            pygame.joystick.init()
        except Exception:
            pass

        self._controller_ok = True
        self._last_joystick_count = -1
        self.controller_status.setText("Controller backend: pygame OK")
        self.refresh_controllers_btn.setEnabled(True)

        try:
            self.refresh_controllers()
        except Exception:
            # Final guard: never leave startup stuck in a hard-error state.
            self._controller_ok = False
            self.controller_combo.clear()
            self.controller_combo.addItem("(no controllers detected)")
            self.controller_combo.setEnabled(False)
            self.controller_status.setText("Controllers detected: 0")

    def refresh_controllers(self, preserve_selection: bool = True):
        if pygame is None:
            return
        if not self._controller_ok:
            # Allow manual refresh to recover from a startup init failure (e.g., controller plugged in later).
            try:
                pygame.init()
                pygame.joystick.init()
                self._controller_ok = True
                self.controller_status.setText("Controller backend: pygame OK")
            except Exception as e:
                self._controller_ok = False
                self.controller_status.setText(f"Controller backend: ERROR ({type(e).__name__}) {e}")
                try:
                    self.controller_combo.blockSignals(True)
                    self.controller_combo.clear()
                    self.controller_combo.addItem("(controller init failed)")
                    self.controller_combo.setEnabled(False)
                finally:
                    try:
                        self.controller_combo.blockSignals(False)
                    except Exception:
                        pass
                return
        # Snapshot current selection (joystick id) so we can preserve it across refreshes.
        prev_jid = self.controller_combo.currentData() if preserve_selection else None

        try:
            try:
                pygame.event.pump()
            except Exception:
                # Some environments can enumerate joysticks even if event pump is unavailable.
                pass
            count = pygame.joystick.get_count()
        except Exception as e:
            # Retry joystick subsystem init once; if it still fails, degrade gracefully to
            # "no controllers" instead of a hard error on startup.
            try:
                pygame.joystick.init()
                count = int(pygame.joystick.get_count())
                self._controller_ok = True
            except Exception:
                self._controller_ok = False
                self._last_joystick_count = 0
                self.controller_combo.blockSignals(True)
                try:
                    self.controller_combo.clear()
                    self.controller_combo.addItem("(no controllers detected)", None)
                    self.controller_combo.setEnabled(False)
                finally:
                    self.controller_combo.blockSignals(False)
                self.controller_status.setText("Controllers detected: 0")
                self._snapshot_active_button_mapping()
                self._active_joystick_id = None
                self._active_joystick = None
                self._active_controller_name = None
                self._axis_count = self._button_count = self._hat_count = 0
                self._rebuild_button_mapping_ui(0, 0, label_font=self._label_font, mono=self._mono_font)
                self._sync_curve_axis_combo()
                return
        # Track device count for hot-plug detection
        self._last_joystick_count = int(count)

        self.controller_combo.blockSignals(True)
        self.controller_combo.clear()

        if count <= 0:
            self.controller_combo.addItem("(no controllers detected)", None)
            self.controller_combo.setEnabled(False)
            self.controller_status.setText("Controllers detected: 0")
        else:
            self.controller_combo.setEnabled(True)
            for jid in range(count):
                try:
                    js = pygame.joystick.Joystick(jid)
                    js.init()
                    name = js.get_name()
                    js.quit()
                except Exception:
                    name = f"Controller {jid}"
                self.controller_combo.addItem(f"{jid}: {name}", jid)
            self.controller_status.setText(f"Controllers detected: {count}")

        self.controller_combo.blockSignals(False)

        # Restore selection if possible; otherwise pick first controller.
        if count > 0:
            target_idx = -1
            if prev_jid is not None:
                try:
                    target_idx = self.controller_combo.findData(int(prev_jid))
                except Exception:
                    target_idx = self.controller_combo.findData(prev_jid)

            if target_idx < 0:
                target_idx = 0

            self.controller_combo.setCurrentIndex(target_idx)
            # Ensure internal joystick object + mapping UI are updated.
            self.on_controller_selected(target_idx)
        else:
            # Clear mapping UI when nothing is connected
            self._snapshot_active_button_mapping()
            self._active_joystick_id = None
            self._active_joystick = None
            self._active_controller_name = None
            self._axis_count = self._button_count = self._hat_count = 0
            self._rebuild_button_mapping_ui(0, 0, label_font=self._label_font, mono=self._mono_font)
            self._sync_curve_axis_combo()


    
    def _on_rate_limit_changed(self, hz: int) -> None:
        """Update VISCA send rate limiter (Hz). 0 disables rate limiting."""
        try:
            hz_i = int(hz)
        except Exception:
            hz_i = 0
        self._send_rate_limit_hz = max(0, hz_i)
        self._send_rate_limit_sec = (1.0 / self._send_rate_limit_hz) if self._send_rate_limit_hz > 0 else 0.0

    def on_controller_selected(self, idx: int):
        if pygame is None or not self._controller_ok:
            return

        jid = self.controller_combo.currentData()
        if jid is None:
            self._snapshot_active_button_mapping()
            self._active_joystick_id = None
            self._active_joystick = None
            self._active_controller_name = None
            self._axis_count = self._button_count = self._hat_count = 0
            self._rebuild_button_mapping_ui(0, 0, label_font=self._label_font, mono=self._mono_font)
            self._sync_curve_axis_combo()
            # Apply any pending profile controller mappings/curves now that the UI exists
            try:
                self._apply_pending_controller_profile_state()
            except Exception:
                pass
            return

        try:
            self._snapshot_active_button_mapping()
            if self._active_joystick is not None:
                try:
                    self._active_joystick.quit()
                except Exception:
                    pass

            js = pygame.joystick.Joystick(int(jid))
            js.init()
            try:
                pygame.event.pump()
            except Exception:
                pass

            self._active_joystick_id = int(jid)
            self._active_joystick = js
            try:
                self._active_controller_name = str(js.get_name())
            except Exception:
                self._active_controller_name = f"Controller {int(jid)}"
            self._axis_count = int(js.get_numaxes())
            self._button_count = js.get_numbuttons()
            self._hat_count = js.get_numhats()

            self._rebuild_button_mapping_ui(self._button_count, self._hat_count, label_font=self._label_font, mono=self._mono_font)

            self.controller_status.setText(
                f"Selected: {js.get_name()} | axes={self._axis_count}, buttons={self._button_count}, hats={self._hat_count}"
            )
            self._sync_curve_axis_combo()
            # Apply any pending profile controller mappings/curves now that mapping UI exists
            try:
                self._apply_pending_controller_profile_state()
            except Exception:
                pass
        except Exception as e:
            self.controller_status.setText(f"Controller select ERROR ({type(e).__name__}) {e}")
            self._active_controller_name = None
            self._sync_curve_axis_combo()
            # Apply any pending profile controller mappings/curves now that mapping UI exists
            try:
                self._apply_pending_controller_profile_state()
            except Exception:
                pass

    def _sync_curve_axis_combo(self):
        if QChart is None:
            return
        axis_count = int(self._axis_count) if int(self._axis_count) > 0 else 1
        self.curve_editor.set_axis_count(axis_count)

    def _rebuild_button_mapping_ui(self, button_count: int, hat_count: int, label_font: QFont, mono: QFont):
        # Clear existing widgets in grid
        while self.mapping_grid.count():
            item = self.mapping_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        self.button_map_combos.clear()
        self.button_map_labels.clear()
        self.hat_map_indices.clear()

        total_inputs = int(button_count) + int(hat_count) * 4

        if total_inputs <= 0:
            msg = QLabel("No controller buttons / hats detected.")
            msg.setFont(mono)
            self.mapping_grid.addWidget(msg, 0, 0)
            return

        # Layout as columns of (Label + Combo)
        # Each input uses one row; we pack multiple columns to reduce vertical scrolling.
        cols = 3
        rows = (total_inputs + cols - 1) // cols

        # Header row for each column
        for c in range(cols):
            hdr = QLabel("Input → Command")
            hdr.setVisible(False)
            hdr.setFont(label_font)
            hdr.setStyleSheet("color: rgb(200, 200, 200);")
            self.mapping_grid.addWidget(hdr, 0, c * 2, 1, 2)

        HAT_DIRS = [
            ("↑", (0, 1)),
            ("↓", (0, -1)),
            ("←", (-1, 0)),
            ("→", (1, 0)),
        ]

        # Build ordered list of inputs: physical buttons first, then hat directions.
        inputs: list[tuple[str, tuple[str, int, int] | None]] = []
        for b in range(int(button_count)):
            inputs.append((f"B{b}", ("button", b, -1)))
        for h in range(int(hat_count)):
            for d_idx, (arrow, _) in enumerate(HAT_DIRS):
                inputs.append((f"H{h}{arrow}", ("hat", h, d_idx)))

        for idx, (name, meta) in enumerate(inputs):
            col = idx // rows
            row = idx % rows
            grid_row = row

            lbl = QLabel(name)
            lbl.setFont(mono)
            lbl.setStyleSheet(self._btn_style_idle)

            cb = QComboBox()
            cb.setFont(mono)
            cb.setFixedHeight(24)
            cb.setMinimumWidth(150)
            self._repopulate_mapping_combo_preserve(cb)

            # Nice dark styling for combos
            cb.setStyleSheet(
                "QComboBox { background: rgb(28,31,36); color: rgb(235,235,235); padding: 2px 6px; font-size: 10pt; }"
                "QComboBox::drop-down { width: 18px; border: 0px; }"
                "QComboBox QAbstractItemView { background: rgb(24,26,30); color: rgb(235,235,235); }"
            )

            self.button_map_labels.append(lbl)
            self.button_map_combos.append(cb)

            # Track hat indices (in the same order they appear in the UI)
            if meta is not None and meta[0] == "hat":
                self.hat_map_indices.append((meta[1], meta[2]))

            self.mapping_grid.addWidget(lbl, grid_row, col * 2)
            self.mapping_grid.addWidget(cb, grid_row, col * 2 + 1)
# stretch at bottom
        self.mapping_grid.setRowStretch(rows + 1, 1)

    def get_button_mapping(self) -> dict[int, object]:
        """Return {input_index: mapping_data} for assigned controller inputs."""
        mapping: dict[int, object] = {}
        for i, cb in enumerate(self.button_map_combos):
            data = cb.currentData()
            if data is None:
                continue
            mapping[i] = data
        return mapping

    def _controller_mapping_key(self) -> str | None:
        """Stable key for current controller capabilities, used for per-controller profile mappings."""
        try:
            name = str(getattr(self, "_active_controller_name", None) or "").strip()
        except Exception:
            name = ""
        if not name:
            return None
        try:
            a = int(getattr(self, "_axis_count", 0) or 0)
            b = int(getattr(self, "_button_count", 0) or 0)
            h = int(getattr(self, "_hat_count", 0) or 0)
        except Exception:
            a = b = h = 0
        return f"{name}|a{a}|b{b}|h{h}"

    def _snapshot_active_button_mapping(self) -> None:
        """Cache current button mapping under the active controller key (if any)."""
        try:
            key = self._controller_mapping_key()
            if not key:
                return
            mapping = self.get_button_mapping()
            self._button_mappings_by_controller[key] = self._jsonable(mapping)
        except Exception:
            pass

    # -----------------------------
    # Continuous PTZ from axes
    # -----------------------------
    @staticmethod
    def _clamp_u8(v: int) -> int:
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(0, min(255, v))

    @staticmethod
    def _clamp_nibble(v: int) -> int:
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(0, min(0x0F, v))

    @staticmethod
    def _axis_phys_to_curve_raw(params: 'AxisCurveParams', raw_phys: float) -> float:
        """Match ResponseCurveEditor's interpretation of a physical [-1..1] axis."""
        try:
            x = float(raw_phys)
        except Exception:
            x = 0.0
        x = max(-1.0, min(1.0, x))
        if bool(getattr(params, 'zero_based', False)):
            # physical [-1..1] -> [0..1]
            x = (x + 1.0) * 0.5
            x = max(0.0, min(1.0, x))
        return float(x)

    @classmethod
    def _signed_speed_and_dir(cls, shaped: float, speed: int, *, pos_dir: int, neg_dir: int, stop_dir: int) -> tuple[int, int]:
        """Return (dir_byte, speed_u8)."""
        try:
            shaped_f = float(shaped)
        except Exception:
            shaped_f = 0.0
        spd = cls._clamp_u8(speed)
        if spd <= 0 or abs(shaped_f) <= 1e-9:
            return int(stop_dir), 0
        return (int(pos_dir) if shaped_f > 0 else int(neg_dir)), spd

    def _build_pan_tilt_payload(self, pan_speed: int, tilt_speed: int, pan_dir: int, tilt_dir: int) -> str:
        """VISCA Pan-Tilt Drive body (no header/terminator).

        The rest of the app (via Camera._send_command) is expected to wrap
        commands as: 81 01 <payload> FF.

        If we returned a fully framed command here, it would be double-framed
        (e.g., 81 01 81 01 ... FF FF) and many cameras will reject it.
        """
        vv = self._clamp_u8(pan_speed)
        ww = self._clamp_u8(tilt_speed)
        xx = self._clamp_u8(pan_dir)
        yy = self._clamp_u8(tilt_dir)
        return f"06 01 {vv:02X} {ww:02X} {xx:02X} {yy:02X}"

    def _build_zoom_payload(self, signed_speed: int) -> str:
        """VISCA Zoom body (no header/terminator).

        Payload is: 04 07 pp
        where pp is 0x2p (tele), 0x3p (wide), or 0x00 (stop).
        """
        v = int(signed_speed)
        if v == 0:
            return "04 07 00"
        # App-level zoom/focus speed factor is 1..8; VISCA nibble p is 0..7.
        p = max(0, min(7, abs(v) - 1))
        b = (0x20 | p) if v > 0 else (0x30 | p)
        return f"04 07 {b:02X}"

    def _build_focus_payload(self, signed_speed: int) -> str:
        """VISCA Focus body (no header/terminator).

        Payload is: 04 08 pp
        where pp is 0x2p (far), 0x3p (near), or 0x00 (stop).
        """
        v = int(signed_speed)
        if v == 0:
            return "04 08 00"
        # App-level zoom/focus speed factor is 1..8; VISCA nibble p is 0..7.
        p = max(0, min(7, abs(v) - 1))
        b = (0x20 | p) if v > 0 else (0x30 | p)
        return f"04 08 {b:02X}"

    def _update_continuous_ptz_from_axes(self, js):
        """Read all mapped axes and send Pan/Tilt/Zoom/Focus commands based on binned speed."""
        if js is None or not hasattr(self, 'curve_editor') or self.curve_editor is None:
            return

        # Collect latest intent per logical control.
        intent: dict[str, tuple[float, int]] = {}  # control -> (shaped, speed)
        try:
            axis_map = getattr(self.curve_editor, 'axis_control_map', {}) or {}
            params_list = getattr(self.curve_editor, 'axis_params', []) or []
            n_axes = int(js.get_numaxes())
        except Exception:
            return

        for ax_i in range(min(n_axes, len(params_list))):
            ctrl = axis_map.get(ax_i)
            if ctrl not in ("Pan", "Tilt", "Zoom", "Focus"):
                continue
            try:
                raw_phys = float(js.get_axis(ax_i))
            except Exception:
                raw_phys = 0.0
            p = params_list[ax_i]
            raw_curve = self._axis_phys_to_curve_raw(p, raw_phys)
            shaped, spd = p.shaped_with_bins(raw_curve)
            shaped_f = float(shaped)
            spd_i = int(spd)

            prev = intent.get(ctrl)
            if prev is None:
                intent[ctrl] = (shaped_f, spd_i)
            else:
                prev_shaped, prev_spd = prev
                # Allow multiple physical axes to map to the same logical control.
                # Use the strongest active input so a resting duplicate axis doesn't overwrite
                # another axis that is currently driving motion.
                if (abs(shaped_f) > abs(float(prev_shaped))) or (
                    abs(shaped_f) == abs(float(prev_shaped)) and abs(spd_i) > abs(int(prev_spd))
                ):
                    intent[ctrl] = (shaped_f, spd_i)

        # --- Pan
        pan_shaped, pan_spd = intent.get("Pan", (0.0, 0))
        # For pan: + => right (0x02), - => left (0x01)
        pan_dir, pan_speed = self._signed_speed_and_dir(pan_shaped, pan_spd, pos_dir=0x02, neg_dir=0x01, stop_dir=0x03)

        # --- Tilt
        tilt_shaped, tilt_spd = intent.get("Tilt", (0.0, 0))
        # For tilt: + => up (0x01), - => down (0x02)
        tilt_dir, tilt_speed = self._signed_speed_and_dir(tilt_shaped, tilt_spd, pos_dir=0x01, neg_dir=0x02, stop_dir=0x03)

        # Send Pan/Tilt only when any component changes.
        try:
            last = getattr(self, '_pt_intent', None) or {}
            changed = (
                int(last.get('pan_speed', -1)) != int(pan_speed)
                or int(last.get('tilt_speed', -1)) != int(tilt_speed)
                or int(last.get('pan_dir', -1)) != int(pan_dir)
                or int(last.get('tilt_dir', -1)) != int(tilt_dir)
            )
            if changed:
                payload = self._build_pan_tilt_payload(pan_speed, tilt_speed, pan_dir, tilt_dir)
                self.send_payload(payload, False, label="[PT]")
                self._pt_intent = {
                    "pan_speed": int(pan_speed),
                    "tilt_speed": int(tilt_speed),
                    "pan_dir": int(pan_dir),
                    "tilt_dir": int(tilt_dir),
                }
        except Exception:
            pass

        # --- Zoom
        zoom_shaped, zoom_spd = intent.get("Zoom", (0.0, 0))
        zoom_signed_speed = 0
        if int(zoom_spd) > 0 and abs(float(zoom_shaped)) > 1e-9:
            zoom_signed_speed = int(zoom_spd) if float(zoom_shaped) > 0 else -int(zoom_spd)
        zoom_payload_byte = 0x00 if zoom_signed_speed == 0 else ((0x20 | self._clamp_nibble(abs(zoom_signed_speed))) if zoom_signed_speed > 0 else (0x30 | self._clamp_nibble(abs(zoom_signed_speed))))

        try:
            last_b = int((getattr(self, '_zf_intent', {}) or {}).get('zoom', 0x00))
            if int(zoom_payload_byte) != last_b:
                self.send_payload(self._build_zoom_payload(zoom_signed_speed), False, label="[Zoom]")
                self._zf_intent["zoom"] = int(zoom_payload_byte)
        except Exception:
            pass

        # --- Focus
        focus_shaped, focus_spd = intent.get("Focus", (0.0, 0))
        focus_signed_speed = 0
        if int(focus_spd) > 0 and abs(float(focus_shaped)) > 1e-9:
            focus_signed_speed = int(focus_spd) if float(focus_shaped) > 0 else -int(focus_spd)
        focus_payload_byte = 0x00 if focus_signed_speed == 0 else ((0x20 | self._clamp_nibble(abs(focus_signed_speed))) if focus_signed_speed > 0 else (0x30 | self._clamp_nibble(abs(focus_signed_speed))))

        try:
            last_b = int((getattr(self, '_zf_intent', {}) or {}).get('focus', 0x00))
            if int(focus_payload_byte) != last_b:
                self.send_payload(self._build_focus_payload(focus_signed_speed), False, label="[Focus]")
                self._zf_intent["focus"] = int(focus_payload_byte)
        except Exception:
            pass

    def _poll_controller_for_curve(self):
        if pygame is None:
            return
        # Throttled background probe so controller status/list updates automatically
        # when a device is plugged in after startup.
        try:
            _now = float(_time.time())
        except Exception:
            _now = 0.0
        try:
            _next_probe = float(getattr(self, "_next_controller_auto_probe_t", 0.0) or 0.0)
        except Exception:
            _next_probe = 0.0

        if (not self._controller_ok) or (self._active_joystick is None):
            if _now >= _next_probe:
                try:
                    self._next_controller_auto_probe_t = _now + 1.0
                    self.refresh_controllers(preserve_selection=True)
                except Exception:
                    pass
            if not self._controller_ok:
                return

        # --- hot-plug detection (device added/removed)
        try:
            # Consume events so JOYDEVICEADDED / JOYDEVICEREMOVED are observed promptly.
            changed = False
            try:
                evs = pygame.event.get()
            except Exception:
                evs = []
            if evs:
                added = getattr(pygame, "JOYDEVICEADDED", None)
                removed = getattr(pygame, "JOYDEVICEREMOVED", None)
                for ev in evs:
                    if added is not None and ev.type == added:
                        changed = True
                        break
                    if removed is not None and ev.type == removed:
                        changed = True
                        break

            # Fallback: detect by count change (covers older pygame or backends without device events)
            try:
                pygame.event.pump()
                count = int(pygame.joystick.get_count())
            except Exception:
                count = int(getattr(self, "_last_joystick_count", 0) or 0)

            last = int(getattr(self, "_last_joystick_count", -1))
            if changed or (last >= 0 and count != last):
                # Refresh list and try to keep user's current selection.
                self.refresh_controllers(preserve_selection=True)
        except Exception:
            pass

        # If no active joystick selected, nothing else to poll (but hot-plug detection above still runs).
        if self._active_joystick is None:
            return
        try:
            pygame.event.pump()
            js = self._active_joystick
            try:
                n_axes_live = int(js.get_numaxes())
            except Exception:
                n_axes_live = int(getattr(self, "_axis_count", 0) or 0)

            # Some controllers/backends report capabilities only after the first event pump.
            # If the live axis count differs, refresh the curve editor axis list immediately.
            if n_axes_live != int(getattr(self, "_axis_count", 0) or 0):
                self._axis_count = int(n_axes_live)
                self._sync_curve_axis_combo()

            # --- live curve point (selected axis)
            n = n_axes_live
            ax = int(self.curve_editor.get_active_axis()) if hasattr(self.curve_editor, 'get_active_axis') else int(self.curve_editor.selected_axis)
            raw = float(js.get_axis(ax)) if ax < n else 0.0
            self.curve_editor.set_controller_value(raw)

            # --- continuous PTZ control from assigned axes (Pan/Tilt/Zoom/Focus)
            # Uses each axis' curve params + optional quantized bins to derive speed.
            self._update_continuous_ptz_from_axes(js)

            bcount = js.get_numbuttons()
            hat_virtual_count = len(self.hat_map_indices)
            total_inputs = int(bcount) + int(hat_virtual_count)

            # Track previous input states to fire on rising edge
            if len(self._last_input_states) != total_inputs:
                self._last_input_states = [0] * total_inputs

            # --- physical buttons
            for i in range(min(bcount, len(self.button_map_labels))):
                pressed = 1 if int(js.get_button(i)) else 0
                prev = self._last_input_states[i]
                self._last_input_states[i] = pressed

                # visual highlight always
                self._set_button_label_pressed(i, bool(pressed))

                # rising edge -> trigger mapping
                if pressed and not prev:
                    self._handle_mapped_input(i)

            # --- hats (treated as virtual buttons after the physical buttons)
            for idx, (hat_i, dir_idx) in enumerate(self.hat_map_indices):
                hx, hy = js.get_hat(hat_i)

                pressed_hat = False
                if dir_idx == 0 and hy > 0:
                    pressed_hat = True   # ↑
                elif dir_idx == 1 and hy < 0:
                    pressed_hat = True   # ↓
                elif dir_idx == 2 and hx < 0:
                    pressed_hat = True   # ←
                elif dir_idx == 3 and hx > 0:
                    pressed_hat = True   # →

                label_index = bcount + idx
                pressed = 1 if pressed_hat else 0
                prev = self._last_input_states[label_index]
                self._last_input_states[label_index] = pressed

                self._set_button_label_pressed(label_index, pressed_hat)

                if pressed and not prev:
                    self.add_log(f"DEBUG: controller input fired: H{hat_i} dir={dir_idx} (virtual idx={label_index})")
                    self._handle_mapped_input(label_index)

        except Exception as e:
            self.add_log(f"ERROR: controller poll failed: {type(e).__name__}: {e}")
            pass


    def _handle_mapped_input(self, input_index: int):
        """Execute the mapping assigned to a controller input (button or hat direction)."""
        if input_index < 0 or input_index >= len(self.button_map_combos):
            self.add_log(f"DEBUG: mapping skipped: input_index={input_index} out of range (combos={len(self.button_map_combos)})")
            return

        cb = self.button_map_combos[input_index]
        data = cb.currentData()
        if data is None:
            self.add_log(f"DEBUG: mapping skipped: input_index={input_index} is Unassigned")
            return

        input_name = ""
        if 0 <= input_index < len(self.button_map_labels):
            try:
                input_name = self.button_map_labels[input_index].text()
            except Exception:
                input_name = ""

        try:
            # Expected format from _repopulate_mapping_combo_preserve:
            #   ("visca", payload, is_query, label)
            if isinstance(data, (list, tuple)) and len(data) >= 1 and data[0] == "visca":
                _, payload, is_query, label = data
                tag = f"[{label}]"
                if input_name:
                    tag = f"{tag} ({input_name})"
                self.send_payload(str(payload), bool(is_query), tag)
                return

            # Target switch:
            #   ("target", slot)
            if isinstance(data, (list, tuple)) and len(data) >= 2 and data[0] == "target":
                slot = data[1]
                if slot is None:
                    self.add_log(f"DEBUG: target mapping selected but no slot available (input_index={input_index})")
                    return
                self._apply_target_slot(int(slot))
                return

            # Compatibility: if some combos store (payload, is_query, label) like the Virtual pad
            if isinstance(data, (list, tuple)) and len(data) == 3 and isinstance(data[0], str):
                payload, is_query, label = data
                tag = f"[{label}]"
                if input_name:
                    tag = f"{tag} ({input_name})"
                self.send_payload(str(payload), bool(is_query), tag)
                return

            # Compatibility: (payload, is_query)
            if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[0], str):
                payload, is_query = data
                tag = "[Mapped]"
                if input_name:
                    tag = f"{tag} ({input_name})"
                self.send_payload(str(payload), bool(is_query), tag)
                return

            self.add_log(f"DEBUG: mapping data unrecognized at input_index={input_index}: {data!r}")

        except Exception as e:
            self.add_log(f"ERROR: Mapping trigger failed: {type(e).__name__}: {e}")

    def _set_button_label_pressed(self, button_index: int, pressed: bool):
        """Visually highlight the button label when input is detected."""
        if button_index < 0 or button_index >= len(self.button_map_labels):
            return
        lbl = self.button_map_labels[button_index]
        lbl.setStyleSheet(self._btn_style_pressed if pressed else self._btn_style_idle)
    # -----------------------------
    # Logging
    # -----------------------------
    def add_log(self, msg: str):
        line = self.logbuf.add(msg)
        self.log_view.appendPlainText(line)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # -----------------------------
    # Camera lifecycle
    # -----------------------------
    def get_camera(self, ip: str, port: int) -> Camera:
        if self._cam is None or self._cam_ip != ip or self._cam_port != port:
            if self._cam is not None:
                try:
                    self._cam.close_connection()
                except Exception:
                    pass
            self._cam = Camera(ip, port)
            self._cam_ip, self._cam_port = ip, port
            self.add_log(f"Camera socket ready for {ip}:{port}")
        return self._cam

    def parse_ip_port(self) -> tuple[str, int]:
        ip = self.ip_edit.text().strip()
        if not ip:
            raise ValueError("IP is empty")

        try:
            port = int(self.port_edit.text().strip())
        except ValueError:
            raise ValueError("VISCA port (UDP) must be an integer")

        if not (1 <= port <= 65535):
            raise ValueError("VISCA port (UDP) must be 1..65535")

        return ip, port


    # -----------------------------
    # Dynamic command catalog (for mappings + virtual pad)
    # -----------------------------
    def get_all_commands(self):
        """Return combined preset + user commands as (label, payload, is_query)."""
        # De-duplicate by label while preserving order:
        # built-ins first, then user commands override same-label entries in place.
        out: list[tuple[str, str, bool]] = []
        by_label: dict[str, int] = {}

        for lbl, payload, is_query in VISCA_COMMANDS:
            key = str(lbl)
            by_label[key] = len(out)
            out.append((key, str(payload), bool(is_query)))

        for lbl, (payload, is_query) in (self.user_commands or {}).items():
            key = str(lbl)
            item = (key, str(payload), bool(is_query))
            idx = by_label.get(key)
            if idx is None:
                by_label[key] = len(out)
                out.append(item)
            else:
                out[idx] = item

        return out

    def register_user_command(self, label: str, payload: str, is_query: bool = False):
        """Register/update a user command and refresh mapping dropdowns."""
        label = (label or "").strip()
        payload = (payload or "").strip()
        if not label or not payload:
            return
        self.user_commands[label] = (payload, bool(is_query))

        # Refresh virtual pad dropdowns to include the new command immediately.
        try:
            if hasattr(self, "virtual_controller") and self.virtual_controller is not None:
                self.virtual_controller.refresh_command_lists()
        except Exception:
            pass
        # Refresh physical button mapping dropdowns too.
        try:
            self._refresh_target_actions_in_mapping_combos()
        except Exception:
            pass

    # -----------------------------
    # Sending logic
    # -----------------------------
    def send_payload(self, payload: str, is_query: bool, label: str = "", _force: bool = False):
        try:
            ip, port = self.parse_ip_port()
        except ValueError as e:
            self.add_log(f"ERROR: {e}")
            return

        try:
            cam = self.get_camera(ip, port)
            payload_display = str(payload).strip()
            payload_for_send = payload_display
            decode_payload = payload_display
            effective_is_query = bool(is_query)

            full_frame_parts: list[str] | None = None
            try:
                parsed_full = parse_visca_full_frame_hex(payload_display)
            except Exception:
                parsed_full = None
            if parsed_full is not None:
                payload_for_send, inferred_query = parsed_full
                effective_is_query = bool(inferred_query)
                full_frame_parts = [p for p in payload_for_send.split() if p]
                if len(full_frame_parts) >= 4:
                    decode_payload = " ".join(full_frame_parts[2:-1]).upper()

            key = (ip, port)
            now = _time.time()

            # --- Rate limiting (non-query), with coalescing
            # Stop commands must never be blocked (otherwise the camera can keep moving).
            stop_payloads = {
                "06 01 00 00 03 03",  # PT Stop
                "04 07 00",           # Zoom Stop
                "04 08 00",           # Focus Stop
            }
            stop_full_frames = {
                "81 01 06 01 00 00 03 03 FF",
                "81 01 04 07 00 FF",
                "81 01 04 08 00 FF",
            }
            normalized = payload_for_send.strip().upper()

            if (not _force) and (not effective_is_query) and (normalized not in stop_payloads) and (normalized not in stop_full_frames) and self._send_rate_limit_sec > 0:
                last_any = self._last_any_send_t.get(key)
                if last_any is not None:
                    elapsed = now - last_any
                    if elapsed < self._send_rate_limit_sec:
                        # Avoid delayed duplicate sends: if the most recently sent payload is
                        # identical to what we're about to queue, don't queue it again.
                        last = self._last_send.get(key)
                        if last is not None:
                            try:
                                last_payload, _last_t = last
                                if normalized == last_payload:
                                    return
                            except Exception:
                                pass

                        # Coalesce: remember the latest command and schedule a single send at the next slot.
                        self._rate_pending[key] = (label, payload_for_send, effective_is_query)
                        remaining = max(0.0, self._send_rate_limit_sec - elapsed)

                        t = self._rate_timers.get(key)
                        if t is None:
                            t = QTimer(self)
                            t.setSingleShot(True)

                            def _fire(k=key):
                                pending = self._rate_pending.pop(k, None)
                                if pending is None:
                                    return
                                plabel, ppayload, pis_query = pending
                                # Drop stale duplicates right before forced send.
                                try:
                                    norm2 = ppayload.strip().upper()
                                    last2 = self._last_send.get(k)
                                    if last2 is not None and norm2 == last2[0]:
                                        return
                                except Exception:
                                    pass
                                # Mark send time now and send (bypass rate check to avoid re-queue).
                                self._last_any_send_t[k] = _time.time()
                                self.send_payload(ppayload, pis_query, label=plabel, _force=True)

                            t.timeout.connect(_fire)
                            self._rate_timers[key] = t

                        if not t.isActive():
                            t.start(int(remaining * 1000))
                        return

            # Passed (or bypassed) rate limiter: record send time (non-query only)
            if not effective_is_query:
                self._last_any_send_t[key] = now

                # Debounce repeated identical VISCA commands (drops rapid duplicates)
                last = self._last_send.get(key)
                if last is not None:
                    last_payload, last_t = last
                    if normalized == last_payload and (now - last_t) < self._send_debounce_sec:
                        return
                self._last_send[key] = (normalized, now)

            if full_frame_parts is not None:
                resp = cam._send_visca_frame(payload_for_send)
            else:
                resp = cam._send_command(payload_for_send, query=effective_is_query)

            if effective_is_query:
                if resp is None:
                    self.add_log(f"QUERY {label}: {payload_for_send} | response: <None>")
                else:
                    self.add_log(f"QUERY {label}: {payload_for_send} | response: {resp.hex(' ')}")
                    # Update status panel when we recognize inquiry payloads
                    try:
                        self._handle_inquiry_response(decode_payload, resp)
                    except Exception:
                        pass
            else:
                self.add_log(f"SENT {label}: {payload_for_send}")

        except NoQueryResponse as e:
            self.add_log(f"ERROR: No query response: {e}")
        except ViscaException as e:
            self.add_log(f"ERROR: VISCA error (0x{e.status_code:02X}) {e.description}")
        except Exception as e:
            self.add_log(f"ERROR: {type(e).__name__}: {e}")

    # -----------------------------
    # Inquiry decode -> status panel
    # -----------------------------
    @staticmethod
    def _nibble(b: int) -> int:
        return int(b) & 0x0F

    @classmethod
    def _parse_u16_from_nibbles(cls, bb: bytes, start: int) -> int | None:
        """Parse 4 x 0n bytes (nibbles) into a 16-bit integer."""
        if bb is None or len(bb) < start + 4:
            return None
        n0 = cls._nibble(bb[start])
        n1 = cls._nibble(bb[start + 1])
        n2 = cls._nibble(bb[start + 2])
        n3 = cls._nibble(bb[start + 3])
        return (n0 << 12) | (n1 << 8) | (n2 << 4) | n3

    @classmethod
    def _parse_u8_from_nibbles(cls, bb: bytes, start: int) -> int | None:
        """Parse 2 x 0n bytes (nibbles) into an 8-bit integer."""
        if bb is None or len(bb) < start + 2:
            return None
        hi = cls._nibble(bb[start])
        lo = cls._nibble(bb[start + 1])
        return (hi << 4) | lo

    def _handle_inquiry_response(self, payload: str, resp: bytes):
        """Decode a subset of VISCA inquiries and populate the bottom-right panel."""
        if not hasattr(self, "status_panel") or self.status_panel is None:
            return

        # Normalize payload for matching (strip spaces/punctuation)
        try:
            payload_norm = sanitize_hex_payload(payload).replace(" ", "").lower()
        except Exception:
            payload_norm = (payload or "").replace(" ", "").replace(",", "").lower()

        # All replies we care about are 90 50 ... FF
        if resp is None or len(resp) < 4:
            return

        # 1) CAM_LensBlockInq  (7E 7E 00)
        # Response template (from datasheet image):
        #   90 50 0u 0u 0u 0u 00 00 0v 0v 0v 0v 00 0w 00 FF
        if payload_norm == "7e7e00":
            zoom = self._parse_u16_from_nibbles(resp, 2)
            focus = self._parse_u16_from_nibbles(resp, 8)
            focus_mode_auto = None
            if len(resp) >= 14:
                w = self._nibble(resp[13])
                focus_mode_auto = bool(w & 0x01)
            self.status_panel.set_zoom_focus(zoom, focus, focus_mode_auto)
            return

        # 2) CAM_CameraBlockInq (7E 7E 01)
        # Response template (datasheet image):
        #   90 50 0p 0p 0q 0q 0r 0s tt 0u vv ww 00 xx 0z FF
        if payload_norm == "7e7e01":
            r_gain = self._parse_u8_from_nibbles(resp, 2)
            b_gain = self._parse_u8_from_nibbles(resp, 4)
            wb_mode = self._nibble(resp[6]) if len(resp) > 6 else None
            aperture = self._nibble(resp[7]) if len(resp) > 7 else None
            ae_mode = resp[8] if len(resp) > 8 else None

            backlight_bit = None
            exposure_comp_bit = None
            if len(resp) > 9:
                u = self._nibble(resp[9])
                backlight_bit = bool(u & 0x04)
                exposure_comp_bit = bool(u & 0x02)

            shutter = self._nibble(resp[10]) if len(resp) > 10 else None
            iris = self._nibble(resp[11]) if len(resp) > 11 else None
            bright = self._nibble(resp[13]) if len(resp) > 13 else None
            exp_comp_pos = self._nibble(resp[14]) if len(resp) > 14 else None

            self.status_panel.set_camera_block(
                r_gain=r_gain,
                b_gain=b_gain,
                wb_mode=wb_mode,
                aperture=aperture,
                ae_mode=ae_mode,
                backlight_bit=backlight_bit,
                exposure_comp_bit=exposure_comp_bit,
                shutter=shutter,
                iris=iris,
                bright=bright,
                exp_comp_pos=exp_comp_pos,
            )
            return

        # 3) CAM_BacklightModeInq (04 33)
        # Response: 90 50 02 FF (On) / 90 50 03 FF (Off)
        if payload_norm == "0433":
            mode = resp[2] if len(resp) > 2 else None
            if mode == 0x02:
                self.status_panel.set_backlight_mode(True)
            elif mode == 0x03:
                self.status_panel.set_backlight_mode(False)
            else:
                self.status_panel.set_backlight_mode(None)
            return

        # 4) CAM_AFSensitivityInq (04 58)
        # Response: 90 50 01 FF (High) / 02 (Normal) / 03 (Low)
        if payload_norm == "0458":
            v = resp[2] if len(resp) > 2 else None
            if v == 0x01:
                self.status_panel.set_af_sensitivity("High")
            elif v == 0x02:
                self.status_panel.set_af_sensitivity("Normal")
            elif v == 0x03:
                self.status_panel.set_af_sensitivity("Low")
            else:
                self.status_panel.set_af_sensitivity(None)
            return

    def refresh_camera_status(self):
        """Send inquiry commands for the status panel (uses current selected target)."""
        try:
            ip, port = self.parse_ip_port()
        except Exception as e:
            self.add_log(f"ERROR: {e}")
            return

        try:
            cam = self.get_camera(ip, port)
        except Exception as e:
            self.add_log(f"ERROR: {type(e).__name__}: {e}")
            return

        inquiries = [
            ("7E 7E 00", "[LensBlockInq]"),
            ("7E 7E 01", "[CameraBlockInq]"),
            ("04 33", "[BacklightModeInq]"),
            ("04 58", "[AFSensitivityInq]"),
        ]

        for payload, tag in inquiries:
            try:
                resp = cam._send_command(payload, query=True)
                if resp is None:
                    self.add_log(f"QUERY {tag}: {payload} | response: <None>")
                else:
                    self.add_log(f"QUERY {tag}: {payload} | response: {resp.hex(' ')}")
                    self._handle_inquiry_response(payload, resp)
            except NoQueryResponse as e:
                self.add_log(f"ERROR: No query response {tag}: {e}")
            except ViscaException as e:
                self.add_log(f"ERROR: VISCA error {tag} (0x{e.status_code:02X}) {e.description}")
            except Exception as e:
                self.add_log(f"ERROR: {type(e).__name__}: {e}")

    def send_current(self):
        try:
            payload = sanitize_hex_payload(self.cmd_edit.text())
        except ValueError as e:
            self.add_log(f"ERROR: {e}")
            return
        self.send_payload(payload, is_query=False, label="(manual)")

    # -----------------------------
    # Preset handling
    # -----------------------------
    def on_preset_selected(self, idx: int):
        if idx < 0:
            return
        label = self.preset_combo.itemText(idx)
        payload, is_query = self.preset_combo.currentData()

        self.cmd_edit.setText(payload)

        self._pending_preset = (label, payload, is_query)
        self._preset_debounce.start(50)

    def _send_selected_preset(self):
        if not self._pending_preset:
            return
        label, payload, is_query = self._pending_preset
        self._pending_preset = None
        self.send_payload(payload, is_query=is_query, label=f"[{label}]")

    # --- ensure at least one target box visible on startup
    def _ensure_initial_target_box(self):
        visible = any(w["box"].isVisible() for w in self.target_boxes.values())
        if not visible and self.target_boxes:
            first = min(self.target_boxes.keys())
            self.target_boxes[first]["box"].setVisible(True)


    # -----------------------------
    # Profiles: save/load all settings
    # -----------------------------
    def _profile_path(self, name: str) -> str:
        safe = "".join(ch for ch in (name or "").strip() if ch.isalnum() or ch in (" ", "_", "-", ".")).strip()
        safe = safe.replace(" ", "_")
        if not safe:
            safe = "default"
        if not safe.lower().endswith(".json"):
            safe += ".json"
        return os.path.join(self._profiles_dir, safe)

    def _read_profiles_meta(self) -> dict[str, object]:
        try:
            if os.path.exists(self._profiles_meta_path):
                with open(self._profiles_meta_path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _write_profiles_meta(self, meta: dict[str, object]) -> None:
        try:
            with open(self._profiles_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta or {}, f, indent=2)
        except Exception:
            pass

    def _get_last_profile_name(self) -> str | None:
        meta = self._read_profiles_meta()
        try:
            v = meta.get("last_profile", None)
            return str(v) if v else None
        except Exception:
            return None

    def _set_last_profile_name(self, name: str | None) -> None:
        meta = self._read_profiles_meta()
        if name:
            meta["last_profile"] = str(name)
        else:
            meta.pop("last_profile", None)
        self._write_profiles_meta(meta)

    def _refresh_profile_list(self) -> None:
        """Refresh the profile combobox from ./profiles/*.json."""
        try:
            existing = set()
            if os.path.isdir(self._profiles_dir):
                for fn in os.listdir(self._profiles_dir):
                    if fn.lower().endswith(".json") and fn != "_meta.json":
                        existing.add(os.path.splitext(fn)[0])
            existing = {e for e in existing if e}
            items = sorted(existing, key=lambda s: s.lower())

            cur_text = self.profile_combo.currentText().strip() if hasattr(self, "profile_combo") else ""
            self.profile_combo.blockSignals(True)
            self.profile_combo.clear()
            for it in items:
                self.profile_combo.addItem(it, it)
            if cur_text:
                # keep typed value
                self.profile_combo.setEditText(cur_text)
            self.profile_combo.blockSignals(False)
        except Exception:
            try:
                self.profile_combo.blockSignals(False)
            except Exception:
                pass

    @staticmethod
    def _jsonable(obj):
        """Convert tuples -> lists recursively so json can serialize mapping data."""
        if isinstance(obj, tuple):
            return [MainWindow._jsonable(x) for x in obj]
        if isinstance(obj, list):
            return [MainWindow._jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): MainWindow._jsonable(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _tuplify(obj):
        """Best-effort restore lists back to tuples for mapping payloads."""
        if isinstance(obj, list):
            # preserve nested structure
            return tuple(MainWindow._tuplify(x) for x in obj)
        if isinstance(obj, dict):
            return {k: MainWindow._tuplify(v) for k, v in obj.items()}
        return obj

    def _collect_profile_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "version": 1,
            "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Targets
        try:
            state["targets"] = self._jsonable(getattr(self, "_targets", {}) or {})
            state["active_target_slot"] = getattr(self, "_active_target_slot", None)
        except Exception:
            pass

        # Controller-related settings
        try:
            state["rate_limit_hz"] = int(self.rate_limit_spin.value())
        except Exception:
            pass

        # Physical button mappings (controller buttons/hats) -> action
        try:
            mapping = self.get_button_mapping()
            state["button_mapping"] = self._jsonable(mapping)
            self._snapshot_active_button_mapping()
            state["button_mappings_by_controller"] = self._jsonable(
                getattr(self, "_button_mappings_by_controller", {}) or {}
            )
        except Exception:
            pass

        # Response curves + axis mappings
        try:
            if hasattr(self, "curve_editor") and self.curve_editor is not None:
                state["curve_state"] = self.curve_editor.get_state()
        except Exception:
            pass

        # Virtual pad mappings
        try:
            if hasattr(self, "virtual_controller") and self.virtual_controller is not None:
                state["virtual_mapping"] = self.virtual_controller.get_mapping_state()
        except Exception:
            pass

        # Hex list
        try:
            if hasattr(self, "hex_list_tab") and self.hex_list_tab is not None:
                state["hex_list"] = self.hex_list_tab.get_rows_state()
        except Exception:
            pass

        return state

    def _apply_profile_state(self, state: dict[str, object]) -> None:
        """Apply profile state immediately where possible; defer controller-bound items if needed."""
        if not isinstance(state, dict):
            return

        # Targets
        try:
            t = state.get("targets", None)
            if isinstance(t, dict):
                # normalize keys to int
                targets: dict[int, dict[str, object]] = {}
                for k, v in t.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        targets[kk] = v
                self._targets = targets
                self._save_targets()
                self._refresh_targets_ui()
                self._refresh_target_actions_in_mapping_combos()
        except Exception:
            pass

        # Active target slot (optional)
        try:
            ats = state.get("active_target_slot", None)
            if ats is None:
                pass
            else:
                self._active_target_slot = int(ats)
                self._refresh_targets_ui()
        except Exception:
            pass

        # Rate limit
        try:
            if "rate_limit_hz" in state:
                self.rate_limit_spin.setValue(int(state.get("rate_limit_hz", self._send_rate_limit_hz)))
        except Exception:
            pass

        # Hex list
        try:
            if "hex_list" in state and hasattr(self, "hex_list_tab") and self.hex_list_tab is not None:
                self.hex_list_tab.apply_rows_state(state.get("hex_list", []) or [])
        except Exception:
            pass

        # Virtual controller mapping
        try:
            if "virtual_mapping" in state and hasattr(self, "virtual_controller") and self.virtual_controller is not None:
                self.virtual_controller.apply_mapping_state(state.get("virtual_mapping", {}) or {})
        except Exception:
            pass

        # Controller-bound settings (button mapping + curves) may need controller-selected UI.
        try:
            bmc = state.get("button_mappings_by_controller", {}) or {}
            self._button_mappings_by_controller = dict(bmc) if isinstance(bmc, dict) else {}
        except Exception:
            self._button_mappings_by_controller = {}
        self._pending_profile_state = {
            "button_mapping": state.get("button_mapping", None),
            "button_mappings_by_controller": state.get("button_mappings_by_controller", None),
            "curve_state": state.get("curve_state", None),
        }

        # Attempt immediate apply if mapping widgets already exist
        try:
            self._apply_pending_controller_profile_state()
        except Exception:
            pass

    def _apply_pending_controller_profile_state(self) -> None:
        """Apply pending profile items that depend on controller/mapping UI."""
        st = getattr(self, "_pending_profile_state", None)
        if not st:
            return

        # Restore physical button mapping into comboboxes
        #
        # Important: profiles typically store ONLY assigned inputs. Any inputs not present
        # in the profile should be reset to "Unassigned" so old selections don't linger.
        try:
            bm = st.get("button_mapping", None)
            per_controller = st.get("button_mappings_by_controller", None)
            key = self._controller_mapping_key()
            if key and isinstance(per_controller, dict):
                bm = per_controller.get(key, bm)

            if hasattr(self, "button_map_combos"):
                # Default everything to Unassigned first
                for cb in self.button_map_combos:
                    try:
                        cb.blockSignals(True)
                        cb.setCurrentIndex(0)
                    except Exception:
                        pass
                    finally:
                        try:
                            cb.blockSignals(False)
                        except Exception:
                            pass

            if bm is not None and hasattr(self, "button_map_combos"):
                bm2 = self._tuplify(bm)  # restore tuples
                if isinstance(bm2, dict):
                    for i, cb in enumerate(self.button_map_combos):
                        # mapping keys in file are stringified in jsonable()
                        data = bm2.get(str(i), bm2.get(i, None))

                        # None means explicitly unassigned
                        if data is None:
                            try:
                                cb.blockSignals(True)
                                cb.setCurrentIndex(0)
                            except Exception:
                                pass
                            finally:
                                try:
                                    cb.blockSignals(False)
                                except Exception:
                                    pass
                            continue

                        try:
                            idx = cb.findData(data)
                            cb.blockSignals(True)
                            if idx >= 0:
                                cb.setCurrentIndex(idx)
                            else:
                                # If the profile refers to a target slot that no longer exists,
                                # or a command that isn't in the combo, fall back to Unassigned.
                                cb.setCurrentIndex(0)
                        except Exception:
                            try:
                                cb.setCurrentIndex(0)
                            except Exception:
                                pass
                        finally:
                            try:
                                cb.blockSignals(False)
                            except Exception:
                                pass
        except Exception:
            pass


        # Restore response curves
        try:
            cs = st.get("curve_state", None)
            if cs is not None and hasattr(self, "curve_editor") and self.curve_editor is not None:
                self.curve_editor.apply_state(cs)
        except Exception:
            pass

    def _load_profile_file(self, name: str) -> dict[str, object] | None:
        try:
            p = self._profile_path(name)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                if isinstance(raw, dict):
                    return raw
        except Exception as e:
            self.add_log(f"ERROR: Failed to load profile '{name}': {e}")
        return None

    def _save_profile_file(self, name: str, state: dict[str, object]) -> bool:
        try:
            p = self._profile_path(name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state or {}, f, indent=2)
            return True
        except Exception as e:
            self.add_log(f"ERROR: Failed to save profile '{name}': {e}")
            return False

    def _on_profile_load_clicked(self):
        name = (self.profile_combo.currentText() or "").strip()
        if not name:
            QMessageBox.information(self, "Load profile", "Enter a profile name to load.")
            return
        st = self._load_profile_file(name)
        if st is None:
            QMessageBox.warning(self, "Load profile", f"Profile '{name}' not found.")
            return

        self._apply_profile_state(st)
        self._current_profile = name
        self._set_last_profile_name(name)
        self._refresh_profile_list()
        self.profile_combo.setEditText(name)
        self.add_log(f"Loaded profile: {name}")

    def _on_profile_save_clicked(self):
        name = (self.profile_combo.currentText() or "").strip()
        if not name:
            self._on_profile_save_as_clicked()
            return
        st = self._collect_profile_state()
        ok = self._save_profile_file(name, st)
        if ok:
            self._current_profile = name
            self._set_last_profile_name(name)
            self._refresh_profile_list()
            self.profile_combo.setEditText(name)
            self.add_log(f"Saved profile: {name}")

    def _on_profile_save_as_clicked(self):
        cur = (self.profile_combo.currentText() or "").strip()
        name, ok = QInputDialog.getText(self, "Save profile as", "Profile name:", text=cur or "default")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return
        st = self._collect_profile_state()
        ok2 = self._save_profile_file(name, st)
        if ok2:
            self._current_profile = name
            self._set_last_profile_name(name)
            self._refresh_profile_list()
            self.profile_combo.setEditText(name)
            self.add_log(f"Saved profile: {name}")

    def _on_profile_delete_clicked(self):
        name = (self.profile_combo.currentText() or "").strip()
        if not name:
            return
        p = self._profile_path(name)
        if not os.path.exists(p):
            return
        resp = QMessageBox.question(self, "Delete profile", f"Delete profile '{name}'?", QMessageBox.Yes | QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        try:
            os.remove(p)
            if self._get_last_profile_name() == name:
                self._set_last_profile_name(None)
            self._refresh_profile_list()
            self.profile_combo.setEditText("")
            self.add_log(f"Deleted profile: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Delete profile", f"Failed to delete '{name}': {e}")

    def _load_last_profile_on_startup(self):
        name = self._get_last_profile_name()
        if not name:
            return
        st = self._load_profile_file(name)
        if st is None:
            return
        self._apply_profile_state(st)
        self._current_profile = name
        try:
            self.profile_combo.setEditText(name)
        except Exception:
            pass
        self.add_log(f"Auto-loaded profile: {name}")


def main():
    app = QApplication(sys.argv)
    apply_always_dark_theme(app)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
