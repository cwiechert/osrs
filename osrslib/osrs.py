import csv
import ctypes
import logging
import os
import platform
import subprocess
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil
from math import exp, log
from random import gauss, uniform
from typing import Optional, Union, Tuple, Callable, List

import cv2
import numpy as np
import pyautogui
from mss import mss
from pynput import keyboard, mouse

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ── Emergency stop ──────────────────────────────────────────────────────────
_emergency_listener: Optional[keyboard.Listener] = None


def enable_failsafe(key: keyboard.Key = keyboard.Key.f6) -> None:
    """
    Starts a global hotkey listener that terminates the process on *key*.

    Runs in a daemon thread — works even if the main thread is blocked.
    Call once at the start of your script.

    Args:
        key: The key that triggers an immediate exit.
    """
    global _emergency_listener
    if _emergency_listener is not None:
        return

    def _on_press(pressed):
        if pressed == key:
            logger.warning("EMERGENCY STOP (%s) — terminating process.", key)
            os._exit(1)

    _emergency_listener = keyboard.Listener(on_press=_on_press, daemon=True)
    _emergency_listener.start()
    logger.info("Failsafe enabled: press %s to kill the process.", key)


# ── Constants ────────────────────────────────────────────────────────────────
TRACKBAR_WINDOW_NAME = "Trackbars"
CAPTURE_WINDOW_NAME = "Screen Capture"
RESULT_WINDOW_NAME = "Detection Result"

DEFAULT_LEFT = 0
DEFAULT_TOP = 0
DEFAULT_WIDTH = 300
DEFAULT_HEIGHT = 200

DEFAULT_HUE_MIN = 0
DEFAULT_HUE_MAX = 179
DEFAULT_SAT_MIN = 0
DEFAULT_SAT_MAX = 255
DEFAULT_VAL_MIN = 0
DEFAULT_VAL_MAX = 255


# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class Region:
    """Immutable description of a rectangular screen area."""

    left: int = DEFAULT_LEFT
    top: int = DEFAULT_TOP
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass
class HSVRange:
    """Mutable HSV lower/upper bounds."""

    hue_min: int = DEFAULT_HUE_MIN
    hue_max: int = DEFAULT_HUE_MAX
    sat_min: int = DEFAULT_SAT_MIN
    sat_max: int = DEFAULT_SAT_MAX
    val_min: int = DEFAULT_VAL_MIN
    val_max: int = DEFAULT_VAL_MAX

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.hue_min, self.sat_min, self.val_min])

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.hue_max, self.sat_max, self.val_max])


class TargetStrategy(Enum):
    """How to choose a single target from multiple detected centers."""

    NEAREST = auto()     # Closest to current mouse position.
    FIRST = auto()       # First contour returned by OpenCV.
    LARGEST = auto()     # Contour with the biggest area.
    CENTROID = auto()    # Average of all centers (original behaviour).


# ── RegionHSV ────────────────────────────────────────────────────────────────
class RegionHSV:
    """
    Detects objects within a screen region based on HSV color filtering.

    Workflow:
        1. Instantiate the class.
        2. (Optional) Call ``configure()`` to interactively tune region & HSV.
        3. Call ``get_centers()`` for coordinates or ``draw_centers()`` for a
           live preview.

    Supports the context-manager protocol::

        with RegionHSV() as detector:
            detector.configure()
            centers = detector.get_centers()

    Args:
        verbose: If ``True``, log status messages.
        play_pause_key: Key character to pause/resume concurrent tasks.
        stop_key: Key character to stop concurrent tasks.
        min_contour_area: Minimum contour area (px²) to consider a detection.
            Contours smaller than this are treated as noise and ignored.
        target_strategy: Strategy for choosing a single point when running
            ``_real_time_coordinates`` with multiple detections.
    """
    def __init__(
        self,
        verbose: bool = True,
        play_pause_key: str = "+",
        stop_key: str = "}",
        min_contour_area: float = 0.0,
        target_strategy: TargetStrategy = TargetStrategy.NEAREST,
    ):
        self.verbose = verbose
        self._sct = mss()

        self.screen_width, self.screen_height = pyautogui.size()
        self.region = Region()
        self.hsv = HSVRange()

        self.min_contour_area = min_contour_area

        # Concurrent-task attributes.
        self.target_strategy = target_strategy
        self.play_pause_key = play_pause_key
        self.stop_key = stop_key
        self.active: Optional[bool] = None

        self._lock = threading.Lock()
        self._centers: List[Tuple[int, int]] = []

        self._stop_event: Optional[threading.Event] = None
        self._run_event: Optional[threading.Event] = None
        self._listener: Optional[keyboard.Listener] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def centers(self) -> List[Tuple[int, int]]:
        with self._lock:
            return list(self._centers)
        
    @centers.setter
    def centers(self, value: List[Tuple[int, int]]):
        with self._lock:
            self._centers = list(value)

    # ── Internal helpers ─────────────────────────────────────────────────
    def _create_trackbars(self) -> None:
        """Creates a window with trackbars for adjusting region & HSV params."""
        cv2.namedWindow(TRACKBAR_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(TRACKBAR_WINDOW_NAME, 400, 480)

        cv2.createTrackbar("Left", TRACKBAR_WINDOW_NAME, self.region.left, self.screen_width, lambda _: None)
        cv2.createTrackbar("Top", TRACKBAR_WINDOW_NAME, self.region.top, self.screen_height, lambda _: None)
        cv2.createTrackbar("Width", TRACKBAR_WINDOW_NAME, self.region.width, self.screen_width, lambda _: None)
        cv2.createTrackbar("Height", TRACKBAR_WINDOW_NAME, self.region.height, self.screen_height, lambda _: None)

        cv2.createTrackbar("Hue Min", TRACKBAR_WINDOW_NAME, self.hsv.hue_min, 179, lambda _: None)
        cv2.createTrackbar("Hue Max", TRACKBAR_WINDOW_NAME, self.hsv.hue_max, 179, lambda _: None)
        cv2.createTrackbar("Sat Min", TRACKBAR_WINDOW_NAME, self.hsv.sat_min, 255, lambda _: None)
        cv2.createTrackbar("Sat Max", TRACKBAR_WINDOW_NAME, self.hsv.sat_max, 255, lambda _: None)
        cv2.createTrackbar("Val Min", TRACKBAR_WINDOW_NAME, self.hsv.val_min, 255, lambda _: None)
        cv2.createTrackbar("Val Max", TRACKBAR_WINDOW_NAME, self.hsv.val_max, 255, lambda _: None)

    def _update_params_from_trackbars(self) -> None:
        """Reads current trackbar positions into ``self.region`` and ``self.hsv``."""
        try:
            left = cv2.getTrackbarPos("Left", TRACKBAR_WINDOW_NAME)
            top = cv2.getTrackbarPos("Top", TRACKBAR_WINDOW_NAME)
            width = max(1, cv2.getTrackbarPos("Width", TRACKBAR_WINDOW_NAME))
            height = max(1, cv2.getTrackbarPos("Height", TRACKBAR_WINDOW_NAME))

            # Clamp to screen bounds.
            left = max(0, min(left, self.screen_width - 1))
            top = max(0, min(top, self.screen_height - 1))
            width = min(width, self.screen_width - left)
            height = min(height, self.screen_height - top)

            hue_min = cv2.getTrackbarPos("Hue Min", TRACKBAR_WINDOW_NAME)
            hue_max = cv2.getTrackbarPos("Hue Max", TRACKBAR_WINDOW_NAME)
            sat_min = cv2.getTrackbarPos("Sat Min", TRACKBAR_WINDOW_NAME)
            sat_max = cv2.getTrackbarPos("Sat Max", TRACKBAR_WINDOW_NAME)
            val_min = cv2.getTrackbarPos("Val Min", TRACKBAR_WINDOW_NAME)
            val_max = cv2.getTrackbarPos("Val Max", TRACKBAR_WINDOW_NAME)
        except cv2.error:
            logger.debug("Trackbars closed - keeping last known values.")
            return

        with self._lock:
            self.region = Region(left, top, width, height)
            self.hsv = HSVRange(hue_min, hue_max, sat_min, sat_max, val_min, val_max)

    def _process_frame(self, sct_object) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Captures a frame from the screen region and returns filtered results.

        Args:
            sct_object: An ``mss`` instance to capture with.

        Returns:
            A tuple of (contours, original_bgr_frame, binary_mask).
        """
        with self._lock:
            region_dict = self.region.as_dict()
            lower = self.hsv.lower
            upper = self.hsv.upper

        screenshot = sct_object.grab(region_dict)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.min_contour_area > 0:
            contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        return contours, frame, mask

    def _select_target(
        self,
        centers: List[Tuple[int, int]],
        areas: List[float],
    ) -> Optional[Tuple[int, int]]:
        """
        Picks one target point from *centers* according to ``self.target_strategy``.

        Args:
            centers: Detected center coordinates (screen-absolute).
            areas: Contour areas corresponding to each center.

        Returns:
            A single (x, y) tuple, or ``None`` if *centers* is empty.
        """
        if not centers:
            return None

        strategy = self.target_strategy

        if strategy == TargetStrategy.FIRST:
            return centers[0]

        if strategy == TargetStrategy.LARGEST:
            idx = int(np.argmax(areas))
            return centers[idx]

        if strategy == TargetStrategy.CENTROID:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            return (int(np.mean(xs)), int(np.mean(ys)))

        # Default: NEAREST to current mouse position.
        mx, my = pyautogui.position()
        dists = [(cx - mx) ** 2 + (cy - my) ** 2 for cx, cy in centers]
        idx = int(np.argmin(dists))
        return centers[idx]

    # ── Public API ───────────────────────────────────────────────────────
    def configure(self, quit_key: str = "q") -> None:
        """
        Opens a GUI for real-time adjustment of the screen region and HSV values.

        The masked result is shown in a preview window.  Adjust sliders until
        only the target objects are visible, then press *quit_key* to save the
        current settings and close the windows.

        Args:
            quit_key: Key to close the GUI and keep current settings.
        """
        self._create_trackbars()
        cv2.namedWindow(CAPTURE_WINDOW_NAME, cv2.WINDOW_NORMAL)

        if self.verbose:
            logger.info(
                "Adjust parameters in '%s'. Press '%s' in '%s' to finish.",
                TRACKBAR_WINDOW_NAME,
                quit_key,
                CAPTURE_WINDOW_NAME,
            )

        try:
            while True:
                self._update_params_from_trackbars()
                _, frame, mask = self._process_frame(self._sct)
                result = cv2.bitwise_and(frame, frame, mask=mask)

                with self._lock:
                    w, h = self.region.width, self.region.height

                cv2.resizeWindow(CAPTURE_WINDOW_NAME, w, h)
                cv2.imshow(CAPTURE_WINDOW_NAME, result)

                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
        finally:
            cv2.destroyAllWindows()

        if self.verbose:
            logger.info("Configuration complete. Ready to detect objects.")

    def export(self, var_name: str = "r") -> str:
        """
        Generates a paste-ready code snippet with the current region and HSV
        settings and copies it to the system clipboard.

        Ideal after ``configure()``: tune the sliders, press quit, call
        ``export()``, and paste into another cell or script.

        Args:
            var_name: Variable name used in the generated code.

        Returns:
            The generated code string (also copied to clipboard).
        """
        with self._lock:
            reg = self.region
            hsv = self.hsv
            mca = self.min_contour_area

        lines = [
            f"{var_name} = RegionHSV(min_contour_area={mca})",
            f"{var_name}.region = Region("
            f"left={reg.left}, top={reg.top}, "
            f"width={reg.width}, height={reg.height})",
            f"{var_name}.hsv = HSVRange("
            f"hue_min={hsv.hue_min}, hue_max={hsv.hue_max}, "
            f"sat_min={hsv.sat_min}, sat_max={hsv.sat_max}, "
            f"val_min={hsv.val_min}, val_max={hsv.val_max})",
        ]
        snippet = "\n".join(lines)

        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["pbcopy"], input=snippet.encode(), check=True)
            elif system == "Linux":
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=snippet.encode(),
                    check=True,
                )
            elif system == "Windows":
                subprocess.run(["clip"], input=snippet.encode(), check=True)
            else:
                raise OSError(f"Unsupported platform: {system}")

            if self.verbose:
                logger.info("Copied to clipboard:\n%s", snippet)
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "Could not copy to clipboard (%s). Code printed below:\n%s",
                exc,
                snippet,
            )

        return snippet

    def get_centers(
        self,
        sct_object=None,
    ) -> List[Tuple[int, int]]:
        """
        Returns screen-absolute center coordinates of every detected object.

        Args:
            sct_object: Optional ``mss`` instance.  A default is used when
                ``None``.

        Returns:
            A list of (x, y) tuples.  Empty when nothing is found.
        """
        if sct_object is None:
            sct_object = self._sct

        contours, _, _ = self._process_frame(sct_object)

        with self._lock:
            region_left = self.region.left
            region_top = self.region.top

        centers: List[Tuple[int, int]] = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + region_left
                cy = int(M["m01"] / M["m00"]) + region_top
                centers.append((cx, cy))

        with self._lock:
            self._centers = list(centers)

        if self.verbose:
            logger.info("Found %d object(s).", len(centers))

        return centers

    def draw_centers(self, show_hsv_mask: bool = False, quit_key: str = "q") -> None:
        """
        Displays a live feed with detected centers drawn as green circles.

        Args:
            show_hsv_mask: When ``True``, draws on the binary mask instead of
                the colour frame.
            quit_key: Key to close the preview window.
        """
        cv2.namedWindow(RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.verbose:
            logger.info(
                "Displaying detections. Press '%s' in '%s' to stop.",
                quit_key,
                RESULT_WINDOW_NAME,
            )

        try:
            while True:
                contours, frame, mask = self._process_frame(self._sct)
                display = cv2.bitwise_and(frame, frame, mask=mask) if show_hsv_mask else frame

                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx_local = int(M["m10"] / M["m00"])
                        cy_local = int(M["m01"] / M["m00"])
                        cv2.circle(display, (cx_local, cy_local), 5, (0, 255, 0), -1)

                with self._lock:
                    w, h = self.region.width, self.region.height

                cv2.resizeWindow(RESULT_WINDOW_NAME, w, h)
                cv2.imshow(RESULT_WINDOW_NAME, display)

                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
        finally:
            cv2.destroyAllWindows()

    # ── Concurrency ──────────────────────────────────────────────────────
    def _real_time_coordinates(
        self,
        run_event: threading.Event,
        stop_event: threading.Event,
        refresh_rate: float = 0.1,
    ) -> None:
        """
        Continuously moves the mouse towards detected objects.

        Runs inside a daemon thread.  The target is chosen using
        ``self.target_strategy``.

        Args:
            run_event: Cleared to pause, set to resume.
            stop_event: Set to terminate the loop.
            refresh_rate: Seconds between capture cycles.
        """
        try:
            with mss() as sct:
                while not stop_event.is_set():
                    run_event.wait()

                    if stop_event.is_set():
                        break

                    try:
                        contours, _, _ = self._process_frame(sct)

                        with self._lock:
                            region_left = self.region.left
                            region_top = self.region.top

                        centers: List[Tuple[int, int]] = []
                        areas: List[float] = []
                        for contour in contours:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"]) + region_left
                                cy = int(M["m01"] / M["m00"]) + region_top
                                centers.append((cx, cy))
                                areas.append(cv2.contourArea(contour))

                            with self._lock:
                                self._centers = list(centers)

                            target = self._select_target(centers, areas)

                        target = self._select_target(centers, areas)

                        if target is None:
                            time.sleep(refresh_rate)
                            continue

                        _move_to(target[0], target[1], duration=0)
                    except Exception:
                        logger.exception("Error in _real_time_coordinates loop.")
                        time.sleep(refresh_rate)
        except Exception:
            logger.exception("Fatal error in _real_time_coordinates thread.")

    def concurrent_tasks(
        self,
        external_function: Callable[[], None],
        refresh_rate: float = 0.1,
    ) -> None:
        """
        Runs ``_real_time_coordinates`` and *external_function* concurrently.

        Keyboard controls (configurable via constructor):
            * ``play_pause_key`` - toggle pause / resume.
            * ``stop_key`` - stop everything and return.

        Args:
            external_function: A no-arg callable executed in a loop alongside
                the detection thread.

        Raises:
            RuntimeError: If *external_function* raises while running.
        """

        self._stop_event = threading.Event()
        self._run_event = threading.Event()
        self._run_event.set()
        self.active = True

        external_error: List[BaseException] = []

        def on_press(key):
            try:
                if key.char == self.play_pause_key:
                    if self._run_event.is_set():
                        self._run_event.clear()
                        self.active = False
                        if self.verbose:
                            logger.info("--- TASKS PAUSED ---")
                    else:
                        self._run_event.set()
                        self.active = True
                        if self.verbose:
                            logger.info("--- TASKS RESUMED ---")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char == self.stop_key:
                    if self.verbose:
                        logger.info("--- STOPPING TASKS ---")
                    self.kill_concurrency()
                    return False
            except AttributeError:
                pass

        def _external_wrapper():
            try:
                while not self._stop_event.is_set():
                    self._run_event.wait()
                    if self._stop_event.is_set():
                        break
                    external_function()
            except Exception as exc:
                external_error.append(exc)
                logger.exception("External function raised an exception - stopping tasks.")
                self.kill_concurrency()

        task_detect = threading.Thread(
            target=self._real_time_coordinates,
            daemon=True,
            args=(self._run_event, self._stop_event, refresh_rate),
        )
        task_external = threading.Thread(target=_external_wrapper, daemon=True)

        if self.verbose:
            logger.info(
                "Starting tasks. Press '%s' to pause/resume. Press '%s' to exit.",
                self.play_pause_key,
                self.stop_key,
            )
        task_detect.start()
        task_external.start()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as self._listener:
            self._listener.join()

        task_detect.join(timeout=2)
        task_external.join(timeout=2)
        if self.verbose:
            logger.info("--- Program finished. ---")

        if external_error:
            raise RuntimeError(
                "External function failed during concurrent execution."
            ) from external_error[0]

    def kill_concurrency(self) -> None:
        """Signals all concurrent tasks to stop immediately."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._run_event is not None:
            self._run_event.set()  # Unblock any .wait() calls.
        if self._listener is not None:
            self._listener.stop()
        self.active = False

    # ── Cleanup ──────────────────────────────────────────────────────────
    def close(self) -> None:
        """Releases the screen-capture handle and destroys OpenCV windows."""
        self.kill_concurrency()
        self._sct.close()
        cv2.destroyAllWindows()
        if self.verbose:
            logger.info("RegionHSV resources closed.")


# ── Recorder ─────────────────────────────────────────────────────────────────

@dataclass
class ClickEvent:
    """A single recorded input event (mouse click or key press)."""

    timestamp: float
    x: int
    y: int
    button: str  # "left" | "right" | ""
    event_type: str = "click"  # "click" | "key"
    key: str = ""  # key name for key events


class Recorder:
    """
    Records and replays mouse-click sequences with precise timing.

    Args:
        record: ``True`` to start in recording mode; ``False`` for playback.
        filename: Path to the CSV file used for saving / loading events.
        stop_key: Key that terminates the recording session.

    Examples:
        Recording::

            rec = Recorder(record=True, filename="clicks.csv")
            rec.record_and_save()

        Playback::

            rec = Recorder(record=False, filename="clicks.csv")
            rec.reproduce(iterations=3, x_rand=5, y_rand=5)
    """
    _CSV_FIELDS = ("timestamps", "x_axis", "y_axis", "button", "event_type", "key")

    def __init__(
        self,
        record: bool = False,
        filename: str = "mouse_record.csv",
        stop_key: keyboard.Key = keyboard.Key.ctrl_l,
    ):
        self.record = record
        self.filename = filename
        self.stop_key = stop_key
        self.events: List[ClickEvent] = []

        if not self.record:
            self._load_recording()

    # ── Persistence ──────────────────────────────────────────────────────
    def _load_recording(self) -> None:
        """Loads click events from a CSV file into ``self.events``."""
        try:
            with open(self.filename, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])
                required = {"timestamps", "x_axis", "y_axis", "button"}
                if not required.issubset(fieldnames):
                    missing = required - fieldnames
                    raise ValueError(f"CSV missing required columns: {missing}")

                has_keys = "event_type" in fieldnames

                self.events = [
                    ClickEvent(
                        timestamp=float(row["timestamps"]),
                        x=int(row["x_axis"]),
                        y=int(row["y_axis"]),
                        button=row["button"],
                        event_type=row.get("event_type", "click") if has_keys else "click",
                        key=row.get("key", "") if has_keys else "",
                    )
                    for row in reader
                ]

        except FileNotFoundError:
            raise FileNotFoundError(f"Recording file '{self.filename}' not found.")
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Malformed CSV - {exc}") from exc

    def _save_to_csv(self) -> None:
        """Persists ``self.events`` to CSV."""
        if not self.events:
            logger.warning("No events recorded - CSV not written.")
            return

        with open(self.filename, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for ev in self.events:
                writer.writerow(
                    {
                        "timestamps": ev.timestamp,
                        "x_axis": ev.x,
                        "y_axis": ev.y,
                        "button": ev.button,
                        "event_type": ev.event_type,
                        "key": ev.key,
                    }
                )

    # ── Recording ────────────────────────────────────────────────────────
    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Mouse listener callback - appends events while recording."""
        if pressed and button in (mouse.Button.left, mouse.Button.right):
            self.events.append(
                ClickEvent(
                    timestamp=time.perf_counter(),
                    x=x,
                    y=y,
                    button=button.name,
                )
            )

    def _on_press(self, key) -> Optional[bool]:
        """Keyboard listener callback - returns ``False`` to stop."""
        if key == self.stop_key:
            return False
        return None

    def _on_press_record(self, key) -> Optional[bool]:
        """Keyboard listener callback that records key presses and stops on stop_key."""
        if key == self.stop_key:
            return False

        try:
            key_name = key.char  # alphanumeric keys
        except AttributeError:
            key_name = key.name  # special keys (shift, ctrl, etc.)

        if key_name:
            self.events.append(
                ClickEvent(
                    timestamp=time.perf_counter(),
                    x=0,
                    y=0,
                    button="",
                    event_type="key",
                    key=key_name,
                )
            )
        return None

    def record_and_save(self, record_keys: bool = False) -> None:
        """
        Begins recording mouse clicks (and optionally key presses) until
        ``self.stop_key`` is pressed.

        Args:
            record_keys: If ``True``, key presses are also recorded alongside
                mouse clicks.

        Raises:
            RuntimeError: If called when not in recording mode.
        """
        if not self.record:
            raise RuntimeError(
                "Not in recording mode. Instantiate with record=True."
            )

        what = "mouse clicks and key presses" if record_keys else "mouse clicks"
        logger.info("Recording %s… Press '%s' to stop.", what, self.stop_key)

        on_press = self._on_press_record if record_keys else self._on_press

        mouse_listener = mouse.Listener(on_click=self._on_click)
        keyboard_listener = keyboard.Listener(on_press=on_press)

        mouse_listener.start()
        keyboard_listener.start()
        keyboard_listener.join()
        mouse_listener.stop()

        self._save_to_csv()
        logger.info(
            "Recording stopped. Saved %d event(s) to '%s'.",
            len(self.events),
            self.filename,
        )

    def print_events(self) -> None:
        """Prints the recorded sequence as readable event/sleep pairs."""
        if not self.events:
            logger.warning("No events to display.")
            return

        for i, ev in enumerate(self.events):
            if ev.event_type == "key":
                print(f"Key({ev.key})")
            else:
                print(f"click({ev.x}, {ev.y}, {ev.button})")

            if i < len(self.events) - 1:
                gap = self.events[i + 1].timestamp - ev.timestamp
                print(f"sleep({gap:.2f})")

    # ── Playback ─────────────────────────────────────────────────────────
    def reproduce(
        self,
        iterations: int = 1,
        x_rand: int = 0,
        y_rand: int = 0,
        move_fraction: float = 0.6,
        strict_timing: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Replays the recorded click sequence replicating the original timing.

        The gap between consecutive recorded events is split into a movement
        phase and a wait phase.  ``move_fraction`` controls the split: at
        0.6 the cursor spends ~60 % of each gap moving and ~40 % idle.

        Args:
            iterations: How many times to replay the full sequence.
            x_rand: Maximum random pixel offset on the X axis.
            y_rand: Maximum random pixel offset on the Y axis.
            move_fraction: Fraction of each inter-event gap used for cursor
                movement (0.0-1.0).
            strict_timing: When ``True`` and a movement overshoots its
                time budget, the next movement duration is shortened to
                stay on schedule.  When ``False`` overshoots are ignored
                and the sequence may drift.
            verbose: Print per-iteration timing stats.

        Raises:
            ValueError: On invalid parameter values.
        """
        if not self.events:
            logger.warning("No recording data to reproduce.")
            return

        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        if x_rand < 0 or y_rand < 0:
            raise ValueError("x_rand and y_rand must be >= 0.")
        if not 0.0 < move_fraction <= 1.0:
            raise ValueError("move_fraction must be in (0.0, 1.0].")

        screen_w, screen_h = _screen_size()
        base_time = self.events[0].timestamp
        relative_times = [ev.timestamp - base_time for ev in self.events]

        # Pre-compute the gap before each event (0 for the first one).
        gaps = [0.0] + [
            relative_times[i] - relative_times[i - 1]
            for i in range(1, len(relative_times))
        ]

        iteration_durations: List[float] = []

        if verbose:
            logger.info("Starting playback (%d iteration(s))…", iterations)

        for q in range(iterations):
            iter_start = time.perf_counter()

            # Per-iteration timing drift: stretch/compress the whole
            # sequence by ±5 % so no two iterations are structurally
            # identical, plus a cumulative random walk.
            tempo_scale = uniform(0.95, 1.05)
            cumulative_drift = 0.0

            # Per-iteration spatial drift: shift ALL base coordinates by a
            # small random offset so the click cluster center moves between
            # iterations (simulates posture/hand repositioning).
            iter_drift_x = int(round(gauss(0, max(x_rand * 0.4, 2))))
            iter_drift_y = int(round(gauss(0, max(y_rand * 0.4, 2))))

            for i, (rel_time, ev) in enumerate(zip(relative_times, self.events)):
                cumulative_drift += gauss(0, 0.008)
                target_time = iter_start + rel_time * tempo_scale + cumulative_drift

                if ev.event_type == "key":
                    # Wait until the key should be pressed.
                    wait = target_time - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    press_key(ev.key)
                    continue

                # Base position shifted by iteration drift.
                base_x = ev.x + iter_drift_x
                base_y = ev.y + iter_drift_y

                # Biased Gaussian offset (overshoot + slight drift).
                mx, my = pyautogui.position()
                dir_x = 1 if base_x >= mx else -1
                dir_y = 1 if base_y >= my else -1
                bias_x = dir_x * uniform(0, x_rand * 0.3) if x_rand else 0
                bias_y = dir_y * uniform(0, y_rand * 0.3) if y_rand else 0
                jx = gauss(0, max(x_rand * 0.5, 1)) if x_rand else 0
                jy = gauss(0, max(y_rand * 0.5, 1)) if y_rand else 0
                rand_x = _clamp(base_x + int(round(bias_x + jx)), 0, screen_w - 1)
                rand_y = _clamp(base_y + int(round(bias_y + jy)), 0, screen_h - 1)

                gap = gaps[i]

                if gap <= 0:
                    # First event: click immediately.
                    _move_to(rand_x, rand_y, duration=0.3)
                else:
                    budget = gap * move_fraction

                    if strict_timing:
                        # If we're behind schedule, shrink the move duration.
                        remaining = target_time - time.perf_counter()
                        budget = min(budget, max(remaining * move_fraction, 0))

                    _move_to(rand_x, rand_y, duration=budget)

                    # Sleep any remaining time until the click should happen.
                    wait = target_time - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)

                _click_mouse(button=ev.button)

            iter_duration = time.perf_counter() - iter_start
            iteration_durations.append(iter_duration)

            if verbose:
                avg = sum(iteration_durations) / len(iteration_durations)
                orig = relative_times[-1] if relative_times else 0.0
                logger.info(
                    "Iteration %d/%d: Actual=%.2fs, Avg=%.2fs, Original=%.2fs",
                    q + 1,
                    iterations,
                    iter_duration,
                    avg,
                    orig,
                )


# ── Win32 low-level input ────────────────────────────────────────────────────

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_SCANCODE = 0x0008
_KEYEVENTF_KEYUP = 0x0002
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010

# Map readable key names → hardware scan codes (Set 1).
_SCAN_CODES: dict = {
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "y": 0x15, "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22,
    "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30,
    "n": 0x31, "m": 0x32,
    "space": 0x39, "enter": 0x1C, "esc": 0x01, "tab": 0x0F,
    "backspace": 0x0E, "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F,
    "f6": 0x40, "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44,
    "f11": 0x57, "f12": 0x58,
}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("u", _U),
    ]


def _send_scan(scan_code: int, key_up: bool = False) -> None:
    """Sends a single scan-code key event via ``SendInput``."""
    flags = _KEYEVENTF_SCANCODE | (_KEYEVENTF_KEYUP if key_up else 0)
    ki = _KEYBDINPUT(
        wVk=0,
        wScan=scan_code,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = _INPUT(type=_INPUT_KEYBOARD)
    inp.ki = ki
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _send_mouse_event(flags: int) -> None:
    """Sends a single mouse event via ``SendInput``."""
    mi = _MOUSEINPUT(
        dx=0, dy=0, mouseData=0, dwFlags=flags, time=0,
        dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = _INPUT(type=_INPUT_MOUSE)
    inp.mi = mi
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _click_mouse(button: str = "left", hold_time: Optional[float] = None) -> None:
    """Simulates a mouse click via ``SendInput``."""
    if hold_time is None:
        hold_time = _human_delay(mu=0.065, sigma=0.03, lo=0.03, hi=0.18)

    if button == "right":
        _send_mouse_event(_MOUSEEVENTF_RIGHTDOWN)
        time.sleep(hold_time)
        _send_mouse_event(_MOUSEEVENTF_RIGHTUP)
    else:
        _send_mouse_event(_MOUSEEVENTF_LEFTDOWN)
        time.sleep(hold_time)
        _send_mouse_event(_MOUSEEVENTF_LEFTUP)


def press_key(
    key: str,
    hold_time: Optional[float] = None,
) -> None:
    """
    Simulates a physical key press using hardware scan codes.

    Unlike ``pyautogui.press()`` which sends virtual-key events, this
    function uses ``SendInput`` with ``KEYEVENTF_SCANCODE``, which is
    recognised by applications that read raw / DirectInput.

    Args:
        key: Key name — a single character (``"a"``-``"z"``, ``"0"``-``"9"``)
            or a named key (``"space"``, ``"enter"``, ``"f1"``-``"f12"``,
            ``"shift"``, ``"ctrl"``, ``"alt"``, ``"esc"``, ``"tab"``,
            ``"backspace"``).
        hold_time: Seconds to hold the key down before releasing.
            Defaults to a random value between 0.04 and 0.09 s.

    Raises:
        ValueError: If *key* is not in the scan-code table.
    """
    scan = _SCAN_CODES.get(key.lower())
    if scan is None:
        raise ValueError(
            f"Unknown key {key!r}. Available: {', '.join(sorted(_SCAN_CODES))}"
        )

    if hold_time is None:
        hold_time = _human_delay(mu=0.075, sigma=0.035, lo=0.03, hi=0.20)

    _send_scan(scan)
    time.sleep(hold_time)
    _send_scan(scan, key_up=True)


# ── Standalone helpers ───────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamps *value* to the inclusive range [lo, hi]."""
    return max(lo, min(value, hi))


_cached_screen_size: Optional[Tuple[int, int]] = None


def _screen_size() -> Tuple[int, int]:
    """Returns screen dimensions, cached after first call."""
    global _cached_screen_size
    if _cached_screen_size is None:
        _cached_screen_size = pyautogui.size()
    return _cached_screen_size


def _human_delay(mu: float, sigma: float, lo: float, hi: float) -> float:
    """Samples a log-normal delay clamped to [lo, hi].

    Log-normal approximates human reaction/hold times better than uniform:
    most values cluster near the median with occasional longer outliers.
    """
    log_mu = log(mu**2 / (mu**2 + sigma**2) ** 0.5)
    log_sigma = (log(1 + sigma**2 / mu**2)) ** 0.5
    return max(lo, min(hi, exp(gauss(log_mu, log_sigma))))


def _jittered_sleep(target_time: float) -> None:
    """Sleeps until *target_time* with right-skewed human-like jitter.

    Humans are more likely to be slightly late than early, so the jitter
    uses a log-normal offset biased toward positive (late) values:
    median ~+1 ms, occasional spikes up to ~12 ms, rarely early by > 2 ms.
    """
    # Right-skewed: mostly +0.5 to +4 ms, occasional longer pauses.
    skew = _human_delay(mu=0.002, sigma=0.003, lo=0.0, hi=0.012)
    # Small chance of being slightly early.
    if uniform(0, 1) < 0.25:
        skew = -skew * 0.4  # early side is smaller
    deadline = target_time + skew
    now = time.perf_counter()
    if deadline - now > 0.002:
        time.sleep(deadline - now - 0.001)
    while time.perf_counter() < deadline:
        pass


def _minimum_jerk(t: float, skew: float = 0.0) -> float:
    """Applies a minimum-jerk velocity profile to parameter *t* (0..1).

    The minimum-jerk model (Flash & Hogan 1985) is the standard kinematic
    model of human reaching movements: rapid acceleration to peak speed
    around the midpoint, then gradual deceleration toward the target.

    The *skew* parameter (typically -0.15 to +0.15) shifts the peak
    velocity earlier or later, adding per-movement variety.
    """
    # Base minimum-jerk: 10t³ - 15t⁴ + 6t⁵
    # With skew we blend toward an asymmetric profile.
    base = 10 * t**3 - 15 * t**4 + 6 * t**5
    if skew == 0.0:
        return base
    # Shift peak by blending with a power curve.
    # Positive skew = decelerate later, negative = decelerate earlier.
    power = max(0.3, 1.0 - skew)
    asymmetric = t ** power
    blend = abs(skew) * 2  # 0..~0.3
    return base * (1 - blend) + asymmetric * blend


def _cubic_bezier(
    t: float,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> Tuple[float, float]:
    """Evaluates a cubic Bézier curve at parameter *t* (0..1)."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def _move_to(
    x: int,
    y: int,
    duration: float = 0.5,
    curve_strength: float = 0.3,
    steps_per_second: int = 120,
) -> None:
    """Moves the cursor to (*x*, *y*) along a randomised Bézier curve."""
    start_x, start_y = pyautogui.position()

    if duration <= 0:
        ctypes.windll.user32.SetCursorPos(x, y)
        return

    dx = x - start_x
    dy = y - start_y
    dist = (dx**2 + dy**2) ** 0.5

    if dist == 0:
        return

    # Perpendicular direction for offsetting control points.
    perp_x = -dy / dist
    perp_y = dx / dist

    # Two control points at ~1/3 and ~2/3 of the path, each randomly
    # offset perpendicular to the straight line.
    offset1 = uniform(-curve_strength, curve_strength) * dist
    offset2 = uniform(-curve_strength, curve_strength) * dist

    t1 = uniform(0.15, 0.45)
    t2 = uniform(0.55, 0.85)
    cp1 = (
        start_x + dx * t1 + perp_x * offset1,
        start_y + dy * t1 + perp_y * offset1,
    )
    cp2 = (
        start_x + dx * t2 + perp_x * offset2,
        start_y + dy * t2 + perp_y * offset2,
    )

    p0 = (float(start_x), float(start_y))
    p3 = (float(x), float(y))

    total_steps = max(2, ceil(duration * steps_per_second))
    set_cursor = ctypes.windll.user32.SetCursorPos
    perf = time.perf_counter
    start_time = perf()

    # Per-movement random skew so the velocity peak shifts slightly.
    skew = uniform(-0.12, 0.12)

    for i in range(1, total_steps + 1):
        target = start_time + (i / total_steps) * duration
        # Remap linear t through minimum-jerk easing: accelerate, peak, decelerate.
        t_linear = i / total_steps
        t = _minimum_jerk(t_linear, skew)
        bx, by = _cubic_bezier(t, p0, cp1, cp2, p3)
        set_cursor(int(round(bx)), int(round(by)))

        # Human-like jittered wait instead of perfect busy-wait.
        _jittered_sleep(target)


def click(
    x: int,
    y: int,
    x_rand: int = 0,
    y_rand: int = 0,
    shift: bool = False,
    duration: Optional[float] = None,
    curve_strength: Optional[float] = None,
) -> None:
    """
    Clicks at screen coordinates with optional randomisation.

    Args:
        x: Base X coordinate.
        y: Base Y coordinate.
        x_rand: Max random pixel offset on the X axis.
        y_rand: Max random pixel offset on the Y axis.
        shift: Hold Shift while clicking.
        duration: Seconds for the cursor movement.  Defaults to a random
            value between 0.30 and 0.50 s.
        curve_strength: How far the cursor path deviates from a straight
            line (0 = straight, 1 = very curved).

    Raises:
        ValueError: If the base coordinates are outside the screen.
    """
    screen_w, screen_h = _screen_size()
    if not (0 <= x <= screen_w and 0 <= y <= screen_h):
        raise ValueError(f"Coordinates ({x}, {y}) are outside screen bounds.")

    if duration is None:
        duration = _human_delay(mu=0.40, sigma=0.15, lo=0.08, hi=1.2)
    if curve_strength is None:
        curve_strength = uniform(0.1, 0.3)

    # Biased offset: slight overshoot from current cursor toward target,
    # plus dominant-hand rightward drift.  Not perfectly symmetric.
    mx, my = pyautogui.position()
    dir_x = 1 if x >= mx else -1
    dir_y = 1 if y >= my else -1
    bias_x = dir_x * uniform(0, x_rand * 0.3) if x_rand else 0
    bias_y = dir_y * uniform(0, y_rand * 0.3) if y_rand else 0
    jitter_x = gauss(0, max(x_rand * 0.5, 1)) if x_rand else 0
    jitter_y = gauss(0, max(y_rand * 0.5, 1)) if y_rand else 0
    target_x = _clamp(x + int(round(bias_x + jitter_x)), 0, screen_w - 1)
    target_y = _clamp(y + int(round(bias_y + jitter_y)), 0, screen_h - 1)

    _move_to(target_x, target_y, duration=duration, curve_strength=curve_strength)

    if shift:
        _send_scan(_SCAN_CODES["shift"])
        time.sleep(0.05)
        _click_mouse()
        time.sleep(0.03)
        _send_scan(_SCAN_CODES["shift"], key_up=True)
    else:
        _click_mouse()


def get_mouse_coordinates(
    verbose: bool = True,
    key: Union[keyboard.Key, Tuple] = (keyboard.Key.shift, keyboard.Key.shift_r),
) -> Optional[Tuple[int, int]]:
    """
    Blocks until the user presses *key*, then returns the mouse position.

    Args:
        verbose: Print instructions.
        key: Key (or tuple of keys) that triggers the capture.

    Returns:
        ``(x, y)`` coordinates, or ``None`` if the capture fails.
    """
    if verbose:
        logger.info("Move mouse to the desired location, then press %s.", key)

    result: List[Optional[Tuple[int, int]]] = [None]

    def on_press(pressed):
        if pressed == key or (isinstance(key, tuple) and pressed in key):
            result[0] = pyautogui.position()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    return result[0]


def get_region(
    verbose: bool = True,
    key: Union[keyboard.Key, Tuple] = (keyboard.Key.shift, keyboard.Key.shift_r),
) -> Region:
    """
    Captures a screen region by recording two opposite corners via key presses.

    Args:
        verbose: Print instructions.
        key: Key (or tuple of keys) used to capture each corner.

    Returns:
        A ``Region`` dataclass.

    Raises:
        ValueError: If both corners are the same point.
    """
    corners: list = []
    count = 0

    if verbose:
        logger.info("Press %s at the first corner, then at the opposite corner.", key)

    def on_press(pressed):
        nonlocal count
        if pressed == key or (isinstance(key, tuple) and pressed in key):
            pos = pyautogui.position()
            corners.append(pos)
            count += 1
            logger.info("Corner %d captured.", count)
            if count >= 2:
                return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    x1, y1 = corners[0]
    x2, y2 = corners[1]

    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x1 - x2)
    height = abs(y1 - y2)

    if width == 0 or height == 0:
        raise ValueError("Region dimensions must be positive - corners cannot overlap.")

    return Region(left, top, width, height)


def wait_for_image(
    needle_image_path: str,
    region: Optional[Union[dict, Region]] = None,
    appear: bool = True,
    confidence: float = 0.8,
    timeout: float = 10,
    poll_interval: float = 0.2,
    verbose: bool = True,
) -> Optional[Tuple[float, float]]:
    """
    Waits for an image to appear or disappear on screen.

    Args:
        needle_image_path: Path to the template image file.
        region: Search area as a ``Region`` instance or a dict with keys
            ``left``, ``top``, ``width``, ``height``.  ``None`` searches the
            entire screen.
        appear: ``True`` to wait for the image to *appear*; ``False`` to wait
            for it to *disappear*.
        confidence: Match confidence threshold (0.0 - 1.0).
        timeout: Maximum seconds to wait.  ``0`` means wait indefinitely.
        poll_interval: Seconds between screen captures.

    Returns:
        ``(center_x, center_y)`` when the image appeared, or ``None`` if the
        timeout expired.  When *appear* is ``False``, returns ``(0.0, 0.0)``
        as a truthy sentinel once the image has gone.
    """
    search_region: Optional[Tuple[int, int, int, int]] = None

    if region is not None:
        if isinstance(region, Region):
            search_region = region.as_tuple()
        elif isinstance(region, dict):
            try:
                search_region = (
                    region["left"],
                    region["top"],
                    region["width"],
                    region["height"],
                )
            except KeyError as exc:
                raise KeyError(f"Region dict missing required key: {exc}") from exc
        else:
            raise TypeError("region must be a Region, dict, or None.")

    start = time.monotonic()

    while timeout == 0 or (time.monotonic() - start) < timeout:
        try:
            location = pyautogui.locateOnScreen(
                needle_image_path,
                region=search_region,
                confidence=confidence,
            )
            if appear and location:
                cx = location.left + location.width / 2
                cy = location.top + location.height / 2
                return (cx, cy)
        except pyautogui.ImageNotFoundException:
            if not appear:
                return (0.0, 0.0)

        time.sleep(poll_interval)

    if verbose:
        logger.warning(
            "Timeout after %.1f s waiting for image to %s.",
            timeout,
            "appear" if appear else "disappear",
        )
    return None
