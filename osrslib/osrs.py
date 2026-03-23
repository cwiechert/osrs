import csv
import logging
import platform
import subprocess
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from random import randint
from typing import Optional, Union, Tuple, Callable, List

import cv2
import numpy as np
import pyautogui
from mss import mss
from pynput import keyboard, mouse


# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

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

                        pyautogui.moveTo(target[0], target[1], duration=refresh_rate)
                    except Exception:
                        logger.exception("Error in _real_time_coordinates loop.")
                        time.sleep(refresh_rate)
        except Exception:
            logger.exception("Fatal error in _real_time_coordinates thread.")

    def concurrent_tasks(
        self,
        external_function: Callable[[], None],
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
            args=(self._run_event, self._stop_event),
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
    """A single recorded mouse-click event."""

    timestamp: float
    x: int
    y: int
    button: str  # "left" | "right"


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
    _CSV_FIELDS = ("timestamps", "x_axis", "y_axis", "button")

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
                required = set(self._CSV_FIELDS)
                if not required.issubset(reader.fieldnames or []):
                    missing = required - set(reader.fieldnames or [])
                    raise ValueError(f"CSV missing required columns: {missing}")

                self.events = [
                    ClickEvent(
                        timestamp=float(row["timestamps"]),
                        x=int(row["x_axis"]),
                        y=int(row["y_axis"]),
                        button=row["button"],
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

    def record_and_save(self) -> None:
        """
        Begins recording mouse clicks until ``self.stop_key`` is pressed.

        Raises:
            RuntimeError: If called when not in recording mode.
        """
        if not self.record:
            raise RuntimeError(
                "Not in recording mode. Instantiate with record=True."
            )

        logger.info("Recording mouse clicks… Press '%s' to stop.", self.stop_key)

        mouse_listener = mouse.Listener(on_click=self._on_click)
        keyboard_listener = keyboard.Listener(on_press=self._on_press)

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

    # ── Playback ─────────────────────────────────────────────────────────
    def reproduce(
        self,
        iterations: int = 1,
        move_duration: float = 0.1,
        x_rand: int = 0,
        y_rand: int = 0,
        verbose: bool = True,
    ) -> None:
        """
        Replays the recorded click sequence with accurate timing.

        Args:
            iterations: How many times to replay the full sequence.
            move_duration: Seconds the cursor takes to travel between points.
            x_rand: Maximum random pixel offset on the X axis.
            y_rand: Maximum random pixel offset on the Y axis.
            verbose: Print per-iteration timing stats.

        Raises:
            ValueError: On invalid parameter values.
        """
        if not self.events:
            logger.warning("No recording data to reproduce.")
            return

        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        if move_duration < 0 or x_rand < 0 or y_rand < 0:
            raise ValueError("move_duration, x_rand, and y_rand must be >= 0.")

        screen_w, screen_h = pyautogui.size()
        base_time = self.events[0].timestamp
        relative_times = [ev.timestamp - base_time for ev in self.events]

        iteration_durations: List[float] = []

        if verbose:
            logger.info("Starting playback (%d iteration(s))…", iterations)

        for q in range(iterations):
            iter_start = time.perf_counter()

            for rel_time, ev in zip(relative_times, self.events):
                target_time = iter_start + rel_time

                rand_x = _clamp(ev.x + randint(-x_rand, x_rand), 0, screen_w - 1)
                rand_y = _clamp(ev.y + randint(-y_rand, y_rand), 0, screen_h - 1)

                move_start = target_time - move_duration
                wait = move_start - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

                pyautogui.moveTo(rand_x, rand_y, duration=move_duration)

                final_wait = target_time - time.perf_counter()
                if final_wait > 0:
                    time.sleep(final_wait)

                pyautogui.click(button=ev.button)

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


# ── Standalone helpers ───────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamps *value* to the inclusive range [lo, hi]."""
    return max(lo, min(value, hi))


def click(
    x: int,
    y: int,
    x_rand: int = 0,
    y_rand: int = 0,
    shift: bool = False,
    duration: Optional[float] = None,
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

    Raises:
        ValueError: If the base coordinates are outside the screen.
    """
    screen_w, screen_h = pyautogui.size()
    if not (0 <= x <= screen_w and 0 <= y <= screen_h):
        raise ValueError(f"Coordinates ({x}, {y}) are outside screen bounds.")

    if duration is None:
        duration = randint(30, 50) / 100

    target_x = _clamp(x + randint(-x_rand, x_rand), 0, screen_w - 1)
    target_y = _clamp(y + randint(-y_rand, y_rand), 0, screen_h - 1)

    pyautogui.moveTo(target_x, target_y, duration=duration)

    if shift:
        pyautogui.keyDown("shift")
        pyautogui.click()
        pyautogui.keyUp("shift")
    else:
        pyautogui.click()


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
