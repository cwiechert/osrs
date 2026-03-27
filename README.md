# osrs_utils

A Python utility library for automating screen interactions using computer vision and low-level input simulation. Built for use with Old School RuneScape (OSRS), but usable for any screen automation task.

All mouse movement uses **Bezier curves** for natural-looking paths and **hardware scan codes** via `SendInput` for input that is recognized by DirectInput applications.

## Installation

```bash
pip install git+https://github.com/cwiechert/osrs.git
```

## Requirements

- Python 3.10+
- Windows (uses `ctypes.windll` for low-level input)
- `mss` — fast screen capture
- `numpy` — image array processing
- `opencv-python` — computer vision (HSV filtering, contour detection)
- `PyAutoGUI` — image search and screen size queries
- `pynput` — keyboard and mouse event listeners

---

## Emergency Stop

Since the library bypasses `pyautogui`'s failsafe, it provides its own global kill switch. Call once at the start of your script:

```python
from osrslib import enable_failsafe
from pynput import keyboard

enable_failsafe()                          # Default: F6 kills the process
enable_failsafe(key=keyboard.Key.f8)       # Custom key
```

Runs in a daemon thread — works even if the main thread is blocked or in a busy-wait loop. Pressing the key calls `os._exit(1)` immediately.

---

## Data Types

These dataclasses are used throughout the library.

### `Region`

Represents a rectangular area on screen.

```python
from osrslib import Region

r = Region(left=0, top=53, width=980, height=1363)

r.left          # 0
r.as_dict()     # {'left': 0, 'top': 53, 'width': 980, 'height': 1363}
r.as_tuple()    # (0, 53, 980, 1363)
```

### `HSVRange`

Encapsulates the lower and upper HSV bounds for color filtering.

```python
from osrslib import HSVRange

hsv = HSVRange(hue_min=81, hue_max=94, sat_min=250, sat_max=255, val_min=247, val_max=255)

hsv.lower  # np.array([81, 250, 247])
hsv.upper  # np.array([94, 255, 255])
```

### `TargetStrategy`

Enum that controls how `RegionHSV` picks a single target when multiple objects are detected.

```python
from osrslib import TargetStrategy

TargetStrategy.NEAREST   # Closest to current mouse position (default)
TargetStrategy.FIRST     # First contour returned by OpenCV
TargetStrategy.LARGEST   # Contour with the biggest area
TargetStrategy.CENTROID  # Average of all centers
```

---

## API Reference

### `click(x, y, ...)`

Moves the mouse to `(x, y)` along a randomized Bezier curve and clicks using `SendInput`. Supports randomization and Shift+click.

```python
from osrslib import click

click(960, 540)                                        # Simple click
click(960, 540, x_rand=5, y_rand=5)                   # ±5px random offset
click(960, 540, shift=True)                            # Shift+click
click(960, 540, duration=0.4)                          # Custom move duration
click(960, 540, curve_strength=0.5)                    # More curved path
click(960, 540, duration=1.0, curve_strength=0)        # Straight line, precise 1s
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `int` | required | X coordinate |
| `y` | `int` | required | Y coordinate |
| `x_rand` | `int` | `0` | Max random pixel offset on X |
| `y_rand` | `int` | `0` | Max random pixel offset on Y |
| `shift` | `bool` | `False` | Hold Shift while clicking |
| `duration` | `float` | random 0.3-0.5s | Mouse movement duration |
| `curve_strength` | `float` | random 0.1-0.3 | Bezier curve deviation (0 = straight line, 1 = very curved) |

Mouse movement uses `ctypes.windll.user32.SetCursorPos` with a cubic Bezier curve and `time.perf_counter` busy-wait for sub-millisecond timing accuracy. Mouse clicks use `SendInput` with proper down/up events and a randomized hold time (40-90ms).

---

### `press_key(key, ...)`

Simulates a physical key press using hardware scan codes via `SendInput` with `KEYEVENTF_SCANCODE`. Recognized by applications that read raw/DirectInput (unlike `pyautogui.press()` which sends virtual-key events).

```python
from osrslib import press_key

press_key("1")                      # Hold time: random 40-90ms
press_key("space", hold_time=0.2)   # Fixed 200ms hold
press_key("f1")                     # Function keys
press_key("enter")                  # Named keys
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | required | Key name: `"a"`-`"z"`, `"0"`-`"9"`, or named key |
| `hold_time` | `float` | random 0.04-0.09s | Seconds to hold the key before releasing |

**Available named keys:** `space`, `enter`, `esc`, `tab`, `backspace`, `shift`, `ctrl`, `alt`, `f1`-`f12`

---

### `get_mouse_coordinates(verbose=True, key=...)`

Blocks until the user presses the specified key, then returns the current mouse position.

```python
from osrslib import get_mouse_coordinates
from pynput import keyboard

coords = get_mouse_coordinates()                              # Default: either Shift key
coords = get_mouse_coordinates(key=keyboard.Key.ctrl_r)      # Custom: Right Ctrl

if coords:
    x, y = coords
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Print instructions to the console |
| `key` | `Key` or `tuple` | `(shift, shift_r)` | Key or tuple of keys that trigger the capture |

**Returns:** `(x, y)` tuple, or `None` if the capture fails.

---

### `get_region(verbose=True, key=...)`

Captures a rectangular screen region by recording two key presses (the two opposite corners). Order does not matter.

```python
from osrslib import get_region
from pynput import keyboard

region = get_region()                                       # Default: either Shift key
region = get_region(key=keyboard.Key.ctrl_r)               # Custom: Right Ctrl

region.left     # 100
region.width    # 400
region.as_dict()  # {'left': 100, 'top': 200, 'width': 400, 'height': 300}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Print instructions to the console |
| `key` | `Key` or `tuple` | `(shift, shift_r)` | Key or tuple of keys used to capture each corner |

**Returns:** A `Region` dataclass.

---

### `Recorder`

Records mouse click events to a CSV file and replays them with accurate timing and optional randomization.

#### Recording

```python
from osrslib import Recorder
from pynput import keyboard

# Default: press Left Ctrl to stop recording
recorder = Recorder(record=True, filename='my_clicks.csv')

# Custom stop key: Right Shift
recorder = Recorder(record=True, stop_key=keyboard.Key.shift_r)

recorder.record_and_save()
```

#### Playback

```python
from osrslib import Recorder

recorder = Recorder(record=False, filename='my_clicks.csv')
recorder.reproduce(
    iterations=5,          # Repeat 5 times
    x_rand=3,              # ±3 pixel random offset on X
    y_rand=3,              # ±3 pixel random offset on Y
    move_fraction=0.6,     # Use 60% of each gap for movement, 40% idle
    strict_timing=True,    # Shorten moves if behind schedule
    verbose=True           # Print timing info per iteration
)
```

#### `__init__` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `record` | `bool` | `False` | `True` to record, `False` to load and replay |
| `filename` | `str` | `'mouse_record.csv'` | CSV file to save to or load from |
| `stop_key` | `keyboard.Key` | `keyboard.Key.ctrl_l` | Key that stops the recording |

#### `reproduce()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | `int` | `1` | Number of times to repeat the sequence |
| `x_rand` | `int` | `0` | Max random pixel offset on X axis |
| `y_rand` | `int` | `0` | Max random pixel offset on Y axis |
| `move_fraction` | `float` | `0.6` | Fraction of each inter-event gap used for cursor movement (0.0-1.0). At 0.6 the cursor moves for 60% of the gap and idles for 40%. |
| `strict_timing` | `bool` | `True` | When `True`, if a movement overshoots its time budget the next move duration is shortened to stay on schedule. When `False`, overshoots are ignored and the sequence may drift. |
| `verbose` | `bool` | `True` | Print actual/average/original timing per iteration |

---

### `RegionHSV`

Detects objects on screen by filtering a region using an HSV color range. Returns the screen coordinates of each detected object.

Supports the context manager protocol for automatic resource cleanup.

**Typical workflow:**

```python
from osrslib import RegionHSV

with RegionHSV() as detector:
    # 1. Open GUI to set screen region and HSV color range
    detector.configure()

    # 2. Export settings to clipboard for reuse
    detector.export()

    # 3. Get the screen coordinates of all detected objects
    centers = detector.get_centers()
    print(centers)  # [(x1, y1), (x2, y2), ...]
```

**Reusing exported settings** (paste from `export()`):

```python
from osrslib import RegionHSV, Region, HSVRange

r = RegionHSV(min_contour_area=50)
r.region = Region(left=0, top=53, width=980, height=1363)
r.hsv = HSVRange(hue_min=81, hue_max=94, sat_min=250, sat_max=255, val_min=247, val_max=255)

centers = r.get_centers()
```

#### `__init__` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Log status messages |
| `play_pause_key` | `str` | `'+'` | Key to pause/resume concurrent tasks |
| `stop_key` | `str` | `'}'` | Key to stop concurrent tasks |
| `min_contour_area` | `float` | `0.0` | Minimum contour area in px² to count as a detection. Filters out noise. |
| `target_strategy` | `TargetStrategy` | `NEAREST` | How to pick one target when multiple objects are detected (used by `concurrent_tasks`). |

#### Methods

| Method | Description |
|--------|-------------|
| `configure(quit_key='q')` | Opens a GUI with sliders to set the capture region and HSV range. Press `quit_key` to save and close. |
| `export(var_name='r')` | Copies a paste-ready code snippet with the current region and HSV settings to the clipboard. Returns the snippet as a string. |
| `get_centers()` | Captures one frame, applies the HSV filter, and returns a list of `(x, y)` center coordinates for each detected object. |
| `draw_centers(show_hsv_mask=False, quit_key='q')` | Opens a real-time window showing detections. Press `quit_key` to close. Pass `show_hsv_mask=True` to see the filtered black-and-white view. |
| `concurrent_tasks(fn)` | Runs detection (moving the mouse to a target chosen by `target_strategy`) and your `fn` in parallel threads. Controlled by `play_pause_key` and `stop_key`. |
| `kill_concurrency()` | Stops all concurrent tasks immediately. |
| `close()` | Stops concurrent tasks, releases the screen capture object, and closes any open windows. Called automatically when using `with`. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `region` | `Region` | The capture area. Set via `configure()` or manually. |
| `hsv` | `HSVRange` | The HSV filter bounds. Set via `configure()` or manually. |
| `centers` | `list` | Last result from `get_centers()`. Thread-safe (returns a copy). |
| `active` | `bool \| None` | `True` if concurrent tasks are running, `False` if paused/stopped, `None` before first run. |

---

### `wait_for_image(needle_image_path, ...)`

Waits for a reference image to appear or disappear on screen. Useful for waiting on game events before proceeding.

```python
from osrslib import wait_for_image

# Wait for image to appear (returns center coordinates or None on timeout)
result = wait_for_image('images/bank.png', region=region, appear=True, timeout=15)
if result:
    x, y = result
    print(f"Found at ({x}, {y})")

# Wait for image to disappear (returns (0.0, 0.0) when gone, None on timeout)
gone = wait_for_image('images/loading.png', appear=False, timeout=30)

# Run indefinitely until found (timeout=0)
result = wait_for_image('images/enemy.png', timeout=0)

# Custom poll speed for fast-changing content
result = wait_for_image('images/flash.png', poll_interval=0.05)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `needle_image_path` | `str` | required | Path to the reference image |
| `region` | `Region`, `dict`, or `None` | `None` | Search area. Accepts a `Region` dataclass or a dict with `left`, `top`, `width`, `height`. Searches full screen if `None`. |
| `appear` | `bool` | `True` | `True` waits for the image to appear, `False` waits for it to disappear |
| `confidence` | `float` | `0.8` | Match confidence threshold (0.0-1.0) |
| `timeout` | `float` | `10` | Max seconds to wait. `0` runs indefinitely. |
| `poll_interval` | `float` | `0.2` | Seconds between screen captures |

**Returns:**
- `(x, y)` — center of matched image if `appear=True` and found
- `(0.0, 0.0)` — if `appear=False` and image disappeared (truthy sentinel)
- `None` — if timeout was reached

---

## Project Structure

```
osrs/
├── osrslib/
│   ├── __init__.py
│   └── osrs.py
├── scripts/
├── pyproject.toml
├── requirements.txt
└── README.md
```
