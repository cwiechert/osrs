# osrs_utils

A Python utility library for automating screen interactions using computer vision and input control. Built for use with Old School RuneScape (OSRS), but usable for any screen automation task.

## Installation

```bash
pip install git+https://github.com/cwiechert/osrs.git
```

## Requirements

- Python 3.10+
- `mss` — fast screen capture
- `numpy` — image array processing
- `opencv-python` — computer vision (HSV filtering, contour detection)
- `PyAutoGUI` — mouse/keyboard control and image search
- `pynput` — keyboard and mouse event listeners

---

## API Reference

### `click(x, y, ...)`

Moves the mouse to `(x, y)` and clicks. Supports randomization and Shift+click.

```python
from osrslib import click

click(960, 540)                          # Simple click at center
click(960, 540, x_rand=5, y_rand=5)     # Click with ±5px random offset
click(960, 540, shift=True)             # Shift+click
click(960, 540, duration=0.4)           # Custom move duration in seconds
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `int` | required | X coordinate |
| `y` | `int` | required | Y coordinate |
| `x_rand` | `int` | `0` | Max random pixel offset on X |
| `y_rand` | `int` | `0` | Max random pixel offset on Y |
| `shift` | `bool` | `False` | Hold Shift while clicking |
| `duration` | `float` | random 0.3–0.5s | Mouse movement duration |

---

### `get_mouse_coordinates(verbose=True, key=...)`

Blocks until the user presses the specified key, then returns the current mouse position.

```python
from osrslib import get_mouse_coordinates
from pynput import keyboard

x, y = get_mouse_coordinates()                              # Default: either Shift key
x, y = get_mouse_coordinates(key=keyboard.Key.ctrl_r)      # Custom: Right Ctrl
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Print instructions to the console |
| `key` | `Key` or `tuple` | `(shift, shift_r)` | Key or tuple of keys that trigger the capture |

---

### `get_region(verbose=True, key=...)`

Captures a rectangular screen region by recording two key presses (the two opposite corners). Order does not matter.

```python
from osrslib import get_region
from pynput import keyboard

region = get_region()                                       # Default: either Shift key
region = get_region(key=keyboard.Key.ctrl_r)               # Custom: Right Ctrl
# {'left': 100, 'top': 200, 'width': 400, 'height': 300}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Print instructions to the console |
| `key` | `Key` or `tuple` | `(shift, shift_r)` | Key or tuple of keys used to capture each corner |

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
    iterations=5,       # Repeat 5 times
    move_duration=0.2,  # Seconds to move mouse between points
    x_rand=3,           # ±3 pixel random offset on X
    y_rand=3,           # ±3 pixel random offset on Y
    verbose=True        # Print timing info per iteration
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
| `move_duration` | `float` | `0.1` | Seconds for mouse to travel between points |
| `x_rand` | `int` | `0` | Max random pixel offset on X axis |
| `y_rand` | `int` | `0` | Max random pixel offset on Y axis |
| `verbose` | `bool` | `True` | Print actual/average/original timing per iteration |

---

### `RegionHSV`

Detects objects on screen by filtering a region using an HSV color range. Returns the screen coordinates of each detected object.

**Typical workflow:**
1. Instantiate the class.
2. Call `configure()` to visually tune the region and HSV values.
3. Call `get_centers()` to get the (x, y) coordinates of detected objects.

```python
from osrslib import RegionHSV

detector = RegionHSV()

# Open GUI to set screen region and HSV color range
detector.configure()

# Get the screen coordinates of all detected objects
centers = detector.get_centers()
print(centers)  # [(x1, y1), (x2, y2), ...]

# Close resources when done
detector.close()
```

#### `__init__` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | Print status messages to the console |
| `play_pause_key` | `str` | `'+'` | Key to pause/resume concurrent tasks |
| `stop_key` | `str` | `'}'` | Key to stop concurrent tasks |

#### Methods

| Method | Description |
|--------|-------------|
| `configure(quit_key='q')` | Opens a GUI with sliders to set the capture region and HSV range. Press `quit_key` to save and close. |
| `get_centers()` | Captures one frame, applies the HSV filter, and returns a list of `(x, y)` center coordinates for each detected object. |
| `draw_centers(show_hsv_mask=False, quit_key='q')` | Opens a real-time window showing detections. Press `quit_key` to close. Pass `show_hsv_mask=True` to see the filtered black-and-white view. |
| `concurrent_tasks(fn)` | Runs `get_centers` (moving the mouse to the average center) and your `fn` in parallel threads. Controlled by `play_pause_key` and `stop_key`. |
| `kill_concurrency()` | Stops all concurrent tasks immediately. |
| `close()` | Releases the screen capture object and closes any open windows. |

#### Attributes set after `configure()`

| Attribute | Type | Description |
|-----------|------|-------------|
| `region` | `dict` | `{'left', 'top', 'width', 'height'}` — the capture area |
| `lower_bound` | `np.array` | Lower HSV bound `[H, S, V]` |
| `upper_bound` | `np.array` | Upper HSV bound `[H, S, V]` |
| `centers` | `list` | Last result from `get_centers()` |

---

### `wait_for_image(needle_image_path, ...)`

Waits for a reference image to appear or disappear on screen. Useful for waiting on game events before proceeding.

```python
from osrslib import wait_for_image

# Wait for image to appear (returns center coordinates or False on timeout)
result = wait_for_image('images/bank.png', region=region, appear=True, timeout=15)
if result:
    x, y = result
    print(f"Found at ({x}, {y})")

# Wait for image to disappear (returns True when gone, False on timeout)
gone = wait_for_image('images/loading.png', appear=False, timeout=30)

# Run indefinitely until found (timeout=0)
result = wait_for_image('images/enemy.png', timeout=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `needle_image_path` | `str` | required | Path to the reference image |
| `region` | `dict` | `None` | Search area `{'left', 'top', 'width', 'height'}`. Searches full screen if `None`. |
| `appear` | `bool` | `True` | `True` waits for the image to appear, `False` waits for it to disappear |
| `confidence` | `float` | `0.8` | Match confidence threshold (0.0–1.0) |
| `timeout` | `int` | `10` | Max seconds to wait. `0` runs indefinitely. |
| `verbose` | `bool` | `True` | Print a message if timeout is reached |

**Returns:**
- `(x, y)` — center of matched image if `appear=True` and found
- `True` — if `appear=False` and image disappeared
- `False` — if timeout was reached

---

## Project Structure

```
osrs/
├── osrslib/
│   ├── __init__.py
│   └── osrs.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
