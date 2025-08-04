import time
import os
import csv
from random import randint
from typing import Optional, Union, Tuple

import cv2
import numpy as np
import pandas as pd
import pyautogui
from mss import mss
from pynput import keyboard, mouse

# Window Names
TRACKBAR_WINDOW_NAME = 'Trackbars'
CAPTURE_WINDOW_NAME = 'Screen Capture'
RESULT_WINDOW_NAME = 'Detection Result'

# Default Region Values
DEFAULT_LEFT = 0
DEFAULT_TOP = 0
DEFAULT_WIDTH = 300
DEFAULT_HEIGHT = 200

# Default HSV Values
DEFAULT_HUE_MIN = 0
DEFAULT_HUE_MAX = 179
DEFAULT_SAT_MIN = 0
DEFAULT_SAT_MAX = 255
DEFAULT_VAL_MIN = 0
DEFAULT_VAL_MAX = 255


class RegionHSV:
    """
    Detects objects within a specified screen region based on a range of HSV color values.

    The intended workflow is:
    1. Instantiate the class.
    2. (Optional) Call `configure_interactively()` to use a GUI to find the
       correct screen region and HSV values.
    3. Call `get_object_centers()` to get the coordinates of detected objects or
       `draw_centers()` to see the detections in real-time.
    """


    def __init__(self, verbose: bool = True):
        """
        Initializes the RegionHSV detector.
        
        Args:
            verbose (bool): If True, prints status messages to the console.
        """
        self.verbose = verbose
        self.sct = mss()
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Initialize region parameters
        self.left = DEFAULT_LEFT
        self.top = DEFAULT_TOP
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.region = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

        # Initialize HSV parameters
        self.hue_min = DEFAULT_HUE_MIN
        self.hue_max = DEFAULT_HUE_MAX
        self.sat_min = DEFAULT_SAT_MIN
        self.sat_max = DEFAULT_SAT_MAX
        self.val_min = DEFAULT_VAL_MIN
        self.val_max = DEFAULT_VAL_MAX
        
        # Main attributes to be configured
        self.lower_bound = np.array([self.hue_min, self.sat_min, self.val_min])
        self.upper_bound = np.array([self.hue_max, self.sat_max, self.val_max])
        self.centers = []


    def _create_trackbars(self, width, height):
        """Creates a window with trackbars for adjusting parameters."""
        cv2.namedWindow(TRACKBAR_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(TRACKBAR_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(TRACKBAR_WINDOW_NAME, width, height)

        # Create trackbars for region
        cv2.createTrackbar('Left', TRACKBAR_WINDOW_NAME, self.left, self.screen_width, lambda x: None)
        cv2.createTrackbar('Top', TRACKBAR_WINDOW_NAME, self.top, self.screen_height, lambda x: None)
        cv2.createTrackbar('Width', TRACKBAR_WINDOW_NAME, self.width, self.screen_width, lambda x: None)
        cv2.createTrackbar('Height', TRACKBAR_WINDOW_NAME, self.height, self.screen_height, lambda x: None)
        
        # Create trackbars for HSV
        cv2.createTrackbar('Hue Min', TRACKBAR_WINDOW_NAME, self.hue_min, 179, lambda x: None)
        cv2.createTrackbar('Hue Max', TRACKBAR_WINDOW_NAME, self.hue_max, 179, lambda x: None)
        cv2.createTrackbar('Sat Min', TRACKBAR_WINDOW_NAME, self.sat_min, 255, lambda x: None)
        cv2.createTrackbar('Sat Max', TRACKBAR_WINDOW_NAME, self.sat_max, 255, lambda x: None)
        cv2.createTrackbar('Val Min', TRACKBAR_WINDOW_NAME, self.val_min, 255, lambda x: None)
        cv2.createTrackbar('Val Max', TRACKBAR_WINDOW_NAME, self.val_max, 255, lambda x: None)


    def _update_params_from_trackbars(self):
        """Updates instance parameters based on current trackbar values."""
        try:
            # Update region
            self.left = cv2.getTrackbarPos('Left', TRACKBAR_WINDOW_NAME)
            self.top = cv2.getTrackbarPos('Top', TRACKBAR_WINDOW_NAME)
            self.width = max(1, cv2.getTrackbarPos('Width', TRACKBAR_WINDOW_NAME))
            self.height = max(1, cv2.getTrackbarPos('Height', TRACKBAR_WINDOW_NAME))
            
            # Update HSV
            self.hue_min = cv2.getTrackbarPos('Hue Min', TRACKBAR_WINDOW_NAME)
            self.hue_max = cv2.getTrackbarPos('Hue Max', TRACKBAR_WINDOW_NAME)
            self.sat_min = cv2.getTrackbarPos('Sat Min', TRACKBAR_WINDOW_NAME)
            self.sat_max = cv2.getTrackbarPos('Sat Max', TRACKBAR_WINDOW_NAME)
            self.val_min = cv2.getTrackbarPos('Val Min', TRACKBAR_WINDOW_NAME)
            self.val_max = cv2.getTrackbarPos('Val Max', TRACKBAR_WINDOW_NAME)
        except cv2.error:
            if self.verbose:
                print("Trackbars closed. Using last known values.")
            return

        # Validate region boundaries
        self.left = max(0, min(self.left, self.screen_width - 1))
        self.top = max(0, min(self.top, self.screen_height - 1))
        self.width = min(self.width, self.screen_width - self.left)
        self.height = min(self.height, self.screen_height - self.top)


    def _process_frame(self):
        """
        Captures a single frame and processes it to find object contours.
        
        Returns:
            A tuple containing (contours, original_frame, hsv_mask).
        """
        screenshot = self.sct.grab(self.region)
        frame = np.array(screenshot)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, frame, mask


    def configure(self, window_width=500, window_height=380):
        """
        Opens a GUI to allow real-time adjustment of the screen region and HSV values.
        
        Press 'q' to close the windows and save the settings.

        :param window_width: Determines the trackbar window width.
        :param window_height: Determines the trackbar window height.
        """
        self._create_trackbars(width=window_width, height=window_height)
        cv2.namedWindow(CAPTURE_WINDOW_NAME, cv2.WINDOW_NORMAL)

        if self.verbose:
            print("Adjust parameters in the 'Trackbars' window.")
            print(f"Press 'q' in the '{CAPTURE_WINDOW_NAME}' window to finish configuration.")
            
        while True:
            self._update_params_from_trackbars()
            
            self.region = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}
            self.lower_bound = np.array([self.hue_min, self.sat_min, self.val_min])
            self.upper_bound = np.array([self.hue_max, self.sat_max, self.val_max])
            
            _, frame, mask = self._process_frame()
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            cv2.resizeWindow(CAPTURE_WINDOW_NAME, self.width, self.height)
            cv2.imshow(CAPTURE_WINDOW_NAME, result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        if self.verbose:
            print("Configuration complete. Ready to detect objects.")


    def get_centers(self) -> list[tuple[int, int]]:
        """
        Finds all objects matching the HSV criteria in the region and returns their centers.

        Returns:
            A list of (x, y) tuples for each object's center. Returns an empty list
            if no objects are found.
        """
        contours, _, _ = self._process_frame()
        self.centers = []
        
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + self.region['left']
                cy = int(M['m01'] / M['m00']) + self.region['top']
                self.centers.append((cx, cy))
        
        if self.verbose:
            print(f"Found {len(self.centers)} object(s).")
        
        return self.centers


    def draw_centers(self, show_hsv_mask: bool = False):
        """
        Displays a real-time feed of the capture region with detected centers drawn on it.
        
        Press 'q' to close the window.
        
        Args:
            show_hsv_mask (bool): If True, draws centers on the black-and-white HSV mask.
                                  Otherwise, draws on the original color frame.
        """
        cv2.namedWindow(RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.verbose:
            print(f"Displaying real-time detections. Press 'q' in the '{RESULT_WINDOW_NAME}' window to stop.")

        while True:
            contours, frame, mask = self._process_frame()
            display_image = cv2.bitwise_and(frame, frame, mask=mask) if show_hsv_mask else frame
            
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    # Coordinates are relative to the smaller frame, not the whole screen
                    cx_local = int(M['m10'] / M['m00'])
                    cy_local = int(M['m01'] / M['m00'])
                    cv2.circle(display_image, (cx_local, cy_local), 5, (0, 255, 0), -1)

            cv2.resizeWindow(RESULT_WINDOW_NAME, self.width, self.height)
            cv2.imshow(RESULT_WINDOW_NAME, display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        
    def close(self):
        """Closes any open resources, like the screen capture object."""
        self.sct.close()
        cv2.destroyAllWindows()
        if self.verbose:
            print("RegionHSV resources closed.")


class Recorder:
    '''
    A class to record and reproduce mouse click actions on the screen.

    This class allows users to:
    - Record mouse click events (coordinates, timestamps, button).
    - Save the recorded data to a CSV file.
    - Reproduce the recorded clicks with proper timing and optional randomization.

    :param record: Boolean flag for recording mode (`True`) or playback mode (`False`). Defaults to `False`.
    :param filename: Name of the CSV file to save or read data. Defaults to 'mouse_record.csv'.

    Attributes:
    - record (bool): Indicates if the instance is in recording or playback mode.
    - filename (str): Path to the data file.
    - times_ (list): Timestamps of mouse click events.
    - coordinates_ (list): (x, y) screen coordinates for each click.
    - button_ (list): Mouse button ('left', 'right') used for each click.

    Usage:
    - **Recording Mode**: 
        recorder = Recorder(record=True, filename='my_recording.csv')
        # To stop, press the Right Shift key instead of the default Left Control
        recorder.record_and_save(stop_key=keyboard.Key.shift_r)
        
    - **Playback Mode**: 
        recorder = Recorder(record=False, filename='my_recording.csv')
        recorder.reproduce(iterations=3, move_duration=0.2, x_rand=5, y_rand=5)
    '''
    

    def __init__(self, record: bool = False, filename: str = 'mouse_record.csv'):
        self.record = record
        self.filename = filename

        if self.record:
            self.times_ = []
            self.coordinates_ = []
            self.button_ = []
        else:
            self._load_recording()


    def _load_recording(self) -> None:
        """Load mouse recording data from a CSV file."""
        try:
            with open(self.filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                required_cols = {'timestamps', 'x_axis', 'y_axis', 'button'}
                if not required_cols.issubset(reader.fieldnames or []):
                    missing = required_cols - set(reader.fieldnames or [])
                    raise ValueError(f"CSV missing required columns: {missing}")

                self.times_ = []
                self.coordinates_ = []
                self.button_ = []
                for row in reader:
                    self.times_.append(float(row['timestamps']))
                    self.coordinates_.append((int(row['x_axis']), int(row['y_axis'])))
                    self.button_.append(row['button'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Recording file '{self.filename}' not found.")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error reading CSV file. It may be malformed. Details: {e}")


    def on_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Mouse click event handler for recording."""
        if pressed and button in (mouse.Button.left, mouse.Button.right):
            self.times_.append(time.time())
            self.coordinates_.append((x, y))
            self.button_.append(button.name)


    def on_press(self, key) -> bool:
        """Keyboard event handler to stop recording."""
        if key == self.stop_key:
            return False # Returning False stops the listener


    def record_and_save(self, stop_key=keyboard.Key.ctrl_l) -> None:
        """
        Start recording mouse events until the specified stop key is pressed.

        :param stop_key: The key to press to stop recording. Defaults to Left Control.
        """
        if not self.record:
            print("Not in recording mode. Instantiate with record=True to record.")
            return

        self.stop_key = stop_key
        print(f"Recording mouse clicks... Press '{self.stop_key}' to stop.")
        
        mouse_listener = mouse.Listener(on_click=self.on_click)
        keyboard_listener = keyboard.Listener(on_press=self.on_press)

        mouse_listener.start()
        keyboard_listener.start()

        keyboard_listener.join()
        mouse_listener.stop()
        
        self._save_to_csv()
        print(f"Recording stopped. Saved {len(self.times_)} events to {self.filename}")


    def _save_to_csv(self) -> None:
        """Save recorded data to a CSV file."""
        if not self.times_:
            print("No events were recorded. CSV file not created.")
            return
            
        header = ['timestamps', 'x_axis', 'y_axis', 'button']
        rows = zip(
            self.times_, 
            [pos[0] for pos in self.coordinates_], 
            [pos[1] for pos in self.coordinates_], 
            self.button_
        )
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


    def reproduce(
            self,
            iterations: int = 1,
            move_duration: float = 0.1,
            x_rand: int = 0,
            y_rand: int = 0,
            verbose: bool = False
        ) -> None:
            """
            Replay recorded mouse events with accurate timing.
            
            :param iterations: Number of times to repeat the sequence.
            :param move_duration: Time in seconds for the mouse to move between points.
            :param x_rand: Random pixel offset range in the X direction.
            :param y_rand: Random pixel offset range in the Y direction.
            :param verbose: If True, prints timing information during playback.
            """
            if not self.times_:
                print("No recording data found to reproduce.")
                return
                
            if iterations < 1:
                raise ValueError("Iterations must be 1 or greater.")
            if any(v < 0 for v in (move_duration, x_rand, y_rand)):
                raise ValueError("Duration and randomization values cannot be negative.")

            base_time = self.times_[0]
            relative_times = [t - base_time for t in self.times_]
            iteration_times = []

            print(f"Starting playback ({iterations} iteration(s))...")
            for q in range(iterations):
                iter_start = time.perf_counter()
                
                for i, (rel_time, (x, y), button) in enumerate(zip(
                    relative_times, self.coordinates_, self.button_
                )):
                    # Calculate when the click should happen
                    target_time = iter_start + rel_time

                    # 1. Move the mouse to the target position first
                    rand_x = x + randint(-x_rand, x_rand)
                    rand_y = y + randint(-y_rand, y_rand)
                    pyautogui.moveTo(rand_x, rand_y, duration=move_duration)

                    # 2. Now, wait until it's time to perform the click
                    current_time = time.perf_counter()
                    if current_time < target_time:
                        time.sleep(target_time - current_time)
                    
                    # 3. Finally, perform the click at the correct time
                    pyautogui.click(button=button)
                
                iter_end = time.perf_counter()
                iter_duration = iter_end - iter_start
                iteration_times.append(iter_duration)
                
                if verbose:
                    avg_time = sum(iteration_times) / len(iteration_times)
                    orig_duration = relative_times[-1] if relative_times else 0
                    print(
                        f"Iteration {q+1}/{iterations}: "
                        f"Actual={iter_duration:.2f}s, "
                        f"Avg={avg_time:.2f}s, "
                        f"Original={orig_duration:.2f}s"
                    )

            print(f"Playback completed. Average iteration time: {sum(iteration_times)/len(iteration_times):.2f}s")


def click(
        x: int, 
        y: int, 
        x_rand: int = 0, 
        y_rand: int = 0, 
        shift: bool = False, 
        duration: float = None
        ) -> None:
    '''
     Simple function to click on a specified (x, y) coordinate on the screen.
    
    :param x_: The x-coordinate of the screen where the click will be performed.
    :param y_: The y-coordinate of the screen where the click will be performed.
    :param x_rand: The random range to offset the x-coordinate for variability (default is 0, no offset).
    :param y_rand: The random range to offset the y-coordinate for variability (default is 0, no offset).
    :param shift: If True, the Shift key will be held down while clicking (default is False).
    :param duration: The duration it takes to move the cursor to the target coordinates (default is a random value between 0.3 and 0.5 seconds).
    
    :return: None
    '''
    screen_width, screen_height = pyautogui.size()
    if not (0 <= x <= screen_width and 0 <= y <= screen_height):
        raise ValueError(f"Base coordinates ({x}, {y}) are out of screen bounds.")
    
    if duration is None:
        duration = randint(30, 50)/100
    
    target_x = min(max(0, x + randint(-x_rand, x_rand)), screen_width - 1)
    target_y = min(max(0, y + randint(-y_rand, y_rand)), screen_height - 1)

    pyautogui.moveTo(
        target_x, 
        target_y, 
        duration=duration
        )
    if shift:
        with pyautogui.hold('shift'):
            pyautogui.click()
    else:
        pyautogui.click()


def get_mouse_coordinates(verbose=False):
    '''
    Returns the current mouse (x, y) coordinates after the user presses a Shift key.
    
    This function blocks execution until a Shift key is pressed.

    :param verbose: If True, prints instructions to the console (default is False).
    :return: A tuple containing the (x, y) coordinates, or (None, None) if interrupted.
    '''
    if verbose:
        print("Move your mouse to the desired location.")
        print("Press either SHIFT key to capture the coordinates.")

    coords = {'x': None, 'y': None}

    def on_press(key):
        if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            coords['x'], coords['y'] = pyautogui.position()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    return coords['x'], coords['y']


def wait_for_image(
        needle_image_path: str, 
        region: Optional[dict] = None, 
        appear: bool = True, 
        confidence: float = 0.8, 
        timeout: int = 10
        ) -> Union[Tuple[int, int], bool]:
    """
    Waits for a specified image to appear or disappear within a given screen region.

    :param needle_image_path: The file path of the image to search for.
    :param region: A tuple (left, top, width, height) defining the search area.
                   If None, searches the entire screen.
    :param appear: If True, waits for the image to appear. If False, waits for it to disappear.
                   Defaults to True.
    :param confidence: The confidence level for matching (0.0 to 1.0). Defaults to 0.8.
    :param timeout: Maximum time in seconds to wait. If 0, runs indefinitely. Defaults to 10.
    
    :return: 
        - Tuple (x, y): Center coordinates of the image if it appeared.
        - True: If the image successfully disappeared.
        - False: If the timeout was reached.
    """
    if region is not None and not isinstance(region, dict):
        raise TypeError("Region parameter must be a dictionary or None.")
                        
    search_region = None
    if region:
        try:
            search_region = (
                region['left'], 
                region['top'], 
                region['width'], 
                region['height']
            )
        except KeyError as e:
            raise KeyError(f"Region dictionary is missing a required key: {e}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            location = pyautogui.locateOnScreen(
                needle_image_path,
                region=search_region,
                confidence=confidence
            )

            if appear and location:
                return (
                    location.left + location.width / 2, # X axis center
                    location.top + location.height / 2 # Y axis center
                )

        except pyautogui.ImageNotFoundException:
            if not appear:
                return True
        
        time.sleep(0.2)
        
    print(f"Timeout: Waited {timeout} seconds but the condition was not met.")
    return False
