# GestureDraw

Draw on your webcam feed using hand gestures, move and scale what you draw, and snap wobbly sketches into perfect geometric shapes.

## Features

- Draw with your right index finger
- Move the drawing by pinching with your right hand
- Scale the drawing with two hands (pinch + distance)
- Perfect shapes: Circle, Square, Rectangle, Triangle, Oval, Pentagon, Hexagon, Octagon, Rhombus, Trapezoid
- Paint-style toolbar: colors, brush sizes, pen, highlighter, eraser, fill, undo/redo, save
- Open palm clears the canvas
- OK sign confirms / snaps shapes
- Smoothing + debouncing for stable detection

## Gesture Cheat Sheet

| Gesture | Hand | Action |
|---|---|---|
| Index only | Right | Draw |
| Thumb + index pinch | Right | Move drawing |
| Index only | Left | Pointer (hovers toolbar) |
| OK sign | Right or Left | Click hovered tool |
| OK sign (off toolbar) | Left | Snap last stroke to perfect shape |
| Both hands in pinch pose | Both | Two-hand scale |
| Open palm | Right | Clear canvas |
| Fist | Any | No action |

## Project structure

gesture_draw/
    main.py
    requirements.txt
    README.md
    config/
        settings.py
    src/
        core/
            hand_tracker.py
            gesture_recognizer.py
            smoother.py
        canvas/
            drawing_canvas.py
            object_transform.py
            shape_snap.py
            flood_fill.py
        ui/
            overlay.py
            toolbar.py
        utils/
            geometry.py
    assets/
        icons/
        demo/
    tests/

## Requirements

- Python 3.9 to 3.12
- Webcam

## Setup

git clone https://github.com/ShehryarRaza09/gesture_draw.git
cd gesture_draw

### Windows

The commands below use Python 3.12 and do not require activating the virtual
environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Run

If the virtual environment is activated, run `python main.py`. On Windows, you
can also run `.\.venv\Scripts\python.exe main.py` without activating it.

## Controls

| Key | Action |
|---|---|
| ESC | Quit |
| C | Clear canvas |
| Z | Undo |
| Y | Redo |
| S | Save PNG to saved/ |

## Tuning

Edit config/settings.py:
- FRAME_WIDTH, FRAME_HEIGHT - camera resolution (lower = faster)
- MIN_SCALE, MAX_SCALE - two-hand scale limits
- PALETTE - color list
- BRUSH_SIZES - brush thickness options

In main.py:
- TRACK_EVERY - run MediaPipe every N-th frame (higher = faster, more lag)

## License

MIT
