# Counter/Simulator

A small Flask + Socket.IO client/server demo that streams iterative simulation slices to a browser-based chart. The client allows starting, pausing, stopping, and time-scrubbing through saved slices.

## Running the app

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the server:

```bash
python server.py
```

3. Open the client at http://localhost:5000.

## Features

- Iterative numerical update for `y(x, t)` using a simple ODE-like rule.
- Periodic slice emission back to the browser over WebSockets.
- Start/Pause/Stop controls with resume from pause.
- Time slider to inspect any previously emitted slice.
- Chart.js visualization of the evolving function.
