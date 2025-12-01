import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit


app = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")


@dataclass
class SimulationState:
    xstart: float
    xend: float
    deltax: float
    time_end: float
    emit_interval: float
    dt: float = 0.05
    running: bool = False
    paused: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread = None
    t: float = 0.0
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)

    def reset_grid(self) -> None:
        self.x_values = [self.xstart + i * self.deltax for i in range(int((self.xend - self.xstart) / self.deltax) + 1)]
        self.y_values = [math.sin(x) for x in self.x_values]
        self.t = 0.0


state: SimulationState = None
state_lock = threading.Lock()


def compute_step(y: List[float], dt: float, t: float, x_values: List[float]) -> List[float]:
    # Simple iterative update: dy/dt = -0.4 * y + sin(x + t)
    return [value + dt * (-0.4 * value + math.sin(x + t)) for value, x in zip(y, x_values)]


def simulation_loop(sim_state: SimulationState) -> None:
    last_emit = 0.0
    sim_state.running = True
    sim_state.stop_event.clear()
    while sim_state.t <= sim_state.time_end and not sim_state.stop_event.is_set():
        if sim_state.paused:
            time.sleep(0.05)
            continue

        sim_state.y_values = compute_step(sim_state.y_values, sim_state.dt, sim_state.t, sim_state.x_values)
        sim_state.t += sim_state.dt

        if sim_state.t - last_emit >= sim_state.emit_interval - 1e-9:
            emit_slice(sim_state)
            last_emit = sim_state.t

        time.sleep(sim_state.dt)

    if not sim_state.stop_event.is_set():
        emit_slice(sim_state)
        socketio.emit("simulation_complete", {"t": sim_state.t})
    sim_state.running = False


def emit_slice(sim_state: SimulationState) -> None:
    payload = {
        "t": round(sim_state.t, 3),
        "x": sim_state.x_values,
        "y": sim_state.y_values,
    }
    socketio.emit("slice", payload)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory(app.static_folder, path)


@socketio.on("start")
def start_simulation(data: Dict):
    global state
    with state_lock:
        if state and state.running:
            state.stop_event.set()
            state.thread.join()

        xstart = float(data.get("xstart", 0))
        xend = float(data.get("xend", 10))
        deltax = max(float(data.get("deltax", 0.1)), 0.01)
        time_end = float(data.get("timeEnd", 10))
        emit_interval = max(float(data.get("emit", 0.5)), 0.01)
        dt = max(float(data.get("dt", 0.05)), 0.01)

        state = SimulationState(xstart=xstart, xend=xend, deltax=deltax, time_end=time_end, emit_interval=emit_interval, dt=dt)
        state.reset_grid()

        thread = threading.Thread(target=simulation_loop, args=(state,), daemon=True)
        state.thread = thread
        thread.start()
    emit("started", {"message": "Simulation started."})


@socketio.on("pause")
def pause_simulation():
    with state_lock:
        if state:
            state.paused = not state.paused
            emit("paused", {"paused": state.paused})


@socketio.on("stop")
def stop_simulation():
    global state
    with state_lock:
        if state:
            state.stop_event.set()
            if state.thread and state.thread.is_alive():
                state.thread.join()
            state = None
    emit("stopped", {"message": "Simulation stopped."})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
