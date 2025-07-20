import tkinter as tk
from tkinter import ttk
import heapq
import time
import math
from typing import Dict, List, Tuple

Coord = Tuple[float, float]
Edge = Tuple[str, int]
Graph = Dict[str, List[Edge]]

class DijkstraVisualizer:
    NODE_RADIUS = 18

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dijkstra's Algorithm Visualizer")

        self.graph: Graph = {
            "A": [("B", 2), ("C", 5)], "B": [("A", 2), ("C", 6), ("D", 1)],
            "C": [("A", 5), ("B", 6), ("E", 3)], "D": [("B", 1), ("E", 1), ("F", 4)],
            "E": [("C", 3), ("D", 1), ("G", 2)], "F": [("D", 4), ("G", 1)],
            "G": [("E", 2), ("F", 1)],
        }
        self.start_node, self.end_node = "A", "G"
        self._arrange_node_positions()

        self._setup_ui()
        self.reset()

    def _arrange_node_positions(self):
        cx, cy, radius = 300, 210, 160
        self.coords: Dict[str, Coord] = {}
        nodes = list(self.graph)
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            self.coords[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    def _setup_ui(self):
        control_panel = ttk.Frame(self.root, padding=10)
        control_panel.grid(row=0, column=0, sticky="n")

        ttk.Label(control_panel, text="Dijkstra Visualizer", font=("Arial", 16, "bold")).grid(pady=(0, 8))
        self.algo_choice = tk.StringVar(value="heap")
        for i, (text, val) in enumerate([("Min-Heap", "heap"), ("Linear Scan", "lin")], 1):
            ttk.Radiobutton(control_panel, text=text, variable=self.algo_choice, value=val).grid(row=i, sticky="w")

        ttk.Button(control_panel, text="Start", command=self.start).grid(pady=(10, 2), sticky="we")
        self.step_button = ttk.Button(control_panel, text="Step", state="disabled", command=self.step)
        self.step_button.grid(pady=2, sticky="we")
        ttk.Button(control_panel, text="Reset", command=self.reset).grid(pady=2, sticky="we")

        self.status_label = ttk.Label(control_panel, wraplength=220)
        self.status_label.grid(pady=(12, 2))
        self.timing_label = ttk.Label(control_panel)
        self.timing_label.grid()

        self.canvas = tk.Canvas(self.root, width=620, height=420, bg="white")
        self.canvas.grid(row=0, column=1, padx=10, pady=10)
        self.queue_canvas = tk.Canvas(self.root, width=200, height=420, bg="#f7f7f7")
        self.queue_canvas.grid(row=0, column=2, padx=10, pady=10)

    def reset(self):
        self.dist = {n: float("inf") for n in self.graph}
        self.prev = {n: None for n in self.graph}
        self.dist[self.start_node] = 0.0
        self.visited = set()
        self.unvisited = set(self.graph)
        self.pq = []
        self.current_node = None
        self.is_running = False
        self.is_finished = False
        self.start_time = 0.0
        self.status_label["text"] = "Select an algorithm and press Start"
        self.timing_label["text"] = ""
        self.step_button["state"] = "disabled"
        self._redraw()

    def start(self):
        if self.is_running:
            return
        if self.algo_choice.get() == "heap":
            heapq.heappush(self.pq, (0.0, self.start_node))
        self.is_running = True
        self.start_time = time.perf_counter()
        self.step_button["state"] = "normal"
        self.status_label["text"] = "Running... Press Step to continue."
        self._redraw()

    def step(self):
        if not self.is_running or self.is_finished:
            return

        if self.algo_choice.get() == "heap":
            while self.pq and self.pq[0][1] in self.visited:
                heapq.heappop(self.pq)
            if not self.pq:
                return self._finish()
            dist_u, u = heapq.heappop(self.pq)
        else:
            u = min(self.unvisited, key=self.dist.get, default=None)
            if u is None or self.dist[u] == float("inf"):
                return self._finish()
            dist_u = self.dist[u]
            self.unvisited.remove(u)

        self.current_node = u
        self.visited.add(u)

        if u == self.end_node:
            return self._finish()

        for neighbor, weight in self.graph[u]:
            if neighbor in self.visited:
                continue
            new_dist = dist_u + weight
            if new_dist < self.dist[neighbor]:
                self.dist[neighbor] = new_dist
                self.prev[neighbor] = u
                if self.algo_choice.get() == "heap":
                    heapq.heappush(self.pq, (new_dist, neighbor))

        self._redraw()

    def _finish(self):
        self.is_finished = True
        self.is_running = False
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.timing_label["text"] = f"{elapsed_ms:.1f} ms"
        path_exists = self.dist[self.end_node] < float("inf")
        self.status_label["text"] = "Path found!" if path_exists else "Target is unreachable."
        self.step_button["state"] = "disabled"
        self._redraw(show_path=True)

    def _get_node_color(self, node_id: str) -> str:
        if node_id == self.current_node:
            return "khaki"
        if node_id in self.visited:
            return "lightgray"
        if node_id == self.start_node:
            return "lightgreen"
        if node_id == self.end_node:
            return "tomato"
        return "skyblue"

    def _redraw(self, *, show_path: bool = False):
        self.canvas.delete("all")
        for u, edges in self.graph.items():
            x1, y1 = self.coords[u]
            for v, w in edges:
                if u < v:
                    x2, y2 = self.coords[v]
                    self.canvas.create_line(x1, y1, x2, y2, fill="#999", width=2)
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2 - 8, text=w, font=("Arial", 9, "italic"))

        for n, (x, y) in self.coords.items():
            fill_color = self._get_node_color(n)
            self.canvas.create_oval(x - self.NODE_RADIUS, y - self.NODE_RADIUS,
                                    x + self.NODE_RADIUS, y + self.NODE_RADIUS,
                                    fill=fill_color, outline="black", width=2)
            self.canvas.create_text(x, y, text=n, font=("Arial", 12, "bold"))
            dist_val = self.dist[n]
            dist_text = "∞" if dist_val == float("inf") else str(int(dist_val))
            self.canvas.create_text(x, y + self.NODE_RADIUS + 12, text=dist_text, font=("Arial", 9))

        if show_path and self.dist[self.end_node] < float("inf"):
            path = []
            curr = self.end_node
            while curr:
                path.append(curr)
                curr = self.prev[curr]
            path.reverse()
            for i in range(len(path) - 1):
                x1, y1 = self.coords[path[i]]
                x2, y2 = self.coords[path[i + 1]]
                self.canvas.create_line(x1, y1, x2, y2, fill="orange", width=4)
            for node in path:
                x, y = self.coords[node]
                self.canvas.create_oval(x - self.NODE_RADIUS, y - self.NODE_RADIUS,
                                        x + self.NODE_RADIUS, y + self.NODE_RADIUS,
                                        fill="orange", outline="black", width=2)

        self.queue_canvas.delete("all")
        is_heap = self.algo_choice.get() == "heap"
        title = "Priority Queue" if is_heap else "Unvisited Nodes"
        self.queue_canvas.create_text(100, 14, text=title, font=("Arial", 11, "bold"))
        if is_heap:
            items = [f"{int(d)}: {v}" for d, v in sorted(self.pq)]
        else:
            items = sorted(self.unvisited, key=self.dist.get)
        for i, item in enumerate(items):
            self.queue_canvas.create_text(100, 36 + i * 18, text=str(item), font=("Arial", 9))

if __name__ == "__main__":
    root = tk.Tk()
    app = DijkstraVisualizer(root)
    root.mainloop()
