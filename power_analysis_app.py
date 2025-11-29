import os
import tempfile
import io
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class PowerAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Power Consumption Analyzer")

        # Data containers
        self.time = np.array([])
        self.power = np.array([])
        self.filtered_power = np.array([])
        self.selected_range = None
        self.events = []
        self.last_threshold = None
        self.last_threshold2 = None

        # Loaded file name
        self.loaded_file_name = None

        # Chart objects
        self.preview_line = None
        self.main_line = None
        self.event_detail_line = None
        self.span_selector = None
        self.event_patches = []
        self.event_lines = []
        self.event_labels = []
        self.event_palette = [
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
        ]

        self._build_layout()
        self._load_default_file()

    def _build_layout(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)

        chart_frame = ttk.Frame(self.root)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        controls_frame = ttk.Frame(self.root)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        controls_frame.columnconfigure(0, weight=1)

        self._build_chart(chart_frame)
        self._build_controls(controls_frame)

    def _build_chart(self, parent):
        fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.1)
        self.ax_preview = fig.add_subplot(grid[0])
        self.ax_main = fig.add_subplot(grid[1])

        self.ax_preview.set_ylabel("Power (mW)")
        self.ax_main.set_ylabel("Power (mW)")
        self.ax_main.set_xlabel("Time (s)")
        self.ax_preview.set_title("Filtered overview (drag to select)")

        self.preview_line, = self.ax_preview.plot([], [], color="#5b8def", lw=1)
        self.main_line, = self.ax_main.plot([], [], color="#1b4f72", lw=1)
        self.ax_main.grid(True, linestyle="--", alpha=0.3)
        self.ax_preview.grid(True, linestyle="--", alpha=0.3)

        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        stats_frame = ttk.Frame(parent)
        stats_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        stats_frame.columnconfigure(0, weight=1)
        self.stats_var = tk.StringVar(value="Load a CSV file to begin.")
        ttk.Label(stats_frame, textvariable=self.stats_var).grid(
            row=0, column=0, sticky="w"
        )

    def _build_controls(self, parent):
        load_frame = ttk.LabelFrame(parent, text="Data")
        load_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        load_frame.columnconfigure(1, weight=1)

        ttk.Button(load_frame, text="Load CSV", command=self._prompt_load).grid(
            row=0, column=0, padx=(4, 4), pady=4, sticky="w"
        )
        self.file_name_var = tk.StringVar(value="")
        ttk.Label(load_frame, textvariable=self.file_name_var).grid(
            row=1, column=0, columnspan=6, padx=(4, 4), pady=(0, 4), sticky="w"
        )
        ttk.Label(load_frame, text="Avg window (samples)").grid(
            row=0, column=1, padx=4, pady=4, sticky="e"
        )
        self.filter_entry = ttk.Entry(load_frame, width=8)
        self.filter_entry.insert(0, "25")
        self.filter_entry.grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Button(load_frame, text="Apply Filter", command=self._apply_filter).grid(
            row=0, column=3, padx=4, pady=4, sticky="e"
        )
        ttk.Label(load_frame, text="Max events").grid(
            row=0, column=4, padx=4, pady=4, sticky="e"
        )
        self.max_events_entry = ttk.Entry(load_frame, width=6)
        self.max_events_entry.insert(0, "20")
        self.max_events_entry.grid(row=0, column=5, padx=4, pady=4, sticky="w")

        detect_frame = ttk.LabelFrame(parent, text="Event detection")
        detect_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            detect_frame.columnconfigure(i, weight=1)

        mode_frame = ttk.Frame(detect_frame)
        mode_frame.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=2)
        ttk.Label(mode_frame, text="Mode").pack(side="left")
        self.detect_mode = tk.StringVar(value="threshold")
        for text, key in [
                ("Threshold", "threshold"),
                ("Rising", "rising"),
                ("Falling", "falling"),
            ]:
            ttk.Radiobutton(mode_frame, text=text, value=key, variable=self.detect_mode).pack(
                side="left", padx=4
            )

        self.threshold_entry = self._labeled_entry(
            detect_frame, "Threshold", "10", 1, 0
        )
        self.threshold2_entry = self._labeled_entry(
            detect_frame, "Threshold2 (mW)", "20", 1, 1
        )

        # Add trace to toggle threshold2_entry state based on detection mode
        self.detect_mode.trace_add("write", self._on_detect_mode_change)
        # Set initial state (threshold mode disables threshold2)
        self._on_detect_mode_change()

        self.min_len_entry = self._labeled_entry(
            detect_frame, "Min length (s)", "0", 1, 2
        )
        self.max_len_entry = self._labeled_entry(
            detect_frame, "Max length (s)", "100", 1, 3
        )

        btn_frame = ttk.Frame(detect_frame)
        btn_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        ttk.Button(btn_frame, text="Detect", command=self._detect_events).grid(
            row=0, column=0, padx=4, pady=2, sticky="ew"
        )
        ttk.Button(btn_frame, text="Clear", command=self._clear_events).grid(
            row=0, column=1, padx=4, pady=2, sticky="ew"
        )

        events_frame = ttk.LabelFrame(parent, text="Events")
        events_frame.grid(row=2, column=0, sticky="nsew")
        events_frame.rowconfigure(1, weight=1)
        events_frame.rowconfigure(2, weight=0)
        events_frame.columnconfigure(0, weight=1)

        # Event detail plot (summary)
        fig = Figure(figsize=(5, 2.4), dpi=100)
        self.ax_event = fig.add_subplot(111)
        self.ax_event.set_title("Events summary (avg power vs length)")
        self.ax_event.set_xlabel("Length (s)")
        self.ax_event.set_ylabel("Avg Power (mW)")
        self.ax_event.grid(True, linestyle="--", alpha=0.3)
        self.event_canvas = FigureCanvasTkAgg(fig, master=events_frame)
        self.event_canvas.draw()
        self.event_canvas.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", padx=4, pady=4
        )

        # Average stats label for all events (above the buttons)
        self.event_avg_var = tk.StringVar(value="")
        self.event_avg_label = ttk.Label(events_frame, textvariable=self.event_avg_var)
        self.event_avg_label.grid(row=2, column=0, sticky="ew", padx=4, pady=(4, 2))

        # Event buttons placed below the summary chart
        self.event_list_container = ttk.Frame(events_frame)
        self.event_list_container.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.event_buttons = []

    def _labeled_entry(self, parent, label, default, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        ttk.Label(frame, text=label).pack(anchor="w")
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default)
        entry.pack(fill="x")
        return entry

    def _on_detect_mode_change(self, *args):
        """Toggle threshold2_entry state based on detection mode.
        
        Disable threshold2 when mode is 'threshold', enable for 'rising'/'falling'.
        Additionally, for 'falling' mode, swap threshold values if threshold2 > threshold
        to ensure the secondary threshold is below the primary.
        """
        mode = self.detect_mode.get()
        if mode == "threshold":
            self.threshold2_entry.configure(state="disabled")
        else:
            self.threshold2_entry.configure(state="normal")
            # For falling mode, swap thresholds if threshold2 > threshold
            if mode == "falling":
                try:
                    threshold = float(self.threshold_entry.get())
                    threshold2 = float(self.threshold2_entry.get())
                    if threshold2 > threshold:
                        # Swap values so threshold2 is below threshold
                        self.threshold_entry.delete(0, tk.END)
                        self.threshold_entry.insert(0, f"{threshold2:g}")
                        self.threshold2_entry.delete(0, tk.END)
                        self.threshold2_entry.insert(0, f"{threshold:g}")
                except ValueError:
                    # If values are not valid numbers, skip the swap
                    pass

    def _load_default_file(self):
        default_path = os.path.join(os.getcwd(), "detection_mode_sample.csv")
        if os.path.exists(default_path):
            self._load_csv(default_path)

    def _prompt_load(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._load_csv(path)

    def _load_csv(self, path):
        try:
            headers, data = self._read_csv_numeric(path)
            num_cols = data.shape[1]
            if num_cols < 2:
                raise ValueError("CSV must have at least two numeric columns.")

            time_idx, power_idx = 0, 1
            if num_cols > 2:
                selection = self._prompt_column_selection(headers)
                if selection is None:
                    messagebox.showinfo("Cancelled", "Load cancelled; no columns selected.")
                    return
                time_idx, power_idx = selection

            time = data[:, time_idx]
            power = data[:, power_idx]
            mask = np.isfinite(time) & np.isfinite(power)
            if not np.any(mask):
                raise ValueError("No finite samples found in CSV.")
            self.time = time[mask]
            self.power = power[mask]
            self.loaded_file_name = os.path.basename(path)
            self.file_name_var.set(self.loaded_file_name)
            self._apply_filter()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV: {exc}")

    def _read_csv_numeric(self, path):
        """
        Load CSV data as floats. If the first row is non-numeric, treat it as a header.
        Returns (headers, data_array).
        """
        with open(path, newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, [])
        if not first_row:
            raise ValueError("CSV is empty.")

        num_cols = len(first_row)

        def _row_is_numeric(row):
            try:
                [float(x) for x in row]
                return True
            except Exception:
                return False

        first_row_is_data = _row_is_numeric(first_row)
        headers = (
            [f"Col {i+1}" for i in range(num_cols)]
            if first_row_is_data
            else [h.strip() or f"Col {i+1}" for i, h in enumerate(first_row)]
        )

        skiprows = 0 if first_row_is_data else 1
        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
        data = np.asarray(data, dtype=float)
        try:
            data = data.reshape(-1, num_cols)
        except Exception:
            raise ValueError("CSV rows have inconsistent column counts.")
        return headers, data

    def _prompt_column_selection(self, headers):
        """
        Ask the user to pick time and power columns when a CSV has >2 columns.
        Returns (time_idx, power_idx) or None if cancelled.
        """
        top = tk.Toplevel(self.root)
        top.title("Select columns")
        top.transient(self.root)
        top.grab_set()

        options = [f"{i+1}: {name}" for i, name in enumerate(headers)]
        time_var = tk.StringVar(value=options[0])
        power_var = tk.StringVar(value=options[1] if len(options) > 1 else options[0])
        result = {"value": None}

        ttk.Label(top, text="Time column:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.OptionMenu(top, time_var, time_var.get(), *options).grid(
            row=0, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Label(top, text="Power column:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        ttk.OptionMenu(top, power_var, power_var.get(), *options).grid(
            row=1, column=1, padx=6, pady=6, sticky="ew"
        )

        def parse_idx(val):
            try:
                return int(val.split(":")[0]) - 1
            except Exception:
                return 0

        def on_ok():
            t_idx = parse_idx(time_var.get())
            p_idx = parse_idx(power_var.get())
            if t_idx == p_idx:
                messagebox.showerror("Error", "Time and power columns must differ.")
                return
            result["value"] = (t_idx, p_idx)
            top.destroy()

        def on_cancel():
            result["value"] = None
            top.destroy()

        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=6)

        top.columnconfigure(1, weight=1)
        self.root.wait_window(top)
        return result["value"]

    def _apply_filter(self):
        if self.power.size == 0:
            messagebox.showerror("Error", "No data loaded to filter.")
            return
        if self.time.size != self.power.size:
            messagebox.showerror("Error", "Time and power arrays differ in length.")
            return
        try:
            window = max(1, int(float(self.filter_entry.get())))
        except ValueError:
            messagebox.showerror("Error", "Average window must be a number.")
            return

        try:
            time_arr = np.asarray(self.time, dtype=float)
            power_arr = np.asarray(self.power, dtype=float)
            mask = np.isfinite(time_arr) & np.isfinite(power_arr)
            if not np.any(mask):
                raise ValueError("No finite samples to filter.")
            time_arr = time_arr[mask]
            power_arr = power_arr[mask]
            clean_power = power_arr
            kernel = np.ones(window, dtype=float) / float(window)
            filtered = np.convolve(clean_power, kernel, mode="same")
            if not np.all(np.isfinite(filtered)):
                raise ValueError("Filtered data contains non-finite values.")
            self.time = time_arr
            self.power = power_arr
            self.filtered_power = filtered
            self._refresh_plots(reset_selection=True)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to apply filter: {exc}")

    def _refresh_plots(self, reset_selection=False):
        if self.time.size == 0:
            return
        self.preview_line.set_data(self.time, self.filtered_power)
        self.main_line.set_data(self.time, self.filtered_power)
        self.ax_preview.relim()
        self.ax_preview.autoscale_view()
        # Ensure preview always spans the full data range on X.
        self.ax_preview.set_xlim(float(self.time[0]), float(self.time[-1]))
        self.ax_main.relim()
        self.ax_main.autoscale_view()

        if reset_selection or self.selected_range is None:
            self.selected_range = (float(self.time[0]), float(self.time[-1]))
        self.ax_main.set_xlim(*self.selected_range)
        self._autoscale_main_y()

        if self.span_selector:
            # Clear any existing span visuals before wiring a fresh selector.
            try:
                self.span_selector.clear()
            except Exception:
                pass
            self.span_selector.disconnect_events()
            self.span_selector = None
        self._clear_preview_selection()
        self.span_selector = SpanSelector(
            self.ax_preview,
            self._on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.2, facecolor="#82c0ff"),
            interactive=True,
        )

        self._update_stats()
        self.canvas.draw_idle()

    def _on_select(self, xmin, xmax):
        if self.time.size == 0:
            return
        if xmin == xmax:
            return
        lo, hi = (xmin, xmax) if xmin < xmax else (xmax, xmin)
        self.selected_range = (lo, hi)
        self.ax_main.set_xlim(lo, hi)
        self._autoscale_main_y()
        self._update_stats()
        self.ax_main.figure.canvas.draw_idle()

    def _on_canvas_click(self, event):
        # Right-click on the overview resets selection to full range.
        if event.button != 3:  # 3 = right click in mpl
            return
        if event.inaxes is not self.ax_preview or self.time.size == 0:
            return
        self.selected_range = (float(self.time[0]), float(self.time[-1]))
        self.ax_main.set_xlim(*self.selected_range)
        self._clear_preview_selection()
        self._autoscale_main_y()
        self._update_stats()
        self.canvas.draw_idle()

    def _clear_preview_selection(self):
        if self.span_selector:
            try:
                self.span_selector.clear()
            except Exception:
                pass
        # Remove any lingering selection shading from the overview plot.
        for patch in list(self.ax_preview.patches):
            try:
                patch.remove()
            except Exception:
                pass

    def _autoscale_main_y(self):
        if self.time.size == 0:
            return
        lo, hi = self.selected_range if self.selected_range else (self.time[0], self.time[-1])
        mask = (self.time >= lo) & (self.time <= hi)
        if not np.any(mask):
            return
        p = self.filtered_power[mask]
        p_min = float(np.min(p))
        p_max = float(np.max(p))
        if p_min == p_max:
            margin = max(1.0, abs(p_min) * 0.05)
        else:
            margin = (p_max - p_min) * 0.05
        self.ax_main.set_ylim(p_min - margin, p_max + margin)

    def _update_stats(self):
        if self.time.size == 0 or self.selected_range is None:
            self.stats_var.set("No data loaded.")
            return
        lo, hi = self.selected_range
        mask = (self.time >= lo) & (self.time <= hi)
        if not np.any(mask):
            self.stats_var.set("No samples in selection.")
            return
        t = self.time[mask]
        p = self.filtered_power[mask]
        duration = t[-1] - t[0]
        energy_j = float(np.trapz(p / 1000.0, t))
        stats_text = (
            f"Samples: {p.size} | Mean: {np.mean(p):.2f} mW | "
            f"Max: {np.max(p):.2f} mW | Min: {np.min(p):.2f} mW | "
            f"Duration: {duration:.6f} s | Energy: {energy_j * 1000:.4f} mJ"
        )
        self.stats_var.set(stats_text)

    def _detect_events(self):
        if self.time.size == 0:
            messagebox.showinfo("Info", "Load data before detecting events.")
            return
        try:
            threshold = float(self.threshold_entry.get())
            threshold2 = float(self.threshold2_entry.get())
            min_len = float(self.min_len_entry.get())
            max_len = float(self.max_len_entry.get())
            max_events = max(1, int(self.max_events_entry.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter numeric detection parameters.")
            return

        mode = self.detect_mode.get()
        if mode == "falling" and threshold2 > threshold:
            # In falling mode ensure secondary threshold is below the primary; swap if needed.
            threshold, threshold2 = threshold2, threshold
            for entry, val in (
                (self.threshold_entry, threshold),
                (self.threshold2_entry, threshold2),
            ):
                entry.delete(0, tk.END)
                entry.insert(0, f"{val:g}")

        self.last_threshold = threshold
        self.last_threshold2 = threshold2
        min_len = max(0.0, min_len)
        max_len = float("inf") if max_len <= 0 else max_len
        max_events = max(1, max_events)

        events = []
        in_event = False
        start_idx = 0
        peak = -np.inf
        peak_idx = 0
        data = self.filtered_power
        trough = np.inf
        trough_idx = 0
        reached_target = False
        # Secondary threshold is used as the target for rising (>=) and falling (<=) modes.
        target_up = threshold2
        target_down = threshold2

        for i in range(1, len(data)):
            val = data[i]
            prev = data[i - 1]
            if not in_event:
                if mode == "threshold" and prev < threshold <= val:
                    in_event = True
                    start_idx = i - 1
                    peak = val
                    peak_idx = i
                elif mode == "rising" and prev < threshold <= val:
                    in_event = True
                    start_idx = i - 1
                    peak = val
                    peak_idx = i
                    reached_target = False
                elif mode == "falling" and prev > threshold >= val:
                    in_event = True
                    start_idx = i - 1
                    trough = val
                    trough_idx = i
            else:
                if mode in ("threshold", "rising"):
                    if val > peak:
                        peak = val
                        peak_idx = i
                    if mode == "threshold":
                        if val < threshold:
                            end_idx = i
                            duration = self.time[end_idx] - self.time[start_idx]
                            required_peak = threshold
                            valid_peak = peak >= required_peak
                            valid_tail = True
                            length_ok = duration >= min_len and duration <= max_len
                            if valid_peak and valid_tail and length_ok:
                                ev = self._build_event(start_idx, end_idx, peak, peak_idx)
                                if ev:
                                    events.append(ev)
                            in_event = False
                    elif mode == "rising":
                        if val >= target_up:
                            end_idx = i
                            duration = self.time[end_idx] - self.time[start_idx]
                            reached_target = True
                            length_ok = duration >= min_len and duration <= max_len
                            if peak >= target_up and length_ok:
                                ev = self._build_event(start_idx, end_idx, peak, peak_idx)
                                if ev:
                                    events.append(ev)
                            in_event = False
                        elif val < threshold and not reached_target:
                            in_event = False  # aborted rise
                        else:
                            duration_now = self.time[i] - self.time[start_idx]
                            if duration_now > max_len:
                                in_event = False  # timeout without reaching target
                elif mode == "falling":
                    if val < trough:
                        trough = val
                        trough_idx = i
                    duration_now = self.time[i] - self.time[start_idx]
                    if duration_now > max_len:
                        in_event = False  # timeout
                        continue
                    if val >= threshold and trough > target_down:
                        # Recovered before enough drop; abort.
                        in_event = False
                        continue
                    if val <= target_down:
                        end_idx = i
                        duration = self.time[end_idx] - self.time[start_idx]
                        length_ok = duration >= min_len and duration <= max_len
                        if length_ok:
                            ev = self._build_event(
                                start_idx, end_idx, peak=trough, peak_idx=trough_idx
                            )
                            if ev:
                                events.append(ev)
                        in_event = False

            if len(events) >= max_events:
                break

        # Events that never return past the threshold are ignored to enforce a clear rise-and-fall.

        self.events = events
        self._render_events()

    def _build_event(self, start_idx, end_idx, peak, peak_idx):
        t_slice = self.time[start_idx : end_idx + 1]
        p_slice = self.filtered_power[start_idx : end_idx + 1]
        # Sanitize power slice to avoid NaNs affecting summaries/plots.
        mask = np.isfinite(p_slice)
        if not np.any(mask):
            # Skip invalid event entirely.
            return None
        clean_power = np.where(mask, p_slice, np.nan)
        fill_value = float(np.nanmean(clean_power))
        clean_power = np.nan_to_num(clean_power, nan=fill_value)
        avg_power = float(np.mean(clean_power))
        max_power = float(np.max(clean_power))
        min_power = float(np.min(clean_power))
        energy_j = float(np.trapz(clean_power / 1000.0, t_slice))
        return {
            "start": self.time[start_idx],
            "end": self.time[end_idx],
            "duration": self.time[end_idx] - self.time[start_idx],
            "energy_mj": energy_j * 1000.0,
            "avg_power": avg_power,
            "max_power": max_power,
            "min_power": min_power,
            "indices": (start_idx, end_idx),
        }

    def _format_event_stats(self, event):
        return (
            f"Start (s)\t{event['start']:.6f}\n"
            f"End (s)\t{event['end']:.6f}\n"
            f"Length (s)\t{event['duration']:.3f}\n"
            f"Avg power (mW)\t{event['avg_power']:.3f}\n"
            f"Max power (mW)\t{event['max_power']:.3f}\n"
            f"Min power (mW)\t{event['min_power']:.3f}\n"
            f"Energy (mJ)\t{event['energy_mj']:.3f}"
        )

    def _copy_fig_to_clipboard(self, fig):
        """
        Best-effort copy of a matplotlib figure as an image to the Windows clipboard.
        Falls back to False if dependencies (Pillow) are missing.
        """
        try:
            from PIL import Image
        except Exception:
            return False

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        try:
            im = Image.open(buf).convert("RGB")
        except Exception:
            return False

        # Convert to DIB (device independent bitmap) for CF_DIB clipboard format
        dib_buf = io.BytesIO()
        im.save(dib_buf, format="BMP")
        bmp_data = dib_buf.getvalue()
        if len(bmp_data) < 15:
            return False
        dib_data = bmp_data[14:]  # strip BMP header; keep DIB header+pixels

        CF_DIB = 8
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            h_global = kernel32.GlobalAlloc(0x2000, len(dib_data))
            if not h_global:
                return False
            locked_mem = kernel32.GlobalLock(h_global)
            if not locked_mem:
                kernel32.GlobalFree(h_global)
                return False
            ctypes.memmove(locked_mem, dib_data, len(dib_data))
            kernel32.GlobalUnlock(h_global)
            if not user32.SetClipboardData(CF_DIB, h_global):
                kernel32.GlobalFree(h_global)
                return False
            # Ownership of h_global is transferred to the system; no free.
            return True
        finally:
            user32.CloseClipboard()

    def _clear_events(self):
        self.events = []
        for patch in self.event_patches:
            patch.remove()
        self.event_patches = []
        for ln in self.event_lines:
            ln.remove()
        self.event_lines = []
        for lbl in self.event_labels:
            lbl.remove()
        self.event_labels = []
        self.ax_main.figure.canvas.draw_idle()
        for btn in self.event_buttons:
            btn.destroy()
        self.event_buttons = []
        self.event_avg_var.set("")
        self.ax_event.cla()
        self.ax_event.set_title("Events summary (avg power vs length)")
        self.ax_event.set_xlabel("Length (s)")
        self.ax_event.set_ylabel("Avg Power (mW)")
        self.ax_event.grid(True, linestyle="--", alpha=0.3)
        self.event_canvas.draw_idle()

    def _render_events(self):
        for patch in self.event_patches:
            patch.remove()
        self.event_patches = []
        for ln in self.event_lines:
            ln.remove()
        self.event_lines = []
        for lbl in self.event_labels:
            lbl.remove()
        self.event_labels = []

        for btn in self.event_buttons:
            btn.destroy()
        self.event_buttons = []

        if not self.events:
            self.event_avg_var.set("")
            self.ax_main.figure.canvas.draw_idle()
            return

        for idx, ev in enumerate(self.events):
            color = self.event_palette[idx % len(self.event_palette)]
            patch = self.ax_main.axvspan(
                ev["start"], ev["end"], color=color, alpha=0.25
            )
            self.event_patches.append(patch)

            # Visible start/end markers and label
            line_start = self.ax_main.axvline(ev["start"], color=color, lw=1.2, alpha=0.9)
            line_end = self.ax_main.axvline(ev["end"], color=color, lw=1.2, alpha=0.9, linestyle="--")
            self.event_lines.extend([line_start, line_end])
            y_top = self.ax_main.get_ylim()[1]
            label = self.ax_main.text(
                (ev["start"] + ev["end"]) / 2.0,
                y_top,
                f"E{idx + 1}",
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                alpha=0.8,
            )
            self.event_labels.append(label)

            btn_text = (
                f"E{idx + 1} | "
                f"Power: {ev['avg_power']:.2f}mW | "
                f"Length: {ev['duration']:.6f}s | "
                f"Energy: {ev['energy_mj']:.3f}mJ"
            )
            btn = ttk.Button(
                self.event_list_container,
                text=btn_text,
                command=lambda e=ev, c=color: self._show_event(e, c),
            )
            # Set style per color
            style_name = f"Event{idx}.TButton"
            style = ttk.Style()
            style.configure(style_name, foreground="black", background=color)
            style.map(style_name, background=[("active", color)])
            btn.configure(style=style_name)

            btn.grid(row=idx // 1, column=0, padx=2, pady=2, sticky="ew")
            self.event_buttons.append(btn)

        # Calculate and display average stats for all events
        avg_power = np.mean([ev['avg_power'] for ev in self.events])
        avg_duration = np.mean([ev['duration'] for ev in self.events])
        avg_energy = np.mean([ev['energy_mj'] for ev in self.events])
        self.event_avg_var.set(
            f"Avg: Power: {avg_power:.2f}mW | Length: {avg_duration:.6f}s | Energy: {avg_energy:.3f}mJ"
        )

        self.ax_main.figure.canvas.draw_idle()
        self._plot_event_summary()

    def _show_event(self, event, color="#d35400"):
        self._plot_event_summary(selected_event=event)
        self._open_zoom_popup(event, color=color)

    def _plot_event_summary(self, selected_event=None):
        self.ax_event.cla()
        self.ax_event.set_title("Events summary (avg power vs event time)")
        self.ax_event.set_xlabel("Length (s)")
        self.ax_event.set_ylabel("Avg Power (mW)")
        self.ax_event.grid(True, linestyle="--", alpha=0.1)

        points = []
        for idx, ev in enumerate(self.events):
            length_s = ev["duration"]
            avg_p = ev["avg_power"]
            if not np.isfinite(length_s) or not np.isfinite(avg_p):
                continue
            points.append((idx, length_s, avg_p))

        if not points:
            self.ax_event.text(
                0.5,
                0.5,
                "No valid events",
                transform=self.ax_event.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666",
            )
            self.event_canvas.draw_idle()
            return

        xs = []
        ys = []
        for idx, length_s, avg_p in points:
            ev = self.events[idx]
            color = self.event_palette[idx % len(self.event_palette)]
            size = 70 if selected_event is ev else 45
            alpha = 0.95 if selected_event is ev else 0.7
            self.ax_event.scatter(
                length_s,
                avg_p,
                color=color,
                s=size,
                alpha=alpha,
                edgecolors="k",
                linewidths=0.5 if selected_event is ev else 0.3,
            )
            label = f"E{idx + 1}"
            self.ax_event.text(
                length_s,
                avg_p,
                label,
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                alpha=alpha,
            )
            xs.append(length_s)
            ys.append(avg_p)

        # Manual autoscale to ensure even single-point events are visible.
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_min == x_max:
            pad_x = max(0.1, abs(x_min) * 0.1)
            x_min -= pad_x
            x_max += pad_x
        else:
            pad_x = (x_max - x_min) * 0.1
            x_min -= pad_x
            x_max += pad_x
        if y_min == y_max:
            pad_y = max(0.1, abs(y_min) * 0.1)
            y_min -= pad_y
            y_max += pad_y
        else:
            pad_y = (y_max - y_min) * 0.1
            y_min -= pad_y
            y_max += pad_y
        self.ax_event.set_xlim(x_min, x_max)
        self.ax_event.set_ylim(y_min, y_max)
        self.event_canvas.draw_idle()

    def _open_zoom_popup(self, event, color="#4a235a"):
        duration = event["duration"]
        pad = duration * 0.333
        start = max(self.time[0], event["start"] - pad)
        end = min(self.time[-1], event["end"] + pad)
        mask = (self.time >= start) & (self.time <= end)
        t = self.time[mask]
        p = self.filtered_power[mask]
        if t.size == 0:
            return

        top = tk.Toplevel(self.root)
        top.title(
            f"Event zoom {event['start']:.6f}s - {event['end']:.6f}s"
        )
        container = ttk.Frame(top)
        container.pack(fill="both", expand=True)

        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(t, p, color=color, lw=1.2)
        ax.axvspan(event["start"], event["end"], color=color, alpha=0.25)
        ax.axvline(event["start"], color="green", linestyle="--", lw=1.0, label="Start")
        ax.axvline(event["end"], color="red", linestyle="--", lw=1.0, label="Stop")
        threshold = self.last_threshold if self.last_threshold is not None else float(
            self.threshold_entry.get()
        )
        ax.axhline(threshold, color="#555", linestyle=":", lw=1.0, label="Threshold")
        try:
            threshold2 = (
                self.last_threshold2
                if self.last_threshold2 is not None
                else float(self.threshold2_entry.get())
            )
        except Exception:
            threshold2 = None
        if threshold2 is not None:
            ax.axhline(
                threshold2,
                color="#777",
                linestyle="--",
                lw=1.0,
                label="Threshold2",
            )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (mW)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats table
        stats_frame = ttk.Frame(container)
        stats_frame.pack(fill="x", padx=6, pady=4)
        stats = [
            ("Start (s)", f"{event['start']:.6f}"),
            ("End (s)", f"{event['end']:.6f}"),
            ("Length (s)", f"{event['duration']:.3f}"),
            ("Avg power (mW)", f"{event['avg_power']:.3f}"),
            ("Max power (mW)", f"{event['max_power']:.3f}"),
            ("Min power (mW)", f"{event['min_power']:.3f}"),
            ("Energy (mJ)", f"{event['energy_mj']:.3f}"),
        ]
        cols = 3  # number of label/value pairs per row
        for i, (label, val) in enumerate(stats):
            r = i // cols
            c = (i % cols) * 2
            ttk.Label(stats_frame, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=1)
            ttk.Label(stats_frame, text=val).grid(row=r, column=c + 1, sticky="e", padx=4, pady=1)

        # Copy actions
        actions = ttk.Frame(container)
        actions.pack(fill="x", padx=6, pady=(0, 6))

        def copy_stats():
            self.root.clipboard_clear()
            self.root.clipboard_append(self._format_event_stats(event))

        ttk.Button(actions, text="Copy stats", command=copy_stats).pack(
            side="left", padx=4, pady=2
        )


def main():
    root = tk.Tk()
    app = PowerAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
