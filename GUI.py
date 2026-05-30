# This file is used to visualize trace files.
import os
import re
import glob
import json
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


ROOT_TAG_IO = "json_g_io_MNCA_selector"
ROOT_TAG_TR = "json_g_traces_MNCA_selector"


_io_runs = sorted(glob.glob(os.path.join(ROOT_TAG_IO, "run_*")))
_tr_runs = sorted(glob.glob(os.path.join(ROOT_TAG_TR, "run_*")))

IO_ROOTS = _io_runs if _io_runs else [ROOT_TAG_IO]
TRACE_ROOTS = _tr_runs if _tr_runs else [ROOT_TAG_TR]


POOL_SIZE_TRAIN = 8


def float01_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_trace_json(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Trace shape must be (T,H,W,3), got {arr.shape} for {os.path.basename(path)}")
    return np.clip(arr, 0.0, 1.0)


def _is_allp_io(obj: dict, path: str, pool_hint: int = 8) -> bool:

    base = os.path.basename(path)
    if "_io_allP" in base:
        return True
    if base.endswith("_io.json") and "train" in path.replace("\\", "/"):

        pass


    val = obj.get("input", None)
    if not isinstance(val, list) or len(val) == 0:
        return False


    try:
        a0 = np.asarray(val[0])
    except Exception:
        return False


    if a0.ndim == 3 and a0.shape[-1] == 3 and len(val) <= max(2*pool_hint, 32):
        return True


    return False


def load_io_json_any(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    with open(path, "r") as f:
        obj = json.load(f)

    if _is_allp_io(obj, path, pool_hint=8):
        def to_imgs(key):
            return [np.asarray(x, dtype=np.float32) for x in obj[key]]
        ins  = [np.clip(a, 0.0, 1.0) for a in to_imgs("input")]
        tgts = [np.clip(a, 0.0, 1.0) for a in to_imgs("target")]
        preds= [np.clip(a, 0.0, 1.0) for a in to_imgs("pred")]
        return ins, tgts, preds


    inp = np.clip(np.asarray(obj["input"],  dtype=np.float32), 0.0, 1.0)
    tgt = np.clip(np.asarray(obj["target"], dtype=np.float32), 0.0, 1.0)
    prd = np.clip(np.asarray(obj["pred"],   dtype=np.float32), 0.0, 1.0)
    return [inp], [tgt], [prd]


def _strip_pool_segment(base: str) -> str:
    return re.sub(r"_P\d+_", "_", base)


def _parse_task_sched_split_from_path(path: str):
    norm = os.path.normpath(path).replace("\\", "/")
    parts = norm.split("/")


    if ROOT_TAG_IO in parts:
        ridx = parts.index(ROOT_TAG_IO)
    elif ROOT_TAG_TR in parts:
        ridx = parts.index(ROOT_TAG_TR)
    else:
        return None, None, None, None


    base = ridx + 1
    run_dir = parts[base] if base < len(parts) and parts[base].startswith("run_") else None
    if run_dir:
        base += 1

    task = parts[base] if base < len(parts) else None
    maybe = parts[base + 1] if base + 1 < len(parts) else None
    nxt   = parts[base + 2] if base + 2 < len(parts) else None

    if maybe in ("train", "test"):
        schedule = None
        split = maybe
    else:
        schedule = maybe
        split = nxt if nxt in ("train", "test") else None

    return task, schedule, split, run_dir


def _find_matching_train_traces(io_path: str) -> list[str]:
    task, schedule, split, run_dir = _parse_task_sched_split_from_path(io_path)
    if split != "train" or task is None:
        return []

    base = os.path.splitext(os.path.basename(io_path))[0]
    base = re.sub(r"_io(_allP)?$", "", base)
    base_noP = re.sub(r"_P\d+_", "_", base)

    m = re.search(r"^(?P<iter>iter_\d+)_pair(?P<pair>\d+)(?:_P\d+)?_(?P<steps>\d+)x$", base_noP)
    if m:
        iter_tag, pair_idx, steps_tag = m.group("iter"), m.group("pair"), m.group("steps")
    else:
        m2 = re.search(r"^(?P<iter>iter_\d+)_pair(?P<pair>\d+)", base_noP)
        if not m2:
            return []
        iter_tag, pair_idx, steps_tag = m2.group("iter"), m2.group("pair"), None


    roots = []
    for rt in TRACE_ROOTS:

        if run_dir and os.path.basename(rt) == run_dir:
            cand = os.path.join(rt, task, *( [schedule] if schedule else [] ), "train")
            if os.path.isdir(cand):
                roots.append(cand)
        elif not run_dir and os.path.basename(rt) != run_dir:

            cand = os.path.join(rt, task, *( [schedule] if schedule else [] ), "train")
            if os.path.isdir(cand):
                roots.append(cand)

    found = []
    for root in roots:
        if steps_tag:
            found += glob.glob(os.path.join(root, f"{iter_tag}_pair{pair_idx}_P*_{steps_tag}x.json"))
        found += glob.glob(os.path.join(root, f"{iter_tag}_pair{pair_idx}_P*.json"))


    keep = []
    for pth in found:
        b = os.path.splitext(os.path.basename(pth))[0]
        ok = b.startswith(f"{iter_tag}_pair{pair_idx}_P")
        if ok and steps_tag is not None:
            ok = (f"_{steps_tag}x" in b)
        if ok:
            keep.append(pth)

    return sorted(set(keep))


def _find_matching_test_trace(io_path: str) -> str | None:
    task, schedule, split, run_dir = _parse_task_sched_split_from_path(io_path)
    if split != "test":
        return None

    base = os.path.splitext(os.path.basename(io_path))[0]


    base = re.sub(r"_io(_allP)?$", "", base)
    base_noP = re.sub(r"_P\d+_", "_", base)
    roots = []
    for rt in TRACE_ROOTS:

        if run_dir and os.path.basename(rt) == run_dir:
            cand = os.path.join(rt, task or "", *( [schedule] if schedule else [] ), "test")
            if os.path.isdir(cand):
                roots.append(cand)
        elif not run_dir:
            cand = os.path.join(rt, task or "", *( [schedule] if schedule else [] ), "test")
            if os.path.isdir(cand):
                roots.append(cand)

    candidates = []
    for r in roots:
        candidates += glob.glob(os.path.join(r, f"{base_noP}*.json"))
    if not candidates:
        for r in roots:
            candidates += glob.glob(os.path.join(r, "*final*test*.json"))

    candidates = [c for c in sorted(set(candidates)) if os.path.isfile(c)]


    return candidates[0] if candidates else None


class GridViewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NCA – IO + P0..P7 trace viewer")
        self.minsize(1200, 820)


        self.mode = "train"


        self.io_input: List[Optional[np.ndarray]] = []
        self.io_target: List[Optional[np.ndarray]] = []
        self.io_pred:   List[Optional[np.ndarray]] = []


        self.traces: List[Optional[np.ndarray]] = []
        self.traces_u8: List[Optional[np.ndarray]] = []
        self.T = 0
        self.curr = 0


        self.playing = False
        self.loop = tk.BooleanVar(value=True)
        self.fps = tk.IntVar(value=8)
        self._tick_id = None


        self.show_overlay = tk.BooleanVar(value=False)
        self.min_scale_for_overlay = 10
        self.overlay_items_per_canvas = {}


        self.io_canvases = {}
        self.io_photos = {"Input": None, "Target": None, "Prediction": None}
        self.trace_frame_container = None
        self.trace_canvases: List[tk.Canvas] = []
        self.trace_photos: List[Optional[ImageTk.PhotoImage]] = []


        self._build_ui()


        self.bind("<Left>", lambda e: self.prev_frame())
        self.bind("<Right>", lambda e: self.next_frame())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if IO_ROOTS and os.path.isdir(IO_ROOTS[0]):
            self.last_io_dir = IO_ROOTS[0]
        elif os.path.isdir(ROOT_TAG_IO):
            self.last_io_dir = ROOT_TAG_IO
        else:
            self.last_io_dir = "."


    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_open_io = ttk.Button(top, text="Open IO (train/test)", command=self.open_io_file)
        self.btn_open_io.pack(side=tk.LEFT)

        ttk.Label(top, text="FPS:").pack(side=tk.LEFT, padx=(12, 2))
        self.spin_fps = ttk.Spinbox(top, from_=1, to=60, width=4, textvariable=self.fps, state="readonly",
                                    command=self._on_fps_changed)
        self.spin_fps.pack(side=tk.LEFT)

        self.btn_play = ttk.Button(top, text="▶ Play", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=(12, 0))

        self.chk_loop = ttk.Checkbutton(top, text="Loop", variable=self.loop)
        self.chk_loop.pack(side=tk.LEFT, padx=(8, 16))

        self.chk_overlay = ttk.Checkbutton(top, text="Show RGB values in cells", variable=self.show_overlay,
                                           command=self._redraw_all)
        self.chk_overlay.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(top, text="No IO selected")
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        nav = ttk.Frame(self)
        nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        self.btn_prev = ttk.Button(nav, text="◀ Previous", command=self.prev_frame, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT)
        self.btn_next = ttk.Button(nav, text="Next ▶", command=self.next_frame, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=(6, 0))
        self.lbl_step = ttk.Label(nav, text="Step: –/–")
        self.lbl_step.pack(side=tk.LEFT, padx=12)


        io_row = ttk.Frame(self)
        io_row.pack(side=tk.TOP, fill=tk.BOTH, padx=8, pady=(0, 6))
        for name in ("Input", "Target", "Prediction"):
            f = ttk.Frame(io_row, borderwidth=1, relief=tk.SOLID)
            f.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 6))
            title = ttk.Label(f, text=name, anchor="center")
            title.pack(side=tk.TOP, pady=(3, 3))
            canv = tk.Canvas(f, background="#222", height=260)
            canv.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=4, pady=(0, 6))
            self.io_canvases[name] = canv
        self.io_canvases["Prediction"].master.pack_configure(padx=(0, 0))


        self.trace_frame_container = ttk.Frame(self)
        self.trace_frame_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._build_trace_area(num_panels=POOL_SIZE_TRAIN)

    def _build_trace_area(self, num_panels: int):
        for w in self.trace_frame_container.winfo_children():
            w.destroy()
        self.trace_canvases.clear()
        self.trace_photos = [None] * num_panels
        self.overlay_items_per_canvas.clear()

        if num_panels == 1:

            holder = ttk.Frame(self.trace_frame_container, borderwidth=1, relief=tk.SOLID)
            holder.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
            ttk.Label(holder, text="Trace", anchor="center").pack(side=tk.TOP, pady=(2, 2))
            canv = tk.Canvas(holder, background="#111", height=420)
            canv.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=4, pady=(0, 6))
            self.trace_canvases.append(canv)
            self.overlay_items_per_canvas[canv] = []
        else:

            rows, cols = 2, 4
            idx = 0
            for r in range(rows):
                row = ttk.Frame(self.trace_frame_container)
                row.pack(fill=tk.BOTH, expand=True)
                for c in range(cols):
                    if idx >= num_panels: break
                    holder = ttk.Frame(row, borderwidth=1, relief=tk.SOLID)
                    holder.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 6), pady=(6, 0))
                    ttk.Label(holder, text=f"P{idx}", anchor="center").pack(side=tk.TOP, pady=(2, 2))
                    canv = tk.Canvas(holder, background="#111", height=220)
                    canv.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=4, pady=(0, 6))
                    self.trace_canvases.append(canv)
                    self.overlay_items_per_canvas[canv] = []
                    idx += 1
                row.pack_configure(padx=(0, 0))


    def open_io_file(self):
        path = filedialog.askopenfilename(
            title="Select IO file (train/test)",
            initialdir=self.last_io_dir if os.path.isdir(self.last_io_dir) else ROOT_TAG_IO,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        self.last_io_dir = os.path.dirname(path)

        try:
            self._load_from_io(path)
            self.lbl_status.config(text=os.path.basename(path))
        except Exception as e:
            messagebox.showerror("Error opening IO", str(e))

    def _load_from_io(self, io_path: str):

        if self.playing:
            self.playing = False
            self.btn_play.config(text="▶ Play")
        if self._tick_id is not None:
            try:
                self.after_cancel(self._tick_id)
            except Exception:
                pass
            self._tick_id = None
        self.io_input, self.io_target, self.io_pred = [], [], []
        self.traces, self.traces_u8 = [], []
        self.T = 0
        self.curr = 0
        self._redraw_all()

        task, schedule, split, run_dir = _parse_task_sched_split_from_path(io_path)
        self.mode = "test" if split == "test" else "train"
        run_txt = f" ({run_dir})" if run_dir else ""
        self.lbl_status.config(text=os.path.basename(io_path) + run_txt)


        in_list, tg_list, pr_list = load_io_json_any(io_path)

        if self.mode == "train":

            if len(in_list) == 1:
                in_list = in_list * POOL_SIZE_TRAIN
                tg_list = tg_list * POOL_SIZE_TRAIN
                pr_list = pr_list * POOL_SIZE_TRAIN
            else:

                in_list = in_list[:POOL_SIZE_TRAIN]
                tg_list = tg_list[:POOL_SIZE_TRAIN]
                pr_list = pr_list[:POOL_SIZE_TRAIN]
            self.io_input = in_list
            self.io_target = tg_list
            self.io_pred   = pr_list
            self._render_io_triplets(show_pred_from_index=0)


            trace_files = _find_matching_train_traces(io_path)
            if not trace_files:
                raise FileNotFoundError("No matching trace files found (train).")

            files_by_p = {}
            for pth in trace_files:
                b = os.path.basename(pth)
                m = re.search(r"_P(\d+)_", b) or re.search(r"_P(\d+)\.json$", b)
                if not m:
                    continue
                k = int(m.group(1))
                if 0 <= k < POOL_SIZE_TRAIN:
                    files_by_p[k] = pth


            traces = [None] * POOL_SIZE_TRAIN
            for k in range(POOL_SIZE_TRAIN):
                pth = files_by_p.get(k, None)
                if pth is None:
                    continue
                try:
                    traces[k] = load_trace_json(pth)
                except Exception as e:
                    print(f"[warning] could not load {pth}: {e}")

            Ts = [tr.shape[0] for tr in traces if tr is not None]
            if not Ts:
                raise FileNotFoundError("No valid trace files found (train).")
            T_min = min(Ts)
            traces = [tr[:T_min] if tr is not None else None for tr in traces]


            self._build_trace_area(num_panels=POOL_SIZE_TRAIN)
            self.traces = traces
            self.traces_u8 = [float01_to_u8(tr) if tr is not None else None for tr in traces]
            self.T = T_min
            self.curr = 0

        else:


            self.io_input = [in_list[0]]
            self.io_target = [tg_list[0]]
            self.io_pred   = [pr_list[0]]
            self._render_io_triplets(show_pred_from_index=0)

            trace_path = _find_matching_test_trace(io_path)
            if not trace_path:
                raise FileNotFoundError("No matching test trace found in json_g_traces/json_traces.")
            tr = load_trace_json(trace_path)


            self._build_trace_area(num_panels=1)
            self.traces = [tr]
            self.traces_u8 = [float01_to_u8(tr)]
            self.T = tr.shape[0]
            self.curr = 0


        state = (tk.NORMAL if self.T > 0 else tk.DISABLED)
        self.btn_prev.config(state=state)
        self.btn_next.config(state=state)
        self.btn_play.config(state=state)

        self._render_all_traces()
        self._update_step_label()


    def _render_io_triplets(self, show_pred_from_index: int = 0):
        if not self.io_input or self.io_input[0] is None:
            for canv in self.io_canvases.values():
                canv.delete("all")
                canv.create_text(10, 10, anchor=tk.NW, text="(missing IO)", fill="#aaa")
            return

        inp = float01_to_u8(self.io_input[0])
        tgt = float01_to_u8(self.io_target[0])
        prd_src_idx = min(show_pred_from_index, len(self.io_pred) - 1)
        prd = float01_to_u8(self.io_pred[prd_src_idx])

        items = [("Input", inp), ("Target", tgt), ("Prediction", prd)]
        for name, img_u8 in items:
            canv = self.io_canvases[name]
            canv.delete("all")
            cw = max(1, canv.winfo_width())
            ch = max(1, canv.winfo_height())
            H, W = img_u8.shape[:2]
            scale = min(cw / W, ch / H)
            new_w = max(1, int(W * scale))
            new_h = max(1, int(H * scale))
            im = Image.fromarray(img_u8, "RGB").resize((new_w, new_h), Image.Resampling.NEAREST)
            ph = ImageTk.PhotoImage(im)
            self.io_photos[name] = ph
            x = (cw - new_w) // 2
            y = (ch - new_h) // 2
            canv.create_image(x, y, anchor=tk.NW, image=ph)

    def _render_all_traces(self):
        for idx, canv in enumerate(self.trace_canvases):
            self._render_trace_panel(idx)

    def _render_trace_panel(self, p: int):
        canv = self.trace_canvases[p]
        canv.delete("all")
        self._clear_overlay(canv)
        if p >= len(self.traces_u8) or self.traces_u8[p] is None or self.T == 0:
            canv.create_text(10, 10, anchor=tk.NW, text="(missing trace)", fill="#aaa")
            return
        img_u8 = self.traces_u8[p][self.curr]
        cw = max(1, canv.winfo_width())
        ch = max(1, canv.winfo_height())
        H, W = img_u8.shape[:2]
        scale = min(cw / W, ch / H)
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        im = Image.fromarray(img_u8, "RGB").resize((new_w, new_h), Image.Resampling.NEAREST)
        ph = ImageTk.PhotoImage(im)
        self.trace_photos[p] = ph
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        canv.create_image(x, y, anchor=tk.NW, image=ph)


        if self.show_overlay.get() and scale >= self.min_scale_for_overlay:
            self._draw_rgb_overlay(canv, self.traces[p][self.curr], x, y, scale)

    def _draw_rgb_overlay(self, canv: tk.Canvas, img_f: np.ndarray, x0: int, y0: int, scale: float):
        H, W, _ = img_f.shape
        fs = max(6, int(scale * 0.12))
        items = []
        for iy in range(H):
            cy = y0 + (iy + 0.5) * scale
            for ix in range(W):
                cx = x0 + (ix + 0.5) * scale
                r, g, b = img_f[iy, ix, 0], img_f[iy, ix, 1], img_f[iy, ix, 2]
                lum = 0.2126*r + 0.7152*g + 0.0722*b
                color = "#FFFFFF" if lum < 0.5 else "#000000"
                text = f"R:{r:.2f}\nG:{g:.2f}\nB:{b:.2f}"
                shadow = "#000" if color == "#FFFFFF" else "#FFF"
                items.append(canv.create_text(cx+1, cy+1, text=text, fill=shadow,
                                              font=("TkDefaultFont", fs), anchor=tk.CENTER))
                items.append(canv.create_text(cx, cy, text=text, fill=color,
                                              font=("TkDefaultFont", fs), anchor=tk.CENTER))
        self.overlay_items_per_canvas[canv] = items

    def _clear_overlay(self, canv: tk.Canvas):
        items = self.overlay_items_per_canvas.get(canv, [])
        for iid in items:
            canv.delete(iid)
        self.overlay_items_per_canvas[canv] = []

    def _redraw_all(self):

        self._render_all_traces()

    def _update_step_label(self):
        if self.T == 0:
            self.lbl_step.config(text="Step: –/–")
        else:
            self.lbl_step.config(text=f"Step: {self.curr+1}/{self.T}")


    def prev_frame(self):
        if self.T == 0: return
        if self.curr > 0:
            self.curr -= 1
        elif self.loop.get():
            self.curr = self.T - 1
        self._render_all_traces()
        self._update_step_label()

    def next_frame(self):
        if self.T == 0: return
        if self.curr < self.T - 1:
            self.curr += 1
        elif self.loop.get():
            self.curr = 0
        self._render_all_traces()
        self._update_step_label()

    def toggle_play(self):
        if not self.playing:
            self.playing = True
            self.btn_play.config(text="⏸ Pause")
            self._schedule_tick()
        else:
            self.playing = False
            self.btn_play.config(text="▶ Play")
            if self._tick_id is not None:
                self.after_cancel(self._tick_id)
                self._tick_id = None

    def _on_fps_changed(self):
        if self.playing:
            self.toggle_play()
            self.toggle_play()

    def _schedule_tick(self):
        delay = int(1000 / max(1, self.fps.get()))
        self._tick_id = self.after(delay, self._tick)

    def _tick(self):
        if not self.playing:
            return
        self.next_frame()
        self._schedule_tick()


    def _on_close(self):
        self.playing = False
        if self._tick_id is not None:
            self.after_cancel(self._tick_id)
        self.destroy()


if __name__ == "__main__":
    app = GridViewer()
    app.mainloop()
