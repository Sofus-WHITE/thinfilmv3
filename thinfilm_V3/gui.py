"""Thinfilm V3 desktop GUI for thin-film simulation, fitting, and experiment review."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from datetime import datetime
import json
import queue
import re
import textwrap
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.parse import parse_qs, quote, urlparse
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon
import numpy as np
import pandas as pd
import yaml

from src.angle_sweep import run_angle_sweep
from src.color import prepare_color_conversion, reflectance_to_srgb, reflectance_to_xyz, xyz_to_srgb
from src.colorimetry import PerceivedColor, perceived_color_from_result
from src.experiments import (
    CachedExperimentResults,
    COLOUR_METRIC_CIE76,
    COLOUR_METRIC_CIEDE2000,
    ExperimentDataStore,
    LatestSputterRate,
    build_stack_from_estimates,
    cie_xy_background,
    colour_metric_label,
    default_experiment_cache_path,
    delta_e_colour,
    load_reflectance_csv,
    load_cached_results,
    normalise_colour_metric,
    normalize_substrate_name,
    sample_series_from_name,
    save_cached_results,
    xyz_to_lab,
    xyz_to_xy,
)
from src.diffuse_redistribution_model import (
    DiffuseRedistributionSettings,
    TMMWithDiffuseRedistributionModel,
)
from src.equipment_calibration_fit import EmpiricalFitResult, fit_empirical_refractive_index_model
from src.materials import (
    Material,
    built_in_materials,
    make_tabulated_material,
    material_profile_names,
    visible_material_table,
)
from src.nk_fitting import (
    default_fitted_constants_path,
    fit_single_film_constants,
    load_fitted_materials,
    save_fitted_constants,
)
from src.material_candidate_fit import (
    default_best_candidate_profile_path,
    fit_refractiveindex_candidates,
    grouped_best_candidate_profile_path,
    load_best_candidate_materials,
)
from src.stack import Layer, NativeOxide, make_stack, make_stack_with_interfaces, native_oxide_for_substrate
from src.rough_tmm_model import RoughnessCorrectionSettings, TMMWithRoughnessModel
from src.roughness_fit import fit_roughness_redistribution_parameters
from src.thickness_sweep import QUALITY_MODES, run_thickness_sweep_1d, run_thickness_sweep_2d
from src.thickness_optimization import (
    ThicknessOptimizationLayerResult,
    ThicknessOptimizationResult,
    default_thickness_optimization_cache_dir,
    optimize_experiment_thicknesses,
    save_optimization_summary_outputs,
)
from src.sputter_rate_fit import fit_sputter_rates_from_colour, save_sputter_rate_fit_outputs
from src.target_search import (
    TargetSearchResult,
    search_thicknesses_for_target_colour,
    wid2016_from_lab,
    xyz_from_lab,
    xyz_from_srgb,
)
from src.refractiveindex_db import (
    default_candidate_config_path,
    default_candidate_data_dir,
    download_candidate_records,
    load_candidate_config,
    material_from_refractiveindex_yaml,
    safe_candidate_name,
)
from src.tmm_model import TMMModel
from src.utils import wavelength_grid


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class LayerRow:
    """Editable GUI state for one deposited layer."""

    frame: ttk.Frame
    material_var: tk.StringVar
    thickness_var: tk.DoubleVar


class HoverTooltip:
    """Small tooltip used to reveal clipped dropdown text."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window: tk.Toplevel | None = None
        self.label: ttk.Label | None = None

    def show(self, text: str, x: int, y: int) -> None:
        text = str(text or "").strip()
        if not text:
            self.hide()
            return
        if self.window is None:
            self.window = tk.Toplevel(self.root)
            self.window.withdraw()
            self.window.overrideredirect(True)
            self.window.attributes("-topmost", True)
            self.label = tk.Label(
                self.window,
                text=text,
                padx=8,
                pady=4,
                justify=tk.LEFT,
                wraplength=620,
                relief=tk.SOLID,
                borderwidth=1,
                background="#fffdf2",
                foreground="#1f2933",
            )
            self.label.pack()
        elif self.label is not None:
            self.label.configure(text=text)
        self.window.geometry(f"+{x + 14}+{y + 14}")
        self.window.deiconify()

    def hide(self) -> None:
        if self.window is not None:
            self.window.withdraw()


class ThinFilmDesignerApp:
    """Tkinter GUI that calls the reusable thin-film backend."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Thinfilm V3 - optical fit workspace")
        self.root.geometry("1420x900")
        self.root.tk.call("tk", "scaling", 1.35)
        self._configure_style()

        self.materials: dict[str, Material] = built_in_materials()
        self.layer_rows: list[LayerRow] = []
        self.update_job: str | None = None
        self.current_stack = None
        self.sweep_counter = 0
        self.experiment_store: ExperimentDataStore | None = None
        self.latest_sputter_rates_cache: dict[str, LatestSputterRate] | None = None
        self.current_experiment_sample = None
        self.experiment_cache: CachedExperimentResults | None = None
        self.last_thickness_optimization_result: ThicknessOptimizationResult | None = None
        self.cached_thickness_fit_records: list[dict[str, object]] = []
        self.plots_before_points_cache: list[dict[str, object]] | None = None
        self.plots_after_points_cache: list[dict[str, object]] | None = None
        self.plots_deposited_samples_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
        self.colour_distance_fit_colour_cache: dict[str, tuple[np.ndarray, float]] = {}
        self.experiment_retune_job: str | None = None
        self.experiment_live_compare_job: str | None = None
        self.experiment_cache_path = self._experiment_cache_path()
        self.fitted_constants_path = default_fitted_constants_path(Path(__file__).resolve().parent)
        self.best_candidate_profile_path = default_best_candidate_profile_path(
            Path(__file__).resolve().parent
        )
        self.settings_path = Path(__file__).resolve().parent / "outputs" / "gui_settings.json"
        self.experiment_cie_ax = None
        self.background_task_running = False
        self.pause_requested = threading.Event()
        self.abort_requested = threading.Event()
        self.tooltip = HoverTooltip(self.root)
        self._combobox_tooltip_callbacks: list[str] = []
        self.fit_filter_combo_sets: list[dict[str, ttk.Combobox]] = []
        self.experiment_tree_hover_item: str | None = None
        self.settings_save_job: str | None = None
        self._saved_layer_settings: list[dict[str, object]] | None = None

        self._build_variables()
        self._load_gui_settings()
        self._build_layout()
        self._update_delta_e_labels()
        self._on_material_profile_changed()
        self._add_default_layers()
        self._load_saved_layers()
        self._refresh_all_layer_choices()
        self._refresh_search_layers()
        self._update_roughness_control_states()
        self._bind_settings_autosave()
        self.load_experiment_samples(show_errors=False)
        self.schedule_reflectance_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        """Apply a slightly cleaner visual style to Tk and Matplotlib."""

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg = "#f6f7f9"
        panel = "#ffffff"
        accent = "#256f7f"
        text = "#1f2933"
        muted = "#52606d"

        self.root.configure(bg=bg)
        self.root.option_add("*Font", ("Segoe UI", 11))
        style.configure(".", font=("Segoe UI", 11), background=bg, foreground=text)
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", background=bg, bordercolor="#d9e2ec")
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 11), foreground=text)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("TButton", padding=(8, 4), background=panel, foreground=text)
        style.map("TButton", background=[("active", "#e6f0f7")])
        style.configure("TCheckbutton", background=bg, foreground=text)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 6), font=("Segoe UI Semibold", 11))
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            rowheight=26,
            bordercolor="#d9e2ec",
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI Semibold", 10),
            background="#edf2f7",
            foreground=text,
        )
        style.configure("Accent.TButton", background=accent, foreground="#ffffff")

        matplotlib.rcParams.update(
            {
                "font.family": "Segoe UI",
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.facecolor": bg,
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#243b53",
                "axes.labelcolor": text,
                "xtick.color": muted,
                "ytick.color": muted,
                "savefig.dpi": 200,
            }
        )

    def _build_variables(self) -> None:
        self.substrate_var = tk.StringVar(value="Si")
        self.material_profile_var = tk.StringVar(value="current")
        self.angle_var = tk.DoubleVar(value=8.0)
        self.model_mode_var = tk.StringVar(value="Effective interface TMM")
        self.roughness_enabled_var = tk.BooleanVar(value=True)
        self.roughness_thickness_var = tk.DoubleVar(value=1.0)
        self.roughness_fraction_var = tk.DoubleVar(value=0.5)
        self.rms_roughness_var = tk.DoubleVar(value=1.0)
        self.scatter_scale_var = tk.DoubleVar(value=1.0)
        self.scatter_exponent_var = tk.DoubleVar(value=0.0)
        self.scatter_max_var = tk.DoubleVar(value=0.85)
        self.native_oxide_enabled_var = tk.BooleanVar(value=True)
        self.native_oxide_thickness_var = tk.DoubleVar(value=2.0)
        self.colour_metric_var = tk.StringVar(value=colour_metric_label(COLOUR_METRIC_CIE76))

        self.sweep_layer_1_var = tk.StringVar()
        self.sweep_layer_2_var = tk.StringVar()
        self.sweep_min_var = tk.DoubleVar(value=20.0)
        self.sweep_max_var = tk.DoubleVar(value=200.0)
        self.sweep_points_1d_var = tk.IntVar(value=100)
        self.sweep_points_2d_var = tk.IntVar(value=35)
        self.sweep_quality_var = tk.StringVar(value="fast")
        self.angle_sweep_min_var = tk.DoubleVar(value=0.0)
        self.angle_sweep_max_var = tk.DoubleVar(value=80.0)
        self.angle_sweep_points_var = tk.IntVar(value=80)
        self.constants_material_var = tk.StringVar(value="TiO2")
        self.constants_url_var = tk.StringVar()
        self.constants_source_var = tk.StringVar()
        default_experiment_data_path = (
            Path(__file__).resolve().parent.parent / "Reflectivity" / "sample_data"
        )
        self.experiment_data_path_var = tk.StringVar(
            value=str(default_experiment_data_path)
        )
        self.experiment_sample_var = tk.StringVar()
        self.experiment_measurement_var = tk.StringVar()
        self.cached_thickness_fit_var = tk.StringVar()
        self.experiment_series_filter_var = tk.StringVar(value="All")
        self.experiment_substrate_filter_var = tk.StringVar(value="All")
        self.experiment_surface_filter_var = tk.StringVar(value="All")
        self.experiment_kind_filter_var = tk.StringVar(value="All")
        self.experiment_plot_text_scale_var = tk.DoubleVar(value=0.72)
        self.experiment_data_overview_var = tk.StringVar(
            value="Load the Reflectivity/sample_data indexes to see the experiment overview."
        )
        self.samples_overview_info_var = tk.StringVar(
            value="Load Reflectivity/sample_data to see all sample names and measurement variants."
        )
        self.sample_measurements_info_var = tk.StringVar(
            value="Select a sample to see each measured spectrum and its condition."
        )
        self.sample_sweep_info_var = tk.StringVar(
            value="Select a sample and measurement row, then run sweeps for that exact configuration."
        )
        self.candidate_fit_group_vars: dict[str, tk.BooleanVar] = {}
        self.thickness_opt_range_percent_var = tk.DoubleVar(value=5.0)
        self.thickness_opt_step_percent_var = tk.DoubleVar(value=1.0)
        self.thickness_fit_mode_var = tk.StringVar(value="Individual layers")
        self.thickness_fit_scale_enabled_var = tk.BooleanVar(value=True)
        self.thickness_fit_scale_min_var = tk.DoubleVar(value=0.70)
        self.thickness_fit_scale_max_var = tk.DoubleVar(value=1.08)
        self.fit_sample_limit_var = tk.IntVar(value=0)
        self.fit_composition_filter_var = tk.StringVar(value="All")
        self.fit_rate_range_percent_var = tk.DoubleVar(value=50.0)
        self.fit_rate_points_var = tk.IntVar(value=81)
        self.colour_distance_source_var = tk.StringVar(value="Best cached thickness fit")
        self.plots_map_var = tk.StringVar()
        self.plots_fit_state_var = tk.StringVar(value="Before thickness fit")
        self.plots_info_var = tk.StringVar(
            value="Load experiment results, then refresh plot choices."
        )
        self.configuration_fit_var = tk.StringVar()
        self.configuration_fit_run_thickness_var = tk.BooleanVar(value=True)
        self.configuration_fit_info_var = tk.StringVar(
            value="Choose a configuration and run a bounded staged fit search."
        )
        self.empirical_fit_materials_var = tk.StringVar(value="TiO2, SiO2, Ag")
        self.empirical_fit_k_var = tk.BooleanVar(value=True)
        self.empirical_fit_thickness_dependence_var = tk.BooleanVar(value=True)
        self.empirical_fit_time_dependence_var = tk.BooleanVar(value=False)
        self.empirical_validation_fraction_var = tk.DoubleVar(value=0.20)
        self.empirical_lab_weight_var = tk.DoubleVar(value=0.02)
        self.empirical_max_evals_var = tk.IntVar(value=120)
        self.empirical_fit_info_var = tk.StringVar(
            value="Use filtered experiment rows to fit an equipment-calibrated effective n/k model."
        )
        self.calibration_group_var = tk.StringVar(value="smooth Si")
        self.calibration_rate_range_var = tk.DoubleVar(value=5.0)
        self.calibration_rate_points_var = tk.IntVar(value=21)
        self.search_target_mode_var = tk.StringVar(value="D65 white")
        self.search_target_hex_var = tk.StringVar(value="#ffffff")
        self.search_target_l_var = tk.DoubleVar(value=100.0)
        self.search_target_a_var = tk.DoubleVar(value=0.0)
        self.search_target_b_var = tk.DoubleVar(value=0.0)
        self.search_min_nm_var = tk.DoubleVar(value=20.0)
        self.search_max_nm_var = tk.DoubleVar(value=220.0)
        self.search_points_var = tk.IntVar(value=31)
        self.search_iterations_var = tk.IntVar(value=4)
        self.search_strategy_var = tk.StringVar(value="coordinate fast")
        self.search_min_lightness_var = tk.DoubleVar(value=92.0)
        self.search_brightness_weight_var = tk.DoubleVar(value=0.35)
        self.status_var = tk.StringVar(value="Ready")
        self.busy_text_var = tk.StringVar(value="")
        self.effective_interface_controls: list[tk.Widget] = []
        self.rms_roughness_controls: list[tk.Widget] = []

    def _build_layout(self) -> None:
        self._build_global_progress_bar()

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        controls_shell = ttk.Frame(main)
        output = ttk.Frame(main, padding=10)
        main.add(controls_shell, weight=0)
        main.add(output, weight=1)

        controls = self._build_left_controls_scroller(controls_shell)
        self.left_controls = controls
        self._build_stack_controls(controls)
        self._build_sweep_controls(controls)
        self._build_constants_candidate_controls(controls)
        self._build_experiment_model_controls(controls)
        self._build_output(output)
        self._update_left_panel_for_tab()

    def _build_left_controls_scroller(self, parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            borderwidth=0,
            bg="#f6f7f9",
            width=430,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        controls = ttk.Frame(canvas, padding=10)
        controls_window = canvas.create_window((0, 0), window=controls, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls.bind(
            "<Configure>",
            lambda _event: self._refresh_left_controls_scroller(),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(controls_window, width=event.width),
        )
        canvas.bind("<Enter>", self._bind_left_controls_mousewheel)
        canvas.bind("<Leave>", self._unbind_left_controls_mousewheel)
        controls.bind("<Enter>", self._bind_left_controls_mousewheel)
        controls.bind("<Leave>", self._unbind_left_controls_mousewheel)
        self.left_controls_canvas = canvas
        self.left_controls_window = controls_window
        return controls

    def _refresh_left_controls_scroller(self) -> None:
        canvas = getattr(self, "left_controls_canvas", None)
        if canvas is None:
            return
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
        except tk.TclError:
            pass

    def _bind_left_controls_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_left_controls_mousewheel)

    def _unbind_left_controls_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _on_left_controls_mousewheel(self, event) -> str:
        canvas = getattr(self, "left_controls_canvas", None)
        if canvas is None:
            return "break"
        delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta * 3, "units")
        return "break"

    def _build_global_progress_bar(self) -> None:
        """Create an always-visible progress strip for every calculation."""

        status_bar = ttk.Frame(self.root, padding=(10, 5), height=44)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.pack_propagate(False)
        self.global_progress_dog_canvas = tk.Canvas(
            status_bar,
            width=54,
            height=30,
            highlightthickness=0,
            bg="#f6f7f9",
        )
        self.global_progress_dog_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_progress_dog(self.global_progress_dog_canvas)
        self.global_progress = ttk.Progressbar(
            status_bar,
            mode="determinate",
            maximum=100,
            length=260,
        )
        self.global_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            status_bar,
            textvariable=self.busy_text_var,
            foreground="#1f2933",
            font=("Segoe UI Semibold", 10),
            width=48,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            status_bar,
            textvariable=self.status_var,
            foreground="#52606d",
            width=42,
            anchor=tk.E,
        ).pack(side=tk.RIGHT, padx=(8, 0))

    def _add_combobox_tooltip(self, combo: ttk.Combobox, text_getter) -> None:
        """Show full text for clipped combobox values and dropdown items."""

        combo.bind(
            "<Enter>",
            lambda event, getter=text_getter: self._show_widget_tooltip(event, getter()),
            add="+",
        )
        combo.bind(
            "<Motion>",
            lambda event, getter=text_getter: self._show_widget_tooltip(event, getter()),
            add="+",
        )
        combo.bind("<Leave>", lambda _event: self.tooltip.hide(), add="+")

        motion_command = self.root.register(self._show_combobox_popdown_tooltip)
        hide_command = self.root.register(lambda *_args: self.tooltip.hide())
        self._combobox_tooltip_callbacks.extend([motion_command, hide_command])
        combo.configure(
            postcommand=lambda c=combo, mc=motion_command, hc=hide_command: self._prepare_combobox_popdown_tooltip(
                c, mc, hc
            )
        )

    def _show_widget_tooltip(self, event: tk.Event, text: str) -> None:
        self.tooltip.show(str(text or ""), int(event.x_root), int(event.y_root))

    def _add_tooltip(self, widget: tk.Widget, text: str) -> tk.Widget:
        def show(event: tk.Event, message: str = text) -> None:
            self._show_widget_tooltip(event, message)

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Motion>", show, add="+")
        widget.bind("<Leave>", lambda _event: self.tooltip.hide(), add="+")
        return widget

    def _prepare_combobox_popdown_tooltip(
        self, combo: ttk.Combobox, motion_command: str, hide_command: str
    ) -> None:
        """Make an opened ttk combobox list readable and bind hover tooltips."""

        values = [str(value) for value in combo.cget("values")]
        if values:
            dropdown_width = max(int(combo.cget("width") or 0), min(max(len(value) for value in values) + 2, 90))
        else:
            dropdown_width = int(combo.cget("width") or 0)
        try:
            popdown = combo.tk.call("ttk::combobox::PopdownWindow", str(combo))
            listbox = f"{popdown}.f.l"
            combo.tk.call(listbox, "configure", "-width", dropdown_width)
            combo.tk.call("bind", listbox, "<Motion>", f"{motion_command} %W %y %X %Y")
            combo.tk.call("bind", listbox, "<Leave>", hide_command)
            combo.tk.call("bind", listbox, "<ButtonRelease-1>", hide_command)
        except tk.TclError:
            pass

    def _show_combobox_popdown_tooltip(self, widget_path: str, y: str, x_root: str, y_root: str) -> None:
        try:
            index = self.root.tk.call(widget_path, "nearest", int(float(y)))
            text = self.root.tk.call(widget_path, "get", index)
        except tk.TclError:
            self.tooltip.hide()
            return
        self.tooltip.show(str(text), int(float(x_root)), int(float(y_root)))

    def _build_stack_controls(self, parent: ttk.Frame) -> None:
        stack_box = ttk.LabelFrame(parent, text="Stack", padding=8)
        self.stack_controls_box = stack_box
        stack_box.pack(fill=tk.X, pady=(0, 8))

        substrate_row = ttk.Frame(stack_box)
        substrate_row.pack(fill=tk.X, pady=2)
        ttk.Label(substrate_row, text="Substrate").pack(side=tk.LEFT)
        substrate_combo = ttk.Combobox(
            substrate_row,
            textvariable=self.substrate_var,
            values=self._substrate_names(),
            width=12,
            state="readonly",
        )
        substrate_combo.pack(side=tk.RIGHT)
        substrate_combo.bind("<<ComboboxSelected>>", self._on_substrate_changed)

        profile_row = ttk.Frame(stack_box)
        profile_row.pack(fill=tk.X, pady=2)
        ttk.Label(profile_row, text="Constants").pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(
            profile_row,
            textvariable=self.material_profile_var,
            values=self._material_profile_choices(),
            width=16,
            state="readonly",
        )
        self.profile_combo.pack(side=tk.RIGHT)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_material_profile_changed)
        self._add_combobox_tooltip(self.profile_combo, self.material_profile_var.get)

        model_row = ttk.Frame(stack_box)
        model_row.pack(fill=tk.X, pady=2)
        ttk.Label(model_row, text="Optical model").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_mode_var,
            values=(
                "Ideal TMM",
                "Effective interface TMM",
                "RMS roughness TMM",
                "Effective interface + RMS",
                "Diffuse redistribution TMM",
                "Effective interface + diffuse redistribution",
            ),
            width=24,
            state="readonly",
        )
        model_combo.pack(side=tk.RIGHT)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_mode_changed)

        angle_row = ttk.Frame(stack_box)
        angle_row.pack(fill=tk.X, pady=2)
        ttk.Label(angle_row, text="Angle (deg)").pack(side=tk.LEFT)
        self._spinbox(angle_row, self.angle_var, 0.0, 89.0, 0.5, self._on_stack_changed).pack(
            side=tk.RIGHT
        )

        rough_check = ttk.Checkbutton(
            stack_box,
            text="Mixed roughness interfaces",
            variable=self.roughness_enabled_var,
            command=self._on_stack_changed,
        )
        rough_check.pack(anchor=tk.W, pady=(6, 0))
        self.effective_interface_controls.append(rough_check)

        rough_row = ttk.Frame(stack_box)
        rough_row.pack(fill=tk.X, pady=2)
        ttk.Label(rough_row, text="Interface nm").pack(side=tk.LEFT)
        interface_spin = self._spinbox(
            rough_row,
            self.roughness_thickness_var,
            0.0,
            20.0,
            0.25,
            self._on_stack_changed,
        )
        interface_spin.pack(side=tk.RIGHT)
        self.effective_interface_controls.append(interface_spin)

        frac_row = ttk.Frame(stack_box)
        frac_row.pack(fill=tk.X, pady=2)
        ttk.Label(frac_row, text="Mix fraction").pack(side=tk.LEFT)
        mix_spin = self._spinbox(
            frac_row,
            self.roughness_fraction_var,
            0.0,
            1.0,
            0.05,
            self._on_stack_changed,
        )
        mix_spin.pack(side=tk.RIGHT)
        self.effective_interface_controls.append(mix_spin)

        rms_row = ttk.Frame(stack_box)
        rms_row.pack(fill=tk.X, pady=2)
        ttk.Label(rms_row, text="RMS roughness nm").pack(side=tk.LEFT)
        rms_spin = self._spinbox(
            rms_row,
            self.rms_roughness_var,
            0.0,
            50.0,
            0.25,
            self._on_stack_changed,
        )
        rms_spin.pack(side=tk.RIGHT)
        self.rms_roughness_controls.append(rms_spin)

        scatter_row = ttk.Frame(stack_box)
        scatter_row.pack(fill=tk.X, pady=2)
        ttk.Label(scatter_row, text="Scatter scale").pack(side=tk.LEFT)
        scatter_spin = self._spinbox(
            scatter_row,
            self.scatter_scale_var,
            0.0,
            10.0,
            0.1,
            self._on_stack_changed,
        )
        scatter_spin.pack(side=tk.RIGHT)
        self.rms_roughness_controls.append(scatter_spin)

        scatter_exp_row = ttk.Frame(stack_box)
        scatter_exp_row.pack(fill=tk.X, pady=2)
        ttk.Label(scatter_exp_row, text="Scatter exponent").pack(side=tk.LEFT)
        scatter_exp_spin = self._spinbox(
            scatter_exp_row,
            self.scatter_exponent_var,
            -4.0,
            6.0,
            0.25,
            self._on_stack_changed,
        )
        scatter_exp_spin.pack(side=tk.RIGHT)
        self.rms_roughness_controls.append(scatter_exp_spin)

        scatter_max_row = ttk.Frame(stack_box)
        scatter_max_row.pack(fill=tk.X, pady=2)
        ttk.Label(scatter_max_row, text="Max scatter fraction").pack(side=tk.LEFT)
        scatter_max_spin = self._spinbox(
            scatter_max_row,
            self.scatter_max_var,
            0.0,
            1.0,
            0.05,
            self._on_stack_changed,
        )
        scatter_max_spin.pack(side=tk.RIGHT)
        self.rms_roughness_controls.append(scatter_max_spin)

        ttk.Checkbutton(
            stack_box,
            text="Native substrate oxide",
            variable=self.native_oxide_enabled_var,
            command=self._on_stack_changed,
        ).pack(anchor=tk.W, pady=(6, 0))

        oxide_row = ttk.Frame(stack_box)
        oxide_row.pack(fill=tk.X, pady=2)
        ttk.Label(oxide_row, text="Oxide nm").pack(side=tk.LEFT)
        self._spinbox(
            oxide_row,
            self.native_oxide_thickness_var,
            0.0,
            50.0,
            0.25,
            self._on_stack_changed,
        ).pack(side=tk.RIGHT)

        layers_header = ttk.Frame(stack_box)
        layers_header.pack(fill=tk.X, pady=(10, 2))
        ttk.Label(layers_header, text="Deposited layers").pack(side=tk.LEFT)
        ttk.Button(layers_header, text="+", width=3, command=self.add_layer).pack(side=tk.RIGHT)

        self.layers_frame = ttk.Frame(stack_box)
        self.layers_frame.pack(fill=tk.X)

        rate_header = ttk.Frame(stack_box)
        rate_header.pack(fill=tk.X, pady=(10, 2))
        ttk.Label(rate_header, text="Latest sputter-rate time").pack(side=tk.LEFT)
        ttk.Button(
            rate_header,
            text="Refresh",
            command=self._refresh_latest_sputter_rates,
        ).pack(side=tk.RIGHT)

        rate_columns = ("layer", "rate", "time", "settings")
        self.sputter_time_tree = ttk.Treeview(
            stack_box,
            columns=rate_columns,
            show="headings",
            height=4,
            selectmode="none",
        )
        rate_headings = {
            "layer": "Layer",
            "rate": "nm/min",
            "time": "Time",
            "settings": "Settings",
        }
        rate_widths = {"layer": 68, "rate": 72, "time": 70, "settings": 170}
        for column in rate_columns:
            self.sputter_time_tree.heading(column, text=rate_headings[column])
            self.sputter_time_tree.column(column, width=rate_widths[column], anchor=tk.W)
        self.sputter_time_tree.pack(fill=tk.X)
        self.sputter_time_total_var = tk.StringVar(
            value="Load Reflectivity data to estimate sputter time."
        )
        ttk.Label(
            stack_box,
            textvariable=self.sputter_time_total_var,
            foreground="#52606d",
            wraplength=320,
        ).pack(anchor=tk.W, pady=(3, 0))

    def _build_sweep_controls(self, parent: ttk.Frame) -> None:
        sweep_box = ttk.LabelFrame(parent, text="Sweep commands", padding=8)
        self.sweep_controls_box = sweep_box
        sweep_box.pack(fill=tk.X)

        ttk.Label(
            sweep_box,
            text="Thickness sweep",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 2))
        for label, variable in (
            ("Layer 1", self.sweep_layer_1_var),
            ("Layer 2", self.sweep_layer_2_var),
        ):
            row = ttk.Frame(sweep_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            combo = ttk.Combobox(row, textvariable=variable, width=16, state="readonly")
            combo.pack(side=tk.RIGHT)
            if label == "Layer 1":
                self.sweep_layer_1_combo = combo
            else:
                self.sweep_layer_2_combo = combo

        for label, variable, minimum, maximum, increment in (
            ("Min nm", self.sweep_min_var, 0.0, 1000.0, 1.0),
            ("Max nm", self.sweep_max_var, 0.0, 1000.0, 1.0),
            ("1D points", self.sweep_points_1d_var, 10, 400, 5),
            ("2D points", self.sweep_points_2d_var, 8, 80, 1),
        ):
            row = ttk.Frame(sweep_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            self._spinbox(row, variable, minimum, maximum, increment, None).pack(side=tk.RIGHT)

        quality_row = ttk.Frame(sweep_box)
        quality_row.pack(fill=tk.X, pady=2)
        ttk.Label(quality_row, text="Quality").pack(side=tk.LEFT)
        quality_combo = ttk.Combobox(
            quality_row,
            textvariable=self.sweep_quality_var,
            values=("fast", "normal", "high_quality"),
            state="readonly",
            width=12,
        )
        quality_combo.pack(side=tk.RIGHT)
        quality_combo.bind("<<ComboboxSelected>>", self._on_quality_changed)

        ttk.Button(sweep_box, text="Run 1-layer thickness sweep", command=self.run_1d_sweep).pack(
            fill=tk.X, pady=(8, 2)
        )
        ttk.Button(sweep_box, text="Run 2-layer thickness sweep", command=self.run_2d_sweep).pack(fill=tk.X)

        ttk.Separator(sweep_box).pack(fill=tk.X, pady=(8, 6))
        ttk.Label(
            sweep_box,
            text="Angular sweep",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 2))
        for label, variable, minimum, maximum, increment in (
            ("Angle min", self.angle_sweep_min_var, 0.0, 89.0, 1.0),
            ("Angle max", self.angle_sweep_max_var, 0.0, 89.0, 1.0),
            ("Angle pts", self.angle_sweep_points_var, 10, 300, 5),
        ):
            row = ttk.Frame(sweep_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            self._spinbox(row, variable, minimum, maximum, increment, None).pack(side=tk.RIGHT)
        ttk.Button(sweep_box, text="Run angular sweep", command=self.run_angle_sweep).pack(
            fill=tk.X, pady=(8, 0)
        )

    def _build_experiment_model_controls(self, parent: ttk.Frame) -> None:
        settings_box = ttk.LabelFrame(parent, text="Experiment simulation settings", padding=8)
        self.experiment_settings_box = settings_box
        settings_box.pack(fill=tk.X, pady=(0, 8))

        profile_row = ttk.Frame(settings_box)
        profile_row.pack(fill=tk.X, pady=2)
        ttk.Label(profile_row, text="Constants").pack(side=tk.LEFT)
        self.experiment_profile_combo = ttk.Combobox(
            profile_row,
            textvariable=self.material_profile_var,
            values=self._material_profile_choices(),
            width=20,
            state="readonly",
        )
        self.experiment_profile_combo.pack(side=tk.RIGHT)
        self.experiment_profile_combo.bind("<<ComboboxSelected>>", self._on_material_profile_changed)
        self._add_combobox_tooltip(self.experiment_profile_combo, self.material_profile_var.get)

        model_row = ttk.Frame(settings_box)
        model_row.pack(fill=tk.X, pady=2)
        ttk.Label(model_row, text="Optical model").pack(side=tk.LEFT)
        experiment_model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_mode_var,
            values=(
                "Ideal TMM",
                "Effective interface TMM",
                "RMS roughness TMM",
                "Effective interface + RMS",
                "Diffuse redistribution TMM",
                "Effective interface + diffuse redistribution",
            ),
            width=24,
            state="readonly",
        )
        experiment_model_combo.pack(side=tk.RIGHT)
        experiment_model_combo.bind("<<ComboboxSelected>>", self._on_model_mode_changed)
        self._add_combobox_tooltip(experiment_model_combo, self.model_mode_var.get)

        angle_row = ttk.Frame(settings_box)
        angle_row.pack(fill=tk.X, pady=2)
        ttk.Label(angle_row, text="Angle (deg)").pack(side=tk.LEFT)
        self._spinbox(angle_row, self.angle_var, 0.0, 89.0, 0.5, self._on_stack_changed).pack(
            side=tk.RIGHT
        )

        metric_row = ttk.Frame(settings_box)
        metric_row.pack(fill=tk.X, pady=2)
        ttk.Label(metric_row, text="Colour metric").pack(side=tk.LEFT)
        self.colour_metric_combo = ttk.Combobox(
            metric_row,
            textvariable=self.colour_metric_var,
            values=(
                colour_metric_label(COLOUR_METRIC_CIE76),
                colour_metric_label(COLOUR_METRIC_CIEDE2000),
            ),
            width=20,
            state="readonly",
        )
        self.colour_metric_combo.pack(side=tk.RIGHT)
        self.colour_metric_combo.bind("<<ComboboxSelected>>", self._on_colour_metric_changed)
        self._add_combobox_tooltip(self.colour_metric_combo, self._current_colour_metric_label)

        rough_check = ttk.Checkbutton(
            settings_box,
            text="Mixed roughness interfaces",
            variable=self.roughness_enabled_var,
            command=self._on_stack_changed,
        )
        rough_check.pack(anchor=tk.W, pady=(6, 0))
        self.effective_interface_controls.append(rough_check)

        for label, variable, minimum, maximum, increment, group in (
            ("Interface nm", self.roughness_thickness_var, 0.0, 20.0, 0.25, "effective"),
            ("Mix fraction", self.roughness_fraction_var, 0.0, 1.0, 0.05, "effective"),
            ("RMS roughness nm", self.rms_roughness_var, 0.0, 50.0, 0.25, "rms"),
            ("Scatter scale", self.scatter_scale_var, 0.0, 10.0, 0.1, "rms"),
            ("Scatter exponent", self.scatter_exponent_var, -4.0, 6.0, 0.25, "rms"),
            ("Max scatter fraction", self.scatter_max_var, 0.0, 1.0, 0.05, "rms"),
        ):
            row = ttk.Frame(settings_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            spin = self._spinbox(row, variable, minimum, maximum, increment, self._on_stack_changed)
            spin.pack(side=tk.RIGHT)
            if group == "effective":
                self.effective_interface_controls.append(spin)
            else:
                self.rms_roughness_controls.append(spin)

        ttk.Checkbutton(
            settings_box,
            text="Native substrate oxide",
            variable=self.native_oxide_enabled_var,
            command=self._on_stack_changed,
        ).pack(anchor=tk.W, pady=(6, 0))

        oxide_row = ttk.Frame(settings_box)
        oxide_row.pack(fill=tk.X, pady=2)
        ttk.Label(oxide_row, text="Oxide nm").pack(side=tk.LEFT)
        self._spinbox(
            oxide_row,
            self.native_oxide_thickness_var,
            0.0,
            50.0,
            0.25,
            self._on_stack_changed,
        ).pack(side=tk.RIGHT)

        ttk.Button(
            settings_box,
            text="Update selected experiment",
            command=self.run_experiment_comparison,
        ).pack(fill=tk.X, pady=(10, 0))
        ttk.Label(
            settings_box,
            text="Applies these settings to the selected measurement only. Use Build / refresh saved results for the full table.",
            foreground="#52606d",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_experiment_optimizer_controls(self, parent: ttk.Frame) -> None:
        opt_box = ttk.LabelFrame(parent, text="Experiment thickness fit", padding=8)
        opt_box.pack(fill=tk.X, pady=(8, 0))

        for label, variable, minimum, maximum, increment in (
            ("Rate error +/- %", self.thickness_opt_range_percent_var, 0.0, 25.0, 0.5),
            ("Step %", self.thickness_opt_step_percent_var, 0.1, 10.0, 0.1),
        ):
            row = ttk.Frame(opt_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            self._spinbox(row, variable, minimum, maximum, increment, None).pack(side=tk.RIGHT)

        optimize_button = ttk.Button(
            opt_box,
            text="Optimize selected experiment",
            command=self.optimize_selected_experiment_thicknesses,
        )
        optimize_button.pack(fill=tk.X, pady=(8, 0))
        self._add_tooltip(
            optimize_button,
            "Fits thicknesses for the selected experiment measurement only. It uses the current constants/model settings and saves reusable cached trials.",
        )
        precalculate_button = ttk.Button(
            opt_box,
            text="Precalculate all...",
            command=self.precalculate_all_thickness_optimizations,
        )
        precalculate_button.pack(fill=tk.X, pady=(4, 0))
        self._add_tooltip(
            precalculate_button,
            "Runs cached thickness fits for the filtered experiment rows. This is intended for long runs; cached trial results are reused later.",
        )

        ttk.Label(
            opt_box,
            text="Select a row for one fit, or pre-cache all rows overnight.",
            foreground="#52606d",
        ).pack(anchor=tk.W, pady=(6, 0))

    def _build_constants_candidate_controls(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Candidate constant fitting", padding=8)
        self.constants_candidate_box = box

        ttk.Label(
            box,
            text=(
                "This tests complete refractiveindex.info constant sets against "
                "experiment groups. It does not fit only the material selected "
                "in the table."
            ),
            foreground="#52606d",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(0, 8))

        saved_settings = {}
        try:
            if self.settings_path.exists():
                saved_settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            saved_settings = {}

        for option in self._candidate_fit_group_options():
            settings_key = f"candidate_fit_group_{option['key']}"
            default_value = bool(saved_settings.get(settings_key, option["default"]))
            variable = tk.BooleanVar(value=default_value)
            self.candidate_fit_group_vars[option["key"]] = variable
            ttk.Checkbutton(
                box,
                text=option["label"],
                variable=variable,
            ).pack(anchor=tk.W, pady=1)

        button_row = ttk.Frame(box)
        button_row.pack(fill=tk.X, pady=(8, 0))
        candidate_button = ttk.Button(
            button_row,
            text="Fit selected groups",
            command=self.fit_refractiveindex_candidate_constants,
        )
        candidate_button.pack(fill=tk.X)
        self._add_tooltip(
            candidate_button,
            "Tests complete refractiveindex.info datasets for the selected experiment groups and saves the best constants profile for each group.",
        )

        ttk.Label(
            box,
            text="Saved results appear as best_candidates_* profiles in the constants dropdown.",
            foreground="#52606d",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(6, 0))


    def _build_output(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        self.reflectance_tab = ttk.Frame(self.notebook)
        self.sweep_tab = ttk.Frame(self.notebook)
        self.experiments_tab = ttk.Frame(self.notebook)
        self.samples_tab = ttk.Frame(self.notebook)
        self.fit_optimize_tab = ttk.Frame(self.notebook)
        self.configuration_fit_tab = ttk.Frame(self.notebook)
        self.empirical_fit_tab = ttk.Frame(self.notebook)
        self.search_tab = ttk.Frame(self.notebook)
        self.colour_distance_tab = ttk.Frame(self.notebook)
        self.plots_tab = ttk.Frame(self.notebook)
        self.constants_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reflectance_tab, text="Reflectance")
        self.notebook.add(self.sweep_tab, text="Sweep")
        self.notebook.add(self.experiments_tab, text="Experiments")
        self.notebook.add(self.samples_tab, text="Samples")
        self.notebook.add(self.fit_optimize_tab, text="Fit & Optimize")
        self.notebook.add(self.configuration_fit_tab, text="Configuration Fit")
        self.notebook.add(self.empirical_fit_tab, text="Empirical Fit")
        self.notebook.add(self.search_tab, text="Search")
        self.notebook.add(self.colour_distance_tab, text="Colour Distance")
        self.notebook.add(self.plots_tab, text="Plots")
        self.notebook.add(self.constants_tab, text="Constants")

        self.reflectance_figure = Figure(figsize=(8.4, 6.2), dpi=170)
        reflectance_header = ttk.Frame(self.reflectance_tab)
        reflectance_header.pack(fill=tk.X)
        self._pack_download_figure_button(
            reflectance_header,
            lambda: self.reflectance_figure,
            "reflectance",
        )
        self.reflectance_canvas = FigureCanvasTkAgg(self.reflectance_figure, self.reflectance_tab)
        self.reflectance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.sweep_results_notebook = ttk.Notebook(self.sweep_tab)
        self.sweep_results_notebook.pack(fill=tk.BOTH, expand=True)
        self._build_samples_tab(self.samples_tab)
        self._build_experiments_tab(self.experiments_tab)
        self._build_fit_optimize_tab(self.fit_optimize_tab)
        self._build_configuration_fit_tab(self.configuration_fit_tab)
        self._build_empirical_fit_tab(self.empirical_fit_tab)
        self._build_search_tab(self.search_tab)
        self._build_colour_distance_tab(self.colour_distance_tab)
        self._build_plots_tab(self.plots_tab)
        self._build_constants_tab(self.constants_tab)

    def _build_fit_optimize_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        workflow = ttk.LabelFrame(wrapper, text="Fit workflow", padding=10)
        workflow.pack(fill=tk.X, pady=(0, 8))

        source_row = ttk.Frame(workflow)
        source_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(source_row, textvariable=self.experiment_data_overview_var, foreground="#52606d").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        load_indexes_button = ttk.Button(source_row, text="Load indexes", command=self.load_experiment_samples)
        load_indexes_button.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(
            load_indexes_button,
            "Reloads the Reflectivity/sample_data index files: samples, measurements, thickness estimates, and substrate/surface labels. It does not run optical simulations.",
        )
        build_cache_button = ttk.Button(source_row, text="Build model cache", command=self.build_experiment_cache)
        build_cache_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            build_cache_button,
            "Simulates every filtered experiment with the current constants, optical model, angle, interface, oxide, and roughness settings. Saves the before-fit model results for later comparison.",
        )

        filter_box = ttk.LabelFrame(workflow, text="1. Choose experiment rows", padding=8)
        filter_box.pack(fill=tk.X, pady=(0, 6))
        filter_row = ttk.Frame(filter_box)
        filter_row.pack(fill=tk.X)
        self._pack_fit_filter_controls(filter_row)

        settings_box = ttk.LabelFrame(workflow, text="2. Fit settings", padding=8)
        settings_box.pack(fill=tk.X, pady=(0, 6))
        thickness_row = ttk.Frame(settings_box)
        thickness_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(thickness_row, text="Thickness +/- %").pack(side=tk.LEFT)
        self._spinbox(
            thickness_row,
            self.thickness_opt_range_percent_var,
            0.0,
            25.0,
            0.5,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(thickness_row, text="Step %").pack(side=tk.LEFT)
        self._spinbox(
            thickness_row,
            self.thickness_opt_step_percent_var,
            0.1,
            10.0,
            0.1,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(thickness_row, text="Thickness mode").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            thickness_row,
            textvariable=self.thickness_fit_mode_var,
            values=("Individual layers", "Same material together"),
            state="readonly",
            width=22,
        )
        mode_combo.pack(side=tk.LEFT, padx=(6, 12))
        self._add_tooltip(
            mode_combo,
            "Individual layers fits each deposited layer separately. Same material together applies one shared percentage change to repeated layers of the same material inside a sample.",
        )
        scale_check = ttk.Checkbutton(
            thickness_row,
            text="Fit reflectance scale",
            variable=self.thickness_fit_scale_enabled_var,
        )
        scale_check.pack(side=tk.LEFT, padx=(4, 8))
        self._add_tooltip(
            scale_check,
            "Also fits one wavelength-flat multiplier for the simulated reflectance. Use this when the curve shape is right but the measured reflectance is lower or higher overall.",
        )
        ttk.Label(thickness_row, text="Scale min/max").pack(side=tk.LEFT)
        self._spinbox(
            thickness_row,
            self.thickness_fit_scale_min_var,
            0.3,
            1.2,
            0.01,
            None,
        ).pack(side=tk.LEFT, padx=(6, 4))
        self._spinbox(
            thickness_row,
            self.thickness_fit_scale_max_var,
            0.5,
            1.5,
            0.01,
            None,
        ).pack(side=tk.LEFT, padx=(4, 12))
        rate_row = ttk.Frame(settings_box)
        rate_row.pack(fill=tk.X)
        ttk.Label(rate_row, text="Rate fit +/- %").pack(side=tk.LEFT)
        self._spinbox(
            rate_row,
            self.fit_rate_range_percent_var,
            1.0,
            100.0,
            1.0,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rate_row, text="Rate points").pack(side=tk.LEFT)
        self._spinbox(
            rate_row,
            self.fit_rate_points_var,
            5,
            201,
            2,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rate_row, text="Sample limit").pack(side=tk.LEFT)
        self._spinbox(
            rate_row,
            self.fit_sample_limit_var,
            0,
            5000,
            1,
            self._on_fit_filter_changed,
        ).pack(side=tk.LEFT, padx=(6, 12))

        action_box = ttk.LabelFrame(workflow, text="3. Run fitting actions", padding=8)
        action_box.pack(fill=tk.X)
        action_row = ttk.Frame(action_box)
        action_row.pack(fill=tk.X)
        thickness_actions = ttk.LabelFrame(action_row, text="Thickness", padding=6)
        thickness_actions.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)
        rate_actions = ttk.LabelFrame(action_row, text="Rates / model", padding=6)
        rate_actions.pack(side=tk.LEFT, padx=(0, 8), fill=tk.Y)
        review_actions = ttk.LabelFrame(action_row, text="Review", padding=6)
        review_actions.pack(side=tk.LEFT, fill=tk.Y)
        optimize_button = ttk.Button(
            thickness_actions,
            text="Optimize selected",
            command=self.optimize_selected_experiment_thicknesses,
        )
        optimize_button.pack(side=tk.LEFT)
        self._add_tooltip(
            optimize_button,
            "Fits thicknesses for only the selected experiment measurement. In Individual layers mode, each layer can move separately to minimize Delta E for that sample.",
        )
        overnight_button = ttk.Button(
            thickness_actions,
            text="Overnight thickness cache",
            command=self.precalculate_all_thickness_optimizations,
        )
        overnight_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            overnight_button,
            "Runs the selected thickness mode for every currently filtered measurement. Individual layers gives one best-fit thickness stack per sample; cached trials are reused.",
        )
        selected_rates_button = ttk.Button(
            rate_actions,
            text="Fit selected rate groups",
            command=self.fit_selected_sputter_rate_groups_from_colour,
        )
        selected_rates_button.pack(side=tk.LEFT)
        self._add_tooltip(
            selected_rates_button,
            "Fits shared sputter-rate corrections for the selected rate groups. This is the group-level fit: all samples in the same sputter-rate group share one fitted rate.",
        )
        all_rates_button = ttk.Button(
            rate_actions,
            text="Fit all visible rate groups",
            command=self.fit_all_sputter_rate_groups_from_colour,
        )
        all_rates_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            all_rates_button,
            "Fits every visible rate group from the current filters and writes summary plots/CSV files. Use this to compare rate corrections across materials and settings.",
        )
        benchmark_button = ttk.Button(
            rate_actions,
            text="Benchmark constants/models",
            command=self.benchmark_all_models_and_constants,
        )
        benchmark_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            benchmark_button,
            "Tests combinations of constants profiles and optical models against the experiment cache. This does not optimize thickness; it ranks model choices.",
        )
        roughness_button = ttk.Button(
            rate_actions,
            text="Fit roughness group",
            command=self.fit_selected_roughness_group,
        )
        roughness_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            roughness_button,
            "Fits roughness/diffuse-scatter settings for the current filtered group. Useful when reflectance shape is reasonable but intensity or colour is systematically off.",
        )
        refresh_distance_button = ttk.Button(
            review_actions,
            text="Refresh colour-distance tab",
            command=self.refresh_colour_distance_plot,
        )
        refresh_distance_button.pack(side=tk.LEFT)
        self._add_tooltip(
            refresh_distance_button,
            "Refreshes the Colour Distance tab using the current model cache and the best saved thickness-fit cache for each measurement.",
        )

        ttk.Label(
            workflow,
            text=(
                "The filters choose which sample group or composition is used for batch fits. "
                "All calculations use the current constants profile, optical model, angle, interface, native oxide, and roughness settings."
            ),
            foreground="#52606d",
            wraplength=1200,
        ).pack(anchor=tk.W, pady=(6, 0))

        self.fit_notebook = ttk.Notebook(wrapper)
        self.fit_notebook.pack(fill=tk.BOTH, expand=True)
        self.rate_groups_tab = ttk.Frame(self.fit_notebook)
        self.calibration_tab = ttk.Frame(self.fit_notebook)
        self.fit_notebook.add(self.rate_groups_tab, text="Rate groups")
        self.fit_notebook.add(self.calibration_tab, text="Model calibration")
        self._build_rate_groups_tab(self.rate_groups_tab)
        self._build_calibration_tab(self.calibration_tab)

    def _build_configuration_fit_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Best-fit search for one configuration", padding=10)
        controls.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Configuration").pack(side=tk.LEFT)
        self.configuration_fit_combo = ttk.Combobox(
            row,
            textvariable=self.configuration_fit_var,
            values=(),
            state="readonly",
            width=58,
        )
        self.configuration_fit_combo.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)
        self._add_combobox_tooltip(self.configuration_fit_combo, self.configuration_fit_var.get)
        ttk.Button(
            row,
            text="Refresh",
            command=self.refresh_configuration_fit_choices,
        ).pack(side=tk.LEFT)
        run_button = ttk.Button(
            row,
            text="Run staged fit search",
            command=self.run_configuration_fit_pipeline,
        )
        run_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            run_button,
            "For the selected configuration, tests constants profiles and optical models, optionally runs cached thickness fits, saves the result, and draws the Delta E reduction process.",
        )
        self._pack_download_figure_button(
            row,
            lambda: self.configuration_fit_figure,
            "configuration_fit",
            side=tk.LEFT,
        )

        options = ttk.Frame(controls)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="Run/reuse thickness optimization after choosing best constants/model",
            variable=self.configuration_fit_run_thickness_var,
        ).pack(side=tk.LEFT)
        ttk.Label(
            options,
            text=(
                "This is bounded: it compares saved constants profiles and optical models, "
                "then fits thicknesses only for this configuration."
            ),
            foreground="#52606d",
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(
            controls,
            textvariable=self.configuration_fit_info_var,
            foreground="#52606d",
            wraplength=1200,
        ).pack(anchor=tk.W, pady=(8, 0))

        self.configuration_fit_figure = Figure(figsize=(9.0, 6.2), dpi=120)
        self.configuration_fit_canvas = FigureCanvasTkAgg(self.configuration_fit_figure, wrapper)
        self.configuration_fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_empirical_fit_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Equipment-calibrated effective n/k fit", padding=10)
        controls.pack(fill=tk.X, pady=(0, 8))

        filter_row = ttk.Frame(controls)
        filter_row.pack(fill=tk.X, pady=(0, 6))
        self._pack_fit_filter_controls(filter_row)
        ttk.Button(filter_row, text="Load indexes", command=self.load_experiment_samples).pack(side=tk.LEFT)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text="Fit materials").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.empirical_fit_materials_var, width=30).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Checkbutton(row, text="Fit k too", variable=self.empirical_fit_k_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Checkbutton(
            row,
            text="Thickness dependence",
            variable=self.empirical_fit_thickness_dependence_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            row,
            text="Sputter-time dependence",
            variable=self.empirical_fit_time_dependence_var,
        ).pack(side=tk.LEFT, padx=(0, 12))

        numeric_row = ttk.Frame(controls)
        numeric_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(numeric_row, text="Validation fraction").pack(side=tk.LEFT)
        self._spinbox(
            numeric_row,
            self.empirical_validation_fraction_var,
            0.0,
            0.6,
            0.05,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(numeric_row, text="Colour weight").pack(side=tk.LEFT)
        self._spinbox(
            numeric_row,
            self.empirical_lab_weight_var,
            0.0,
            0.20,
            0.01,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(numeric_row, text="Max evaluations").pack(side=tk.LEFT)
        self._spinbox(
            numeric_row,
            self.empirical_max_evals_var,
            20,
            800,
            20,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        run_button = ttk.Button(
            numeric_row,
            text="Run empirical n/k fit",
            command=self.run_empirical_refractive_index_fit,
        )
        run_button.pack(side=tk.LEFT)
        self._add_tooltip(
            run_button,
            "Fits loose effective refractive-index corrections for the filtered samples. This is a predictive equipment calibration, not a physical constants fit.",
        )
        self._pack_download_figure_button(
            numeric_row,
            lambda: self.empirical_fit_figure,
            "empirical_fit",
            side=tk.LEFT,
        )

        ttk.Label(
            controls,
            text=(
                "The model is n_eff = n_base + dn0 + dn_thickness*x_thickness + dn_time*x_time "
                "and similarly for k when enabled. Bounds are intentionally loose but finite; validation samples are held out by sample name."
            ),
            foreground="#52606d",
            wraplength=1200,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(
            controls,
            textvariable=self.empirical_fit_info_var,
            foreground="#52606d",
            wraplength=1200,
        ).pack(anchor=tk.W, pady=(4, 0))

        self.empirical_fit_figure = Figure(figsize=(9.0, 6.2), dpi=120)
        self.empirical_fit_canvas = FigureCanvasTkAgg(self.empirical_fit_figure, wrapper)
        self.empirical_fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _pack_fit_filter_controls(self, parent: ttk.Frame) -> None:
        combo_set: dict[str, ttk.Combobox] = {}
        ttk.Label(parent, text="Series").pack(side=tk.LEFT)
        self.fit_series_filter_combo = ttk.Combobox(
            parent,
            textvariable=self.experiment_series_filter_var,
            values=("All",),
            state="readonly",
            width=9,
        )
        self.fit_series_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.fit_series_filter_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)
        combo_set["series"] = self.fit_series_filter_combo

        ttk.Label(parent, text="Substrate").pack(side=tk.LEFT)
        self.fit_substrate_filter_combo = ttk.Combobox(
            parent,
            textvariable=self.experiment_substrate_filter_var,
            values=("All", "Si", "Ti"),
            state="readonly",
            width=9,
        )
        self.fit_substrate_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.fit_substrate_filter_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)
        combo_set["substrate"] = self.fit_substrate_filter_combo

        ttk.Label(parent, text="Surface").pack(side=tk.LEFT)
        self.fit_surface_filter_combo = ttk.Combobox(
            parent,
            textvariable=self.experiment_surface_filter_var,
            values=("All", "smooth", "rough", "unknown"),
            state="readonly",
            width=11,
        )
        self.fit_surface_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.fit_surface_filter_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)
        combo_set["surface"] = self.fit_surface_filter_combo

        ttk.Label(parent, text="Measurement").pack(side=tk.LEFT)
        self.fit_kind_filter_combo = ttk.Combobox(
            parent,
            textvariable=self.experiment_kind_filter_var,
            values=("All", "specular", "integrating_sphere", "diffuse_sphere", "unknown"),
            state="readonly",
            width=18,
        )
        self.fit_kind_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.fit_kind_filter_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)
        combo_set["kind"] = self.fit_kind_filter_combo

        ttk.Label(parent, text="Composition").pack(side=tk.LEFT)
        self.fit_composition_combo = ttk.Combobox(
            parent,
            textvariable=self.fit_composition_filter_var,
            values=self._composition_filter_values(),
            state="readonly",
            width=20,
        )
        self.fit_composition_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.fit_composition_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)
        combo_set["composition"] = self.fit_composition_combo
        self.fit_filter_combo_sets.append(combo_set)

    @staticmethod
    def _composition_filter_values() -> tuple[str, ...]:
        return (
            "All",
            "Single layer",
            "TiO2 single layer",
            "SiO2 single layer",
            "ZrO2 single layer",
            "Ag single layer",
            "Au single layer",
            "Multilayer",
            "Ag containing",
            "TiO2/SiO2/Ag",
        )

    def _build_colour_distance_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Compare colour-distance fits", padding=8)
        controls.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        self._pack_fit_filter_controls(row)
        ttk.Label(row, text="Compare").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(
            row,
            textvariable=self.colour_distance_source_var,
            values=("Best cached thickness fit", "Raw model cache only"),
            state="readonly",
            width=24,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(row, text="Refresh", command=self.refresh_colour_distance_plot).pack(side=tk.LEFT)
        ttk.Button(row, text="Load model cache", command=self.load_experiment_cache).pack(side=tk.LEFT, padx=(6, 0))

        export_row = ttk.Frame(controls)
        export_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(export_row, text="Download").pack(side=tk.LEFT)
        combined_button = ttk.Button(
            export_row,
            text="Combined figure",
            command=lambda: self._download_figure(
                self.colour_distance_figure,
                "colour_distance_combined",
            ),
        )
        combined_button.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(
            combined_button,
            "Saves the visible Colour Distance view with both the Delta E plot and the colour strip.",
        )
        before_after_button = ttk.Button(
            export_row,
            text="Before/after only",
            command=self.download_colour_distance_before_after_figure,
        )
        before_after_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            before_after_button,
            "Saves only the Delta E before/after plot.",
        )
        colours_button = ttk.Button(
            export_row,
            text="Colours only",
            command=self.download_colour_distance_colours_figure,
        )
        colours_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            colours_button,
            "Saves only the measured, after-fit, and before-fit colour strip.",
        )

        ttk.Label(
            controls,
            text=(
                "This tab uses the experiment model cache plus any saved thickness-fit caches. "
                "It is meant for checking which samples improved, which groups still fail, and which fits are worth reusing for search."
            ),
            foreground="#52606d",
            wraplength=1160,
        ).pack(anchor=tk.W, pady=(6, 0))

        body = ttk.PanedWindow(wrapper, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(body)
        figure_frame = ttk.Frame(body)
        body.add(table_frame, weight=1)
        body.add(figure_frame, weight=3)

        columns = (
            "sample",
            "series",
            "substrate",
            "surface",
            "mode",
            "model_delta",
            "fit_delta",
            "improvement",
            "fit_model",
        )
        self.colour_distance_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
        )
        headings = {
            "sample": "Sample / measurement",
            "series": "Series",
            "substrate": "Substrate",
            "surface": "Surface",
            "mode": "Mode",
            "model_delta": "Model Delta E",
            "fit_delta": "Fit Delta E",
            "improvement": "Improvement",
            "fit_model": "Fit profile / model",
        }
        widths = {
            "sample": 380,
            "series": 70,
            "substrate": 90,
            "surface": 90,
            "mode": 135,
            "model_delta": 110,
            "fit_delta": 100,
            "improvement": 105,
            "fit_model": 260,
        }
        for column in columns:
            self.colour_distance_tree.heading(column, text=headings[column])
            self.colour_distance_tree.column(column, width=widths[column], anchor=tk.CENTER)
        self.colour_distance_tree.column("sample", anchor=tk.W)
        self.colour_distance_tree.column("fit_model", anchor=tk.W)
        distance_scroll = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.colour_distance_tree.yview,
        )
        self.colour_distance_tree.configure(yscrollcommand=distance_scroll.set)
        self.colour_distance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        distance_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.colour_distance_figure = Figure(figsize=(8.4, 5.6), dpi=120)
        self.colour_distance_canvas = FigureCanvasTkAgg(self.colour_distance_figure, figure_frame)
        self.colour_distance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_plots_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Measured samples on sweep maps", padding=8)
        controls.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Map").pack(side=tk.LEFT)
        self.plots_map_combo = ttk.Combobox(
            row,
            textvariable=self.plots_map_var,
            values=(),
            state="readonly",
            width=34,
        )
        self.plots_map_combo.pack(side=tk.LEFT, padx=(8, 14))
        self.plots_map_combo.bind("<<ComboboxSelected>>", lambda *_args: self.draw_selected_experiment_plot_map())

        ttk.Label(row, text="Thickness").pack(side=tk.LEFT)
        state_combo = ttk.Combobox(
            row,
            textvariable=self.plots_fit_state_var,
            values=(
                "Before thickness fit",
                "After best cached thickness fit",
                "After individual thickness fit",
                "After same-material thickness fit",
            ),
            state="readonly",
            width=30,
        )
        state_combo.pack(side=tk.LEFT, padx=(8, 14))
        state_combo.bind("<<ComboboxSelected>>", lambda *_args: self.draw_selected_experiment_plot_map())

        ttk.Button(row, text="Refresh choices", command=self.refresh_plots_map_choices).pack(side=tk.LEFT)
        ttk.Button(row, text="Draw map", command=self.draw_selected_experiment_plot_map).pack(side=tk.LEFT, padx=(6, 0))

        diagnostic_row = ttk.Frame(controls)
        diagnostic_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(diagnostic_row, text="Diagnostics").pack(side=tk.LEFT)
        ttk.Button(
            diagnostic_row,
            text="Colour distances",
            command=self.draw_experiment_plot_colour_distances,
        ).pack(side=tk.LEFT, padx=(10, 0))
        fit_impact_button = ttk.Button(
            diagnostic_row,
            text="Fit impact",
            command=self.draw_fit_impact_workflow_plot,
        )
        fit_impact_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            fit_impact_button,
            "Builds a dashboard from saved caches to show which fit family lowered Delta E most: thickness, constants/model, refractive index, roughness, or empirical calibration.",
        )
        adjustment_button = ttk.Button(
            diagnostic_row,
            text="Thickness adjustments",
            command=self.draw_individual_thickness_adjustment_plot,
        )
        adjustment_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            adjustment_button,
            "Shows how individual thickness fits usually change each layer as a function of the starting thickness estimate.",
        )

        map_export_row = ttk.Frame(controls)
        map_export_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(map_export_row, text="Download map").pack(side=tk.LEFT)
        full_button = ttk.Button(
            map_export_row,
            text="Full with samples",
            command=lambda: self._download_figure(self.plots_figure, "plots_full"),
        )
        full_button.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(
            full_button,
            "Saves the full Plots figure, including the Configuration and samples panel when it is visible.",
        )
        clean_button = ttk.Button(
            map_export_row,
            text="Clean plot only",
            command=lambda: self._download_primary_axis_figure(self.plots_figure, "plots_clean"),
        )
        clean_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            clean_button,
            "Saves only the main plot area, without the Configuration and samples panel.",
        )

        sensitivity_row = ttk.Frame(controls)
        sensitivity_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sensitivity_row, text="Colour change from current stack").pack(side=tk.LEFT)
        one_layer_button = ttk.Button(
            sensitivity_row,
            text="1-layer Delta E",
            command=self.draw_1d_thickness_colour_difference,
        )
        one_layer_button.pack(side=tk.LEFT, padx=(10, 0))
        self._add_tooltip(
            one_layer_button,
            "Plots how much the predicted colour changes when Layer 1 thickness is swept. Delta E is measured relative to the current stack colour.",
        )
        two_layer_button = ttk.Button(
            sensitivity_row,
            text="2-layer Delta E map",
            command=self.draw_2d_thickness_colour_difference,
        )
        two_layer_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            two_layer_button,
            "Plots a heatmap of colour difference while Layer 1 and Layer 2 thicknesses are swept together. Delta E is relative to the current stack colour.",
        )
        ttk.Label(
            sensitivity_row,
            text="Uses the Sweep layer/range/point controls on the left.",
            foreground="#52606d",
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(
            controls,
            textvariable=self.plots_info_var,
            foreground="#52606d",
            wraplength=1160,
        ).pack(anchor=tk.W, pady=(6, 0))

        self.plots_figure = Figure(figsize=(8.6, 6.0), dpi=120)
        self.plots_canvas = FigureCanvasTkAgg(self.plots_figure, wrapper)
        self.plots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _select_fit_optimize_section(self, section: str = "Rate groups") -> None:
        self.notebook.select(self.fit_optimize_tab)
        if hasattr(self, "fit_notebook"):
            if section.lower().startswith("model"):
                self.fit_notebook.select(self.calibration_tab)
            else:
                self.fit_notebook.select(self.rate_groups_tab)

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Colour target search", padding=8)
        controls.pack(fill=tk.X, pady=(0, 8))

        target_row = ttk.Frame(controls)
        target_row.pack(fill=tk.X, pady=2)
        ttk.Label(target_row, text="Target").pack(side=tk.LEFT)
        ttk.Combobox(
            target_row,
            textvariable=self.search_target_mode_var,
            values=("D65 white", "2016 WID white", "sRGB hex", "CIELAB"),
            state="readonly",
            width=18,
        ).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(target_row, text="Hex").pack(side=tk.LEFT)
        ttk.Entry(target_row, textvariable=self.search_target_hex_var, width=10).pack(side=tk.LEFT, padx=(4, 16))
        for label, variable in (
            ("L*", self.search_target_l_var),
            ("a*", self.search_target_a_var),
            ("b*", self.search_target_b_var),
        ):
            ttk.Label(target_row, text=label).pack(side=tk.LEFT)
            self._spinbox(target_row, variable, -150.0, 150.0, 1.0, None).pack(side=tk.LEFT, padx=(4, 10))

        search_row = ttk.Frame(controls)
        search_row.pack(fill=tk.X, pady=2)
        for label, variable, low, high, step in (
            ("Min nm", self.search_min_nm_var, 0.0, 1000.0, 1.0),
            ("Max nm", self.search_max_nm_var, 0.0, 1000.0, 1.0),
            ("Points/layer", self.search_points_var, 5, 201, 2),
            ("Iterations", self.search_iterations_var, 1, 20, 1),
            ("Min L*", self.search_min_lightness_var, 0.0, 100.0, 1.0),
            ("Brightness weight", self.search_brightness_weight_var, 0.0, 5.0, 0.05),
        ):
            ttk.Label(search_row, text=label).pack(side=tk.LEFT)
            self._spinbox(search_row, variable, low, high, step, None).pack(side=tk.LEFT, padx=(4, 12))

        strategy_row = ttk.Frame(controls)
        strategy_row.pack(fill=tk.X, pady=2)
        ttk.Label(strategy_row, text="Search strategy").pack(side=tk.LEFT)
        ttk.Combobox(
            strategy_row,
            textvariable=self.search_strategy_var,
            values=("coordinate fast", "full grid"),
            state="readonly",
            width=18,
        ).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(
            strategy_row,
            text="Full grid tests points/layer^selected layers; coordinate fast tests far fewer candidates.",
            foreground="#52606d",
        ).pack(side=tk.LEFT)

        layer_row = ttk.Frame(controls)
        layer_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(layer_row, text="Layers to vary").pack(side=tk.LEFT)
        self.search_layer_listbox = tk.Listbox(
            layer_row,
            selectmode=tk.MULTIPLE,
            height=4,
            exportselection=False,
            font=("Segoe UI", 10),
        )
        self.search_layer_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        ttk.Button(layer_row, text="Refresh layers", command=self._refresh_search_layers).pack(side=tk.LEFT)
        ttk.Button(layer_row, text="Run search", command=self.run_colour_target_search).pack(side=tk.LEFT, padx=(8, 0))
        self._pack_download_figure_button(
            layer_row,
            lambda: self.search_figure,
            "search",
            side=tk.LEFT,
        )

        ttk.Label(
            controls,
            text=(
                "Uses the current stack, constants, optical model, angle, interface, oxide, and roughness settings. "
                "The score is the selected colour metric plus a lightness penalty, so bright near-white samples rank above dull grey."
            ),
            foreground="#52606d",
            wraplength=1100,
        ).pack(anchor=tk.W, pady=(6, 0))

        matches_frame = ttk.LabelFrame(wrapper, text="Closest measured samples to search colour", padding=6)
        matches_frame.pack(fill=tk.X, pady=(0, 8))
        match_columns = ("configuration", "count", "closest", "delta", "lstar", "wid", "colour")
        self.search_measured_matches_tree = ttk.Treeview(
            matches_frame,
            columns=match_columns,
            show="headings",
            height=5,
        )
        match_headings = {
            "configuration": "Configuration",
            "count": "Rows",
            "closest": "Closest samples",
            "delta": self._delta_e_label(),
            "lstar": "L*",
            "wid": "WID",
            "colour": "Colour",
        }
        match_widths = {
            "configuration": 260,
            "count": 54,
            "closest": 360,
            "delta": 78,
            "lstar": 62,
            "wid": 70,
            "colour": 82,
        }
        for column in match_columns:
            self.search_measured_matches_tree.heading(column, text=match_headings[column])
            self.search_measured_matches_tree.column(
                column,
                width=match_widths[column],
                anchor=tk.CENTER if column != "configuration" and column != "closest" else tk.W,
            )
        matches_scroll = ttk.Scrollbar(
            matches_frame,
            orient=tk.VERTICAL,
            command=self.search_measured_matches_tree.yview,
        )
        self.search_measured_matches_tree.configure(yscrollcommand=matches_scroll.set)
        self.search_measured_matches_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        matches_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.search_figure = Figure(figsize=(8.4, 5.8), dpi=170)
        self.search_canvas = FigureCanvasTkAgg(self.search_figure, wrapper)
        self.search_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._refresh_search_layers()

    def _on_main_tab_changed(self, *_args) -> None:
        self._update_left_panel_for_tab()
        active = self.notebook.tab(self.notebook.select(), "text")
        if active == "Search":
            self._refresh_search_layers()
        elif active == "Colour Distance":
            self.refresh_colour_distance_plot(redraw_only=True)
        elif active == "Plots":
            self.refresh_plots_map_choices(redraw_only=True)
        elif active == "Configuration Fit":
            self.refresh_configuration_fit_choices()
        elif active == "Samples":
            self.refresh_samples_overview(redraw_only=True)

    def _update_left_panel_for_tab(self) -> None:
        """Show only the left-side controls relevant to the active main tab."""

        boxes = (
            getattr(self, "stack_controls_box", None),
            getattr(self, "sweep_controls_box", None),
            getattr(self, "constants_candidate_box", None),
            getattr(self, "experiment_settings_box", None),
        )
        for box in boxes:
            if box is not None:
                box.pack_forget()

        if not hasattr(self, "notebook"):
            active = "Reflectance"
        else:
            try:
                active = self.notebook.tab(self.notebook.select(), "text")
            except tk.TclError:
                active = "Reflectance"

        if active == "Reflectance":
            self.stack_controls_box.pack(fill=tk.X, pady=(0, 8))
            self.sweep_controls_box.pack(fill=tk.X)
        elif active == "Sweep":
            self.stack_controls_box.pack(fill=tk.X, pady=(0, 8))
            self.sweep_controls_box.pack(fill=tk.X)
        elif active == "Search":
            self.stack_controls_box.pack(fill=tk.X, pady=(0, 8))
        elif active == "Constants":
            self.constants_candidate_box.pack(fill=tk.X, pady=(0, 8))
        elif active == "Experiments":
            self.experiment_settings_box.pack(fill=tk.X, pady=(0, 8))
        elif active == "Fit & Optimize":
            self.experiment_settings_box.pack(fill=tk.X, pady=(0, 8))
        elif active == "Colour Distance":
            self.experiment_settings_box.pack(fill=tk.X, pady=(0, 8))
        elif active == "Plots":
            self.stack_controls_box.pack(fill=tk.X, pady=(0, 8))
            self.sweep_controls_box.pack(fill=tk.X)
        self.root.after_idle(self._refresh_left_controls_scroller)

    def _refresh_search_layers(self) -> None:
        if not hasattr(self, "search_layer_listbox"):
            return
        previous = {
            self.search_layer_listbox.get(index)
            for index in self.search_layer_listbox.curselection()
        }
        self.search_layer_listbox.delete(0, tk.END)
        for index, row in enumerate(self.layer_rows, start=1):
            thickness = self._try_float_variable(row.thickness_var)
            thickness_label = "..." if thickness is None else f"{thickness:g}"
            label = f"{row.material_var.get()} #{index} ({thickness_label} nm)"
            self.search_layer_listbox.insert(tk.END, label)
            if not previous or label in previous:
                self.search_layer_listbox.selection_set(tk.END)

    def run_colour_target_search(self) -> None:
        try:
            stack = self._build_stack_from_controls()
            selected = list(self.search_layer_listbox.curselection())
            if not selected:
                selected = list(range(len(self.layer_rows)))
            if not selected:
                raise ValueError("Add at least one deposited layer before running a colour search.")

            min_nm = float(self.search_min_nm_var.get())
            max_nm = float(self.search_max_nm_var.get())
            layer_bounds = [
                (index, self.layer_rows[index].material_var.get(), min_nm, max_nm)
                for index in selected
            ]
            model = self._model_from_controls()
            wavelengths_nm = wavelength_grid(400.0, 700.0, 61)
            angle_deg = float(self.angle_var.get())
            target_xyz = self._search_target_xyz()
            score_mode = "wid_2016" if self.search_target_mode_var.get() == "2016 WID white" else "delta_e"
            iterations = int(self.search_iterations_var.get())
            points_per_layer = int(self.search_points_var.get())
            min_lightness = float(self.search_min_lightness_var.get())
            brightness_weight = float(self.search_brightness_weight_var.get())
            strategy_label = self.search_strategy_var.get()
            strategy = "full_grid" if strategy_label == "full grid" else "coordinate"
            colour_metric = self._current_colour_metric()
            if strategy == "full_grid":
                total = max(points_per_layer, 2) ** max(len(layer_bounds), 1)
            else:
                total = 1 + max(iterations, 1) * max(len(layer_bounds), 1) * max(points_per_layer, 2)

            def task(progress):
                def search_progress(done: int, total_count: int, message: str) -> None:
                    self._wait_if_paused(progress)
                    progress(done, message)

                return search_thicknesses_for_target_colour(
                    stack=stack,
                    model=model,
                    wavelengths_nm=wavelengths_nm,
                    angle_deg=angle_deg,
                    layer_bounds_nm=layer_bounds,
                    target_xyz=target_xyz,
                    iterations=iterations,
                    points_per_layer=points_per_layer,
                    min_lightness=min_lightness,
                    brightness_weight=brightness_weight,
                    strategy=strategy,
                    colour_metric=colour_metric,
                    score_mode=score_mode,
                    progress_callback=search_progress,
                )

            def on_success(result: TargetSearchResult) -> str:
                measured_matches = self._closest_measured_search_matches(
                    result.target_xyz,
                    result.colour_metric,
                )
                self._populate_search_measured_matches_tree(measured_matches)
                self._draw_colour_target_search_result(result, measured_matches)
                self.notebook.select(self.search_tab)
                best = result.best
                score_text = (
                    f"WID {best.whiteness_index:.2f}"
                    if result.score_mode == "wid_2016"
                    else f"score {best.score:.2f}"
                )
                return (
                    f"Search complete: best {colour_metric_label(result.colour_metric)} {best.delta_e:.2f}, "
                    f"L* {best.lab[0]:.1f}, {score_text}."
                )

            self._run_background(
                task,
                on_success,
                title="Colour target search",
                busy_message="searching bright target colour",
                progress_max=total,
            )
        except Exception as exc:
            messagebox.showerror("Colour target search", str(exc))

    def _search_target_xyz(self) -> tuple[float, float, float]:
        mode = self.search_target_mode_var.get()
        if mode in {"D65 white", "2016 WID white"}:
            return (95.047, 100.0, 108.883)
        if mode == "CIELAB":
            return xyz_from_lab(
                (
                    float(self.search_target_l_var.get()),
                    float(self.search_target_a_var.get()),
                    float(self.search_target_b_var.get()),
                )
            )
        return xyz_from_srgb(self._parse_hex_colour(self.search_target_hex_var.get()))

    @staticmethod
    def _parse_hex_colour(value: str) -> tuple[float, float, float]:
        text = str(value).strip().lstrip("#")
        if len(text) != 6:
            raise ValueError("Hex target colour must look like #ffffff.")
        try:
            channels = [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
        except ValueError as exc:
            raise ValueError("Hex target colour must contain only 0-9 and a-f.") from exc
        return tuple(float(channel) for channel in channels)

    def _closest_measured_search_matches(
        self,
        target_xyz: tuple[float, float, float],
        colour_metric: str,
        *,
        per_configuration: int = 3,
    ) -> list[dict[str, object]]:
        try:
            if self.experiment_store is None:
                self.load_experiment_samples(show_errors=False)
            if self.experiment_cache is None:
                self.load_experiment_cache()
        except Exception:
            return []
        if self.experiment_store is None or self.experiment_cache is None:
            return []

        metric = normalise_colour_metric(colour_metric)
        groups: dict[tuple[tuple[str, ...], str, str], dict[str, object]] = {}
        sample_cache: dict[str, object] = {}
        for index in range(self.experiment_cache.count):
            sample_name = str(self.experiment_cache.sample_names[index])
            measurement = str(self.experiment_cache.measurement_descriptions[index])
            try:
                sample = sample_cache.get(sample_name)
                if sample is None:
                    sample = self.experiment_store.load_sample(sample_name)
                    sample_cache[sample_name] = sample
            except Exception:
                continue
            materials = tuple(layer.material_name for layer in sample.layer_estimates)
            if not materials:
                continue
            substrate = str(self.experiment_cache.substrate_classes[index])
            surface = str(self.experiment_cache.surface_classes[index])
            group_key = (materials, substrate, surface)
            group = groups.setdefault(
                group_key,
                {
                    "configuration": self._search_measured_configuration_label(materials, substrate, surface),
                    "materials": materials,
                    "substrate": substrate,
                    "surface": surface,
                    "count": 0,
                    "matches": [],
                },
            )
            group["count"] = int(group["count"]) + 1
            measured_xyz = tuple(float(value) for value in self.experiment_cache.measured_xyz[index])
            measured_rgb = np.clip(np.asarray(self.experiment_cache.measured_rgb[index], dtype=float), 0.0, 1.0)
            lab = tuple(float(value) for value in xyz_to_lab(measured_xyz))
            delta = delta_e_colour(target_xyz, measured_xyz, metric=metric)
            match = {
                "sample_name": sample_name,
                "measurement": measurement,
                "delta": float(delta),
                "lab": lab,
                "wid": wid2016_from_lab(lab),
                "rgb": measured_rgb,
                "hex": self._rgb_tuple_to_hex(measured_rgb),
                "stack_label": str(self.experiment_cache.stack_labels[index]),
            }
            matches = group["matches"]
            if isinstance(matches, list):
                matches.append(match)

        grouped_rows: list[dict[str, object]] = []
        for group in groups.values():
            matches = group.get("matches", [])
            if not isinstance(matches, list) or not matches:
                continue
            ordered_matches = sorted(matches, key=lambda item: float(item["delta"]))
            group["matches"] = ordered_matches[:per_configuration]
            group["best_delta"] = float(ordered_matches[0]["delta"])
            group["best_lab"] = ordered_matches[0]["lab"]
            group["best_wid"] = float(ordered_matches[0]["wid"])
            group["best_rgb"] = ordered_matches[0]["rgb"]
            group["best_hex"] = str(ordered_matches[0]["hex"])
            grouped_rows.append(group)
        return sorted(
            grouped_rows,
            key=lambda group: (
                float(group["best_delta"]),
                str(group["configuration"]),
            ),
        )

    @staticmethod
    def _search_measured_configuration_label(
        materials: tuple[str, ...],
        substrate: str,
        surface: str,
    ) -> str:
        material_text = " / ".join(materials)
        location_parts = [
            str(surface or "").strip(),
            str(substrate or "").strip(),
        ]
        location = " ".join(part for part in location_parts if part).strip()
        return f"{material_text} on {location}" if location else material_text

    def _populate_search_measured_matches_tree(
        self,
        grouped_rows: list[dict[str, object]],
    ) -> None:
        if not hasattr(self, "search_measured_matches_tree"):
            return
        self.search_measured_matches_tree.heading("delta", text=self._delta_e_label())
        for item in self.search_measured_matches_tree.get_children():
            self.search_measured_matches_tree.delete(item)
        if not grouped_rows:
            self.search_measured_matches_tree.insert(
                "",
                tk.END,
                values=(
                    "No experiment cache loaded",
                    "",
                    "Build/load experiment results to compare the target with measured samples.",
                    "",
                    "",
                    "",
                    "",
                ),
            )
            return
        for row in grouped_rows:
            matches = row.get("matches", [])
            closest = ""
            if isinstance(matches, list):
                closest = "; ".join(
                    f"{match['sample_name']} ({float(match['delta']):.2f})"
                    for match in matches[:3]
                )
            best_lab = tuple(float(value) for value in row.get("best_lab", (float("nan"),) * 3))
            self.search_measured_matches_tree.insert(
                "",
                tk.END,
                values=(
                    str(row["configuration"]),
                    int(row["count"]),
                    closest,
                    f"{float(row['best_delta']):.2f}",
                    f"{best_lab[0]:.1f}",
                    f"{float(row['best_wid']):.1f}",
                    str(row["best_hex"]),
                ),
            )

    def _draw_colour_target_search_result(
        self,
        result: TargetSearchResult,
        measured_matches: list[dict[str, object]] | None = None,
    ) -> None:
        self.search_figure.clear()
        grid = self.search_figure.add_gridspec(
            2,
            2,
            height_ratios=[0.76, 1.45],
            width_ratios=[1.0, 1.42],
            hspace=0.35,
            wspace=0.28,
        )
        swatch_ax = self.search_figure.add_subplot(grid[0, 0])
        spectra_ax = self.search_figure.add_subplot(grid[1, 0])
        table_ax = self.search_figure.add_subplot(grid[0, 1])
        measured_ax = self.search_figure.add_subplot(grid[1, 1])
        table_ax.axis("off")
        measured_ax.axis("off")

        best = result.best
        delta_label = "Delta E00" if normalise_colour_metric(result.colour_metric) == COLOUR_METRIC_CIEDE2000 else "Delta E*"
        score_is_wid = result.score_mode == "wid_2016"
        score_label = "WID 2016" if score_is_wid else "Score"
        target_rgb = xyz_to_srgb(result.target_xyz)
        swatches = np.array([[target_rgb, best.srgb]], dtype=float)
        swatch_ax.imshow(swatches, aspect="auto", interpolation="nearest")
        swatch_ax.set_xticks([0, 1])
        swatch_ax.set_xticklabels(["Target", "Best"])
        swatch_ax.set_yticks([])
        swatch_ax.set_title(
            (
                f"Best colour: WID {best.whiteness_index:.2f}, "
                f"{delta_label} {best.delta_e:.2f}, L* {best.lab[0]:.1f}"
            )
            if score_is_wid
            else f"Best colour: {delta_label} {best.delta_e:.2f}, L* {best.lab[0]:.1f}, score {best.score:.2f}",
            fontweight="semibold",
        )
        for spine in swatch_ax.spines.values():
            spine.set_visible(False)

        spectra_ax.plot(
            result.wavelengths_nm,
            best.reflectance,
            color="#111827",
            linewidth=1.8,
            label="Best candidate",
        )
        for candidate in result.candidates[1:6]:
            spectra_ax.plot(
                result.wavelengths_nm,
                candidate.reflectance,
                color=candidate.srgb,
                linewidth=1.0,
                alpha=0.45,
            )
        spectra_ax.set_xlabel("Wavelength (nm)")
        spectra_ax.set_ylabel("Reflectance")
        spectra_ax.set_ylim(0.0, 1.02)
        spectra_ax.grid(True, color="#bcccdc", alpha=0.35)
        spectra_ax.legend(loc="best", fontsize=8)

        rows = []
        display_candidates = result.candidates[:3]
        for rank, candidate in enumerate(display_candidates, start=1):
            thickness_text = "\n".join(
                f"{name} {value:.1f} nm"
                for name, value in zip(candidate.layer_names, candidate.thicknesses_nm)
            )
            rows.append(
                [
                    rank,
                    f"{candidate.delta_e:.2f}",
                    f"{candidate.lab[0]:.1f}",
                    f"{candidate.whiteness_index:.2f}" if score_is_wid else f"{candidate.score:.2f}",
                    thickness_text,
                    "",
                ]
            )
        table = table_ax.table(
            cellText=rows,
            colLabels=("Rank", delta_label, "L*", score_label, "Thicknesses", "Colour"),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.55)
        for cell_key, cell in table.get_celld().items():
            cell.set_linewidth(0.7)
            cell.get_text().set_wrap(True)
        column_widths = {0: 0.08, 1: 0.11, 2: 0.10, 3: 0.11, 4: 0.42, 5: 0.12}
        for (row_index, column_index), cell in table.get_celld().items():
            if column_index in column_widths:
                cell.set_width(column_widths[column_index])
        for row_index, candidate in enumerate(display_candidates, start=1):
            table[(row_index, 5)].set_facecolor(candidate.srgb)
            table[(row_index, 5)].get_text().set_text("")
        table_ax.set_title("Best thickness candidates", fontweight="semibold", pad=8)

        measured_rows = []
        measured_display = (measured_matches or [])[:8]
        for rank, group in enumerate(measured_display, start=1):
            matches = group.get("matches", [])
            closest_text = ""
            if isinstance(matches, list):
                closest_text = "\n".join(
                    f"{match['sample_name']} ({float(match['delta']):.1f})"
                    for match in matches[:2]
                )
            measured_rows.append(
                [
                    rank,
                    textwrap.shorten(str(group["configuration"]), width=30, placeholder="..."),
                    closest_text,
                    f"{float(group['best_delta']):.2f}",
                    f"{float(group['best_wid']):.1f}",
                    "",
                ]
            )
        if measured_rows:
            measured_table = measured_ax.table(
                cellText=measured_rows,
                colLabels=("Rank", "Configuration", "Closest samples", delta_label, "WID", "Colour"),
                loc="center",
                cellLoc="center",
                colLoc="center",
            )
            measured_table.auto_set_font_size(False)
            measured_table.set_fontsize(6.2)
            measured_table.scale(1.0, 1.72)
            for cell in measured_table.get_celld().values():
                cell.set_linewidth(0.65)
                cell.get_text().set_wrap(True)
            measured_widths = {0: 0.07, 1: 0.30, 2: 0.28, 3: 0.11, 4: 0.09, 5: 0.10}
            for (row_index, column_index), cell in measured_table.get_celld().items():
                if column_index in measured_widths:
                    cell.set_width(measured_widths[column_index])
            for row_index, group in enumerate(measured_display, start=1):
                measured_table[(row_index, 5)].set_facecolor(group["best_rgb"])
                measured_table[(row_index, 5)].get_text().set_text("")
            measured_ax.set_title(
                f"Closest measured samples by configuration ({len(measured_matches or [])} groups)",
                fontweight="semibold",
                pad=7,
            )
        else:
            measured_ax.text(
                0.5,
                0.5,
                "No measured experiment cache available.\nBuild or load experiment results to list closest real samples.",
                ha="center",
                va="center",
                fontsize=8,
                color="#52606d",
            )
            measured_ax.set_title("Closest measured samples by configuration", fontweight="semibold", pad=7)

        self.search_figure.suptitle(
            f"Target search | {result.stack_label} | evaluated {result.evaluated_count:,} candidates",
            fontsize=11,
            fontweight="semibold",
        )
        self.search_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90)
        self.search_canvas.draw_idle()

    def _build_constants_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(wrapper)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Material").pack(side=tk.LEFT)
        self.constants_material_combo = ttk.Combobox(
            top,
            textvariable=self.constants_material_var,
            values=self._material_names(),
            state="readonly",
            width=18,
        )
        self.constants_material_combo.pack(side=tk.LEFT, padx=(8, 12))
        self.constants_material_combo.bind("<<ComboboxSelected>>", self._on_constants_material_selected)
        ttk.Button(top, text="Load current values", command=self.load_constants_editor).pack(
            side=tk.LEFT
        )
        ttk.Button(top, text="Apply table to material", command=self.apply_constants_table).pack(
            side=tk.LEFT, padx=6
        )
        single_film_fit_button = ttk.Button(
            top,
            text="Fit from single films",
            command=self.fit_constants_from_single_films,
        )
        single_film_fit_button.pack(side=tk.LEFT)
        self._add_tooltip(
            single_film_fit_button,
            "Fits simple wavelength-independent n/k constants from single-film calibration samples. This updates the fitted_single_films constants profile.",
        )
        candidate_fit_button = ttk.Button(
            top,
            text="Fit selected candidate groups",
            command=self.fit_refractiveindex_candidate_constants,
        )
        candidate_fit_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            candidate_fit_button,
            "Tests configured refractiveindex.info material datasets against the selected experiment groups and saves the best candidate profile.",
        )

        source_row = ttk.Frame(wrapper)
        source_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(source_row, text="Source for selected material").pack(side=tk.LEFT)
        self.constants_source_combo = ttk.Combobox(
            source_row,
            textvariable=self.constants_source_var,
            values=(),
            state="readonly",
            width=42,
        )
        self.constants_source_combo.pack(side=tk.LEFT, padx=(8, 6))
        self._add_combobox_tooltip(self.constants_source_combo, self.constants_source_var.get)
        ttk.Button(
            source_row,
            text="Use source",
            command=self.apply_material_source,
        ).pack(side=tk.LEFT)

        url_box = ttk.LabelFrame(wrapper, text="Import from refractiveindex.info YAML", padding=8)
        url_box.pack(fill=tk.X, pady=(12, 8))
        url_row = ttk.Frame(url_box)
        url_row.pack(fill=tk.X)
        ttk.Label(url_row, text="YAML URL").pack(side=tk.LEFT)
        ttk.Entry(url_row, textvariable=self.constants_url_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(url_row, text="Import", command=self.import_constants_from_url).pack(side=tk.RIGHT)
        ttk.Label(
            url_box,
            text=(
                "Use a raw YAML file from the refractiveindex.info database. "
                "Tabulated n or nk data are supported."
            ),
            foreground="#52606d",
        ).pack(anchor=tk.W, pady=(4, 0))

        table_box = ttk.LabelFrame(wrapper, text="Visible-spectrum n/k table", padding=8)
        table_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            table_box,
            text="Enter one row per wavelength: wavelength_nm, n, k",
            foreground="#52606d",
        ).pack(anchor=tk.W)
        self.constants_text = tk.Text(table_box, height=14, wrap="none", font=("Consolas", 11))
        self.constants_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.load_constants_editor()

    def _build_samples_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(wrapper)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(controls, text="Sample data").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.experiment_data_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(controls, text="Browse", command=self.browse_experiment_folder).pack(side=tk.LEFT)
        ttk.Button(controls, text="Load indexes", command=self.load_experiment_samples).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(controls, text="Refresh", command=self.refresh_samples_overview).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        ttk.Label(
            wrapper,
            textvariable=self.samples_overview_info_var,
            foreground="#52606d",
            wraplength=1180,
        ).pack(anchor=tk.W, pady=(0, 8))

        sweep_box = ttk.LabelFrame(wrapper, text="Selected sample sweeps", padding=8)
        sweep_box.pack(fill=tk.X, pady=(0, 8))
        sweep_buttons = ttk.Frame(sweep_box)
        sweep_buttons.pack(fill=tk.X)
        layer_sweeps_button = ttk.Button(
            sweep_buttons,
            text="Thickness sweep each layer",
            command=self.run_selected_sample_layer_sweeps,
        )
        layer_sweeps_button.pack(side=tk.LEFT)
        self._add_tooltip(
            layer_sweeps_button,
            "Runs one 1D colour-vs-thickness sweep for every deposited layer in the selected sample. Other layers stay fixed at the estimated sample thicknesses.",
        )
        angle_sweep_button = ttk.Button(
            sweep_buttons,
            text="Angle sweep",
            command=self.run_selected_sample_angle_sweep,
        )
        angle_sweep_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            angle_sweep_button,
            "Runs a colour-vs-angle sweep for the selected sample and selected measurement configuration.",
        )
        all_sweeps_button = ttk.Button(
            sweep_buttons,
            text="Layer + angle sweeps",
            command=self.run_selected_sample_all_sweeps,
        )
        all_sweeps_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            all_sweeps_button,
            "Runs all layer thickness sweeps and the angle sweep for the selected sample configuration.",
        )
        ttk.Label(
            sweep_box,
            textvariable=self.sample_sweep_info_var,
            foreground="#52606d",
            wraplength=1180,
        ).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(
            sweep_box,
            text=(
                "Uses the Sweep min/max/points/quality and angle min/max/points controls on the left. "
                "If the measurement has no explicit angle, thickness sweeps use the current Reflectance angle."
            ),
            foreground="#52606d",
            wraplength=1180,
        ).pack(anchor=tk.W, pady=(2, 0))

        content = ttk.PanedWindow(wrapper, orient=tk.VERTICAL)
        content.pack(fill=tk.BOTH, expand=True)

        sample_box = ttk.LabelFrame(content, text="All sample names", padding=8)
        detail_box = ttk.LabelFrame(content, text="Measurements for selected sample", padding=8)
        content.add(sample_box, weight=3)
        content.add(detail_box, weight=2)

        columns = (
            "sample",
            "series",
            "layers",
            "stack",
            "measurement_rows",
            "loaded_spectra",
            "condition_count",
            "surfaces",
            "substrates",
            "kinds",
            "indexed_spectra",
            "indexed_colors",
        )
        self.samples_tree = ttk.Treeview(
            sample_box,
            columns=columns,
            show="headings",
            height=14,
            selectmode="browse",
        )
        sample_headings = {
            "sample": "Sample",
            "series": "Series",
            "layers": "Layers",
            "stack": "Estimated stack",
            "measurement_rows": "Measurement rows",
            "loaded_spectra": "Loaded spectra",
            "condition_count": "Condition variants",
            "surfaces": "Surfaces",
            "substrates": "Substrates",
            "kinds": "Measurement kinds",
            "indexed_spectra": "Indexed spectra",
            "indexed_colors": "Indexed colours",
        }
        sample_widths = {
            "sample": 90,
            "series": 70,
            "layers": 70,
            "stack": 330,
            "measurement_rows": 125,
            "loaded_spectra": 110,
            "condition_count": 145,
            "surfaces": 150,
            "substrates": 190,
            "kinds": 210,
            "indexed_spectra": 115,
            "indexed_colors": 115,
        }
        for column in columns:
            self.samples_tree.heading(column, text=sample_headings[column])
            self.samples_tree.column(column, width=sample_widths[column], anchor=tk.W)
        self.samples_tree.column("series", anchor=tk.CENTER)
        self.samples_tree.column("layers", anchor=tk.CENTER)
        self.samples_tree.column("measurement_rows", anchor=tk.CENTER)
        self.samples_tree.column("loaded_spectra", anchor=tk.CENTER)
        self.samples_tree.column("condition_count", anchor=tk.CENTER)
        self.samples_tree.column("indexed_spectra", anchor=tk.CENTER)
        self.samples_tree.column("indexed_colors", anchor=tk.CENTER)
        self.samples_tree.tag_configure("multiple_variants", background="#edf7fb")
        self.samples_tree.tag_configure("no_loaded_spectra", foreground="#7b8794")

        sample_box.columnconfigure(0, weight=1)
        sample_box.rowconfigure(0, weight=1)
        sample_y = ttk.Scrollbar(sample_box, orient=tk.VERTICAL, command=self.samples_tree.yview)
        sample_x = ttk.Scrollbar(sample_box, orient=tk.HORIZONTAL, command=self.samples_tree.xview)
        self.samples_tree.configure(yscrollcommand=sample_y.set, xscrollcommand=sample_x.set)
        self.samples_tree.grid(row=0, column=0, sticky="nsew")
        sample_y.grid(row=0, column=1, sticky="ns")
        sample_x.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            sample_box,
            text=(
                "Blue rows have more than one condition variant under the same sample name. "
                "Grey text means no loaded spectra are currently indexed for that sample."
            ),
            foreground="#52606d",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.samples_tree.bind("<<TreeviewSelect>>", self._on_samples_tree_selected)
        self._add_tooltip(
            self.samples_tree,
            "Each row is one sample name. Condition variants count unique surface / substrate / measurement-kind groups under that name.",
        )
        self.samples_overview_rows: dict[str, dict[str, object]] = {}

        ttk.Label(
            detail_box,
            textvariable=self.sample_measurements_info_var,
            foreground="#52606d",
            wraplength=1180,
        ).pack(anchor=tk.W, pady=(0, 6))

        detail_frame = ttk.Frame(detail_box)
        detail_frame.pack(fill=tk.BOTH, expand=True)
        detail_columns = (
            "condition",
            "surface",
            "substrate",
            "kind",
            "source",
            "instrument_sample",
            "description",
            "csv",
        )
        self.sample_measurements_tree = ttk.Treeview(
            detail_frame,
            columns=detail_columns,
            show="headings",
            height=9,
            selectmode="browse",
        )
        detail_headings = {
            "condition": "#",
            "surface": "Surface",
            "substrate": "Substrate",
            "kind": "Kind",
            "source": "Source",
            "instrument_sample": "Instrument sample",
            "description": "Description",
            "csv": "CSV",
        }
        detail_widths = {
            "condition": 55,
            "surface": 90,
            "substrate": 150,
            "kind": 140,
            "source": 170,
            "instrument_sample": 130,
            "description": 420,
            "csv": 420,
        }
        for column in detail_columns:
            self.sample_measurements_tree.heading(column, text=detail_headings[column])
            self.sample_measurements_tree.column(column, width=detail_widths[column], anchor=tk.W)
        self.sample_measurements_tree.column("condition", anchor=tk.CENTER)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        detail_y = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.sample_measurements_tree.yview)
        detail_x = ttk.Scrollbar(detail_frame, orient=tk.HORIZONTAL, command=self.sample_measurements_tree.xview)
        self.sample_measurements_tree.configure(yscrollcommand=detail_y.set, xscrollcommand=detail_x.set)
        self.sample_measurements_tree.grid(row=0, column=0, sticky="nsew")
        detail_y.grid(row=0, column=1, sticky="ns")
        detail_x.grid(row=1, column=0, sticky="ew")
        self.sample_measurements_tree.bind("<<TreeviewSelect>>", self._on_sample_measurement_selected)

    def _build_experiments_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(wrapper, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        self.experiment_paned = paned

        controls_shell = ttk.Frame(paned)
        controls_canvas = tk.Canvas(
            controls_shell,
            highlightthickness=0,
            bg="#f6f7f9",
            height=430,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_shell,
            orient=tk.VERTICAL,
            command=controls_canvas.yview,
        )
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor=tk.NW)
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )
        controls_canvas.bind("<Enter>", self._bind_experiment_controls_mousewheel)
        controls_canvas.bind("<Leave>", self._unbind_experiment_controls_mousewheel)
        self.experiment_controls_canvas = controls_canvas
        paned.add(controls_shell, weight=0)

        path_row = ttk.Frame(controls)
        path_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(path_row, text="Sample data").pack(side=tk.LEFT)
        ttk.Entry(path_row, textvariable=self.experiment_data_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(path_row, text="Browse", command=self.browse_experiment_folder).pack(
            side=tk.LEFT
        )
        ttk.Button(path_row, text="Load", command=self.load_experiment_samples).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        overview_box = ttk.LabelFrame(controls, text="Reflectivity index overview", padding=8)
        overview_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            overview_box,
            textvariable=self.experiment_data_overview_var,
            foreground="#52606d",
            wraplength=1120,
        ).pack(anchor=tk.W)
        nav_row = ttk.Frame(overview_box)
        nav_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            nav_row,
            text="Open Fit & Optimize",
            command=lambda: self.notebook.select(self.fit_optimize_tab),
        ).pack(side=tk.LEFT)
        ttk.Button(
            nav_row,
            text="Open Colour Distance",
            command=lambda: self.notebook.select(self.colour_distance_tab),
        ).pack(side=tk.LEFT, padx=(6, 0))

        progress_box = ttk.Frame(controls)
        progress_box.pack(fill=tk.X, pady=(0, 8))
        self.experiment_progress_dog_canvas = tk.Canvas(
            progress_box,
            width=54,
            height=30,
            highlightthickness=0,
            bg="#f6f7f9",
        )
        self.experiment_progress_dog_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_progress_dog(self.experiment_progress_dog_canvas)
        self.experiment_progress = ttk.Progressbar(
            progress_box,
            mode="determinate",
            maximum=100,
        )
        self.experiment_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.experiment_progress_label = ttk.Label(
            progress_box,
            textvariable=self.busy_text_var,
            foreground="#1f2933",
            font=("Segoe UI Semibold", 10),
            width=44,
        )
        self.experiment_progress_label.pack(side=tk.LEFT, padx=(8, 0))
        self.pause_button = ttk.Button(
            progress_box,
            text="Pause",
            command=self.toggle_pause_calculation,
            state=tk.DISABLED,
        )
        self.pause_button.pack(side=tk.LEFT, padx=(8, 0))
        self.abort_button = ttk.Button(
            progress_box,
            text="Abort",
            command=self.abort_calculation,
            state=tk.DISABLED,
        )
        self.abort_button.pack(side=tk.LEFT, padx=(6, 0))
        self.experiment_status_label = ttk.Label(
            controls,
            textvariable=self.status_var,
            foreground="#52606d",
        )
        self.experiment_status_label.pack(anchor=tk.W, pady=(0, 6))

        cache_row = ttk.Frame(controls)
        cache_row.pack(fill=tk.X, pady=(0, 8))
        build_saved_button = ttk.Button(
            cache_row,
            text="Build / refresh saved results",
            command=self.build_experiment_cache,
        )
        build_saved_button.pack(side=tk.LEFT)
        self._add_tooltip(
            build_saved_button,
            "Builds the experiment model cache: before-fit measured-vs-simulated comparisons for all current experiment rows.",
        )
        load_saved_button = ttk.Button(cache_row, text="Load saved results", command=self.load_experiment_cache)
        load_saved_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            load_saved_button,
            "Loads the saved experiment model cache without recalculating. Use this before plotting or reviewing previous results.",
        )
        compare_selected_button = ttk.Button(cache_row, text="Compare selected", command=self.run_experiment_comparison)
        compare_selected_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            compare_selected_button,
            "Simulates only the selected measurement with the current model settings and shows measured vs simulated reflectance/colour.",
        )
        benchmark_button = ttk.Button(
            cache_row,
            text="Benchmark models/constants",
            command=self.benchmark_all_models_and_constants,
        )
        benchmark_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            benchmark_button,
            "Runs the experiment cache through multiple constants profiles and optical models, then ranks which model choices reduce colour error.",
        )

        display_row = ttk.Frame(controls)
        display_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(display_row, text="Plot text size").pack(side=tk.LEFT)
        self._spinbox(
            display_row,
            self.experiment_plot_text_scale_var,
            0.45,
            1.05,
            0.05,
            self._redraw_selected_experiment_plot,
        ).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Label(
            display_row,
            text="Use the divider below to give either controls or plots more room.",
            foreground="#52606d",
        ).pack(side=tk.LEFT)

        filter_box = ttk.LabelFrame(controls, text="Experiment sorting", padding=8)
        filter_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(filter_box, text="Series").pack(side=tk.LEFT)
        self.experiment_series_filter_combo = ttk.Combobox(
            filter_box,
            textvariable=self.experiment_series_filter_var,
            values=("All",),
            state="readonly",
            width=10,
        )
        self.experiment_series_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.experiment_series_filter_combo.bind("<<ComboboxSelected>>", self._on_experiment_filter_changed)
        ttk.Label(filter_box, text="Substrate").pack(side=tk.LEFT)
        self.experiment_substrate_filter_combo = ttk.Combobox(
            filter_box,
            textvariable=self.experiment_substrate_filter_var,
            values=("All", "Si", "Ti"),
            state="readonly",
            width=10,
        )
        self.experiment_substrate_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.experiment_substrate_filter_combo.bind("<<ComboboxSelected>>", self._on_experiment_filter_changed)
        ttk.Label(filter_box, text="Surface").pack(side=tk.LEFT)
        self.experiment_surface_filter_combo = ttk.Combobox(
            filter_box,
            textvariable=self.experiment_surface_filter_var,
            values=("All", "smooth", "rough", "unknown"),
            state="readonly",
            width=12,
        )
        self.experiment_surface_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.experiment_surface_filter_combo.bind("<<ComboboxSelected>>", self._on_experiment_filter_changed)
        ttk.Label(filter_box, text="Measurement").pack(side=tk.LEFT)
        self.experiment_kind_filter_combo = ttk.Combobox(
            filter_box,
            textvariable=self.experiment_kind_filter_var,
            values=("All", "specular", "integrating_sphere", "diffuse_sphere", "unknown"),
            state="readonly",
            width=20,
        )
        self.experiment_kind_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.experiment_kind_filter_combo.bind("<<ComboboxSelected>>", self._on_experiment_filter_changed)
        ttk.Label(filter_box, text="Layer filter").pack(side=tk.LEFT)
        self.experiment_composition_filter_combo = ttk.Combobox(
            filter_box,
            textvariable=self.fit_composition_filter_var,
            values=self._composition_filter_values(),
            state="readonly",
            width=20,
        )
        self.experiment_composition_filter_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.experiment_composition_filter_combo.bind("<<ComboboxSelected>>", self._on_fit_filter_changed)

        opt_box = ttk.LabelFrame(controls, text="Quick selected-fit actions", padding=8)
        opt_box.pack(fill=tk.X, pady=(0, 8))
        settings_row = ttk.Frame(opt_box)
        settings_row.pack(fill=tk.X)
        self.experiment_dog_canvas = tk.Canvas(
            settings_row,
            width=54,
            height=30,
            highlightthickness=0,
            bg="#f6f7f9",
        )
        self.experiment_dog_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_progress_dog(self.experiment_dog_canvas)
        ttk.Label(settings_row, text="Sputter-rate error +/- (%)").pack(side=tk.LEFT)
        self._spinbox(
            settings_row,
            self.thickness_opt_range_percent_var,
            0.0,
            25.0,
            0.5,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(settings_row, text="Step (%)").pack(side=tk.LEFT)
        self._spinbox(
            settings_row,
            self.thickness_opt_step_percent_var,
            0.1,
            10.0,
            0.1,
            None,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(settings_row, text="Mode").pack(side=tk.LEFT)
        quick_mode_combo = ttk.Combobox(
            settings_row,
            textvariable=self.thickness_fit_mode_var,
            values=("Individual layers", "Same material together"),
            state="readonly",
            width=20,
        )
        quick_mode_combo.pack(side=tk.LEFT, padx=(6, 12))
        self._add_tooltip(
            quick_mode_combo,
            "Individual layers lets each layer thickness move separately for the selected sample. Same material together ties repeated layers of the same material.",
        )
        scale_check = ttk.Checkbutton(
            settings_row,
            text="Fit reflectance scale",
            variable=self.thickness_fit_scale_enabled_var,
        )
        scale_check.pack(side=tk.LEFT, padx=(4, 8))
        self._add_tooltip(
            scale_check,
            "Tick this before optimizing to also fit one flat reflectance multiplier. This shows immediately whether the remaining error is mostly overall intensity rather than layer thickness.",
        )
        ttk.Label(settings_row, text="Scale min/max").pack(side=tk.LEFT)
        self._spinbox(
            settings_row,
            self.thickness_fit_scale_min_var,
            0.3,
            1.2,
            0.01,
            None,
        ).pack(side=tk.LEFT, padx=(6, 4))
        self._spinbox(
            settings_row,
            self.thickness_fit_scale_max_var,
            0.5,
            1.5,
            0.01,
            None,
        ).pack(side=tk.LEFT, padx=(4, 12))

        selected_row = ttk.Frame(opt_box)
        selected_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(selected_row, text="Selected row").pack(side=tk.LEFT)
        optimize_selected_button = ttk.Button(
            selected_row,
            text="Optimize selected measurement",
            command=self.optimize_selected_experiment_thicknesses,
        )
        optimize_selected_button.pack(side=tk.LEFT, padx=(12, 0))
        self._add_tooltip(
            optimize_selected_button,
            "Shortcut: runs the same selected-measurement thickness fit as Fit & Optimize. Use this for one sample you are currently inspecting.",
        )
        resimulate_button = ttk.Button(
            selected_row,
            text="Re-simulate fitted stack with current roughness",
            command=self.resimulate_optimized_stack_with_current_roughness,
        )
        resimulate_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            resimulate_button,
            "Keeps the cached optimized thicknesses fixed, then recalculates the spectrum with the current roughness/interface settings. Useful for testing whether roughness explains the remaining error.",
        )

        all_row = ttk.Frame(opt_box)
        all_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(all_row, text="All rows").pack(side=tk.LEFT)
        fit_filtered_button = ttk.Button(
            all_row,
            text="Fit filtered measurements...",
            command=self.precalculate_all_thickness_optimizations,
        )
        fit_filtered_button.pack(side=tk.LEFT, padx=(32, 0))
        self._add_tooltip(
            fit_filtered_button,
            "Shortcut to the overnight thickness cache: optimizes every currently filtered measurement and saves/reuses cached trials.",
        )
        fit_rates_button = ttk.Button(
            all_row,
            text="Fit sputter rates from colour",
            command=self.fit_sputter_rates_from_colour,
        )
        fit_rates_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            fit_rates_button,
            "Fits shared sputter-rate corrections from measured colour errors. This changes deposition-rate estimates, not individual measurement thicknesses.",
        )
        ttk.Label(
            all_row,
            text="Opens popup settings and reuses cached progress.",
            foreground="#52606d",
        ).pack(side=tk.LEFT, padx=(8, 0))

        cached_row = ttk.Frame(opt_box)
        cached_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(cached_row, text="Cached thickness fits").pack(side=tk.LEFT)
        self.cached_thickness_fit_combo = ttk.Combobox(
            cached_row,
            textvariable=self.cached_thickness_fit_var,
            values=(),
            state="readonly",
            width=64,
        )
        self.cached_thickness_fit_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 6))
        self.cached_thickness_fit_combo.bind("<<ComboboxSelected>>", self._on_cached_thickness_fit_selected)
        self._add_combobox_tooltip(self.cached_thickness_fit_combo, self.cached_thickness_fit_var.get)
        show_cached_button = ttk.Button(
            cached_row,
            text="Show cached fit",
            command=self.show_selected_cached_thickness_fit,
        )
        show_cached_button.pack(side=tk.LEFT)
        self._add_tooltip(
            show_cached_button,
            "Loads the selected saved thickness-fit result and redraws the measured, before-fit, and after-fit comparison.",
        )

        map_row = ttk.Frame(opt_box)
        map_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(map_row, text="Measured colour maps").pack(side=tk.LEFT)
        ttk.Button(
            map_row,
            text="TiO2",
            command=lambda: self.plot_single_material_experiment_colour_map("TiO2", use_optimized=False),
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(
            map_row,
            text="SiO2",
            command=lambda: self.plot_single_material_experiment_colour_map("SiO2", use_optimized=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            map_row,
            text="ZrO2",
            command=lambda: self.plot_single_material_experiment_colour_map("ZrO2", use_optimized=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            map_row,
            text="Ag",
            command=lambda: self.plot_single_material_experiment_colour_map("Ag", use_optimized=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            map_row,
            text="TiO2/SiO2/Ag",
            command=lambda: self.plot_tio2_sio2_experiment_colour_map(use_optimized=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            map_row,
            text="Optimized",
            command=lambda: self.plot_tio2_sio2_experiment_colour_map(use_optimized=True),
        ).pack(side=tk.LEFT, padx=(6, 0))

        roughness_row = ttk.Frame(opt_box)
        roughness_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(roughness_row, text="Roughness group").pack(side=tk.LEFT)
        roughness_fit_button = ttk.Button(
            roughness_row,
            text="Fit roughness group",
            command=self.fit_selected_roughness_group,
        )
        roughness_fit_button.pack(side=tk.LEFT, padx=(12, 0))
        self._add_tooltip(
            roughness_fit_button,
            "Fits roughness/diffuse-scatter settings for the current filtered group. Use after thickness/rate fitting if spectra are still systematically too bright or too dark.",
        )
        ttk.Label(
            opt_box,
            text=(
                "Thickness fits use the current constants, optical model, angle, interface, oxide, and roughness settings."
            ),
            foreground="#52606d",
        ).pack(anchor=tk.W, pady=(6, 0))

        list_frame = ttk.Frame(controls)
        list_frame.pack(fill=tk.X, pady=(0, 8))
        columns = ("sample", "series", "substrate", "surface", "mode", "delta_e", "measured", "simulated")
        self.experiment_results_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            height=8,
            selectmode="extended",
        )
        self.experiment_results_tree.heading("sample", text="Sample")
        self.experiment_results_tree.heading("series", text="Series")
        self.experiment_results_tree.heading("substrate", text="Substrate")
        self.experiment_results_tree.heading("surface", text="Surface")
        self.experiment_results_tree.heading("mode", text="Mode")
        self.experiment_results_tree.heading("delta_e", text="Delta E*")
        self.experiment_results_tree.heading("measured", text="Measured")
        self.experiment_results_tree.heading("simulated", text="Simulated")
        self.experiment_results_tree.column("sample", width=390, anchor=tk.W)
        self.experiment_results_tree.column("series", width=70, anchor=tk.CENTER)
        self.experiment_results_tree.column("substrate", width=80, anchor=tk.CENTER)
        self.experiment_results_tree.column("surface", width=90, anchor=tk.CENTER)
        self.experiment_results_tree.column("mode", width=135, anchor=tk.CENTER)
        self.experiment_results_tree.column("delta_e", width=90, anchor=tk.CENTER)
        self.experiment_results_tree.column("measured", width=110, anchor=tk.CENTER)
        self.experiment_results_tree.column("simulated", width=110, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=self.experiment_results_tree.yview,
        )
        self.experiment_results_tree.configure(yscrollcommand=scrollbar.set)
        self.experiment_results_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.experiment_results_tree.bind("<<TreeviewSelect>>", self._on_experiment_result_selected)
        self.experiment_results_tree.bind("<Motion>", self._on_experiment_tree_motion, add="+")
        self.experiment_results_tree.bind("<Leave>", self._on_experiment_tree_leave, add="+")

        self.experiment_info_var = tk.StringVar(value="Load experiment data to begin.")
        ttk.Label(controls, textvariable=self.experiment_info_var, foreground="#52606d").pack(
            anchor=tk.W
        )

        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=1)
        self.experiment_figure = Figure(figsize=(8.4, 5.2), dpi=120)
        experiment_plot_header = ttk.Frame(plot_frame)
        experiment_plot_header.pack(fill=tk.X)
        self._pack_download_figure_button(
            experiment_plot_header,
            lambda: self.experiment_figure,
            "experiments",
        )
        self.experiment_canvas = FigureCanvasTkAgg(self.experiment_figure, plot_frame)
        experiment_canvas_widget = self.experiment_canvas.get_tk_widget()
        experiment_canvas_widget.configure(height=420)
        experiment_canvas_widget.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.experiment_canvas.mpl_connect("scroll_event", self._on_experiment_scroll)
        self.root.after_idle(self._set_experiment_initial_sash)

    def _set_experiment_initial_sash(self) -> None:
        paned = getattr(self, "experiment_paned", None)
        if paned is None:
            return
        try:
            total_height = paned.winfo_height()
            if total_height <= 1:
                self.root.after(80, self._set_experiment_initial_sash)
                return
            controls_height = max(280, min(500, total_height - 420))
            paned.sashpos(0, controls_height)
        except tk.TclError:
            pass

    def _bind_experiment_controls_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_experiment_controls_mousewheel)

    def _unbind_experiment_controls_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _on_experiment_controls_mousewheel(self, event) -> None:
        canvas = getattr(self, "experiment_controls_canvas", None)
        if canvas is None:
            return
        delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta * 3, "units")

    def _redraw_selected_experiment_plot(self, *_args) -> None:
        if (
            getattr(self, "experiment_cache", None) is not None
            and hasattr(self, "experiment_results_tree")
            and self.experiment_results_tree.selection()
        ):
            self._on_experiment_result_selected()

    def _build_rate_groups_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(wrapper)
        controls.pack(fill=tk.X, pady=(0, 8))
        refresh_groups_button = ttk.Button(controls, text="Refresh groups", command=self.refresh_rate_groups)
        refresh_groups_button.pack(side=tk.LEFT)
        self._add_tooltip(
            refresh_groups_button,
            "Rebuilds the table of sputter-rate groups from the loaded sample data. No fitting is run.",
        )
        fit_selected_groups_button = ttk.Button(
            controls,
            text="Fit selected groups from colour",
            command=self.fit_selected_sputter_rate_groups_from_colour,
        )
        fit_selected_groups_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            fit_selected_groups_button,
            "Fits a shared sputter-rate correction for the selected rate groups using measured colour errors.",
        )
        fit_all_groups_button = ttk.Button(
            controls,
            text="Fit all groups + make figure",
            command=self.fit_all_sputter_rate_groups_from_colour,
        )
        fit_all_groups_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            fit_all_groups_button,
            "Fits all currently visible rate groups and writes summary figures/CSV files for comparing rate corrections.",
        )
        ttk.Label(
            controls,
            text="Select groups with matching material, target, pressure, and gas flow.",
            foreground="#52606d",
        ).pack(side=tk.LEFT, padx=(12, 0))

        content = ttk.PanedWindow(wrapper, orient=tk.VERTICAL)
        content.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(content)
        plot_frame = ttk.Frame(content)
        content.add(table_frame, weight=2)
        content.add(plot_frame, weight=1)

        columns = ("material", "target", "pressure", "sccm", "date_interval", "rate", "samples", "measurements", "sample_names")
        self.rate_groups_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=18,
            selectmode="extended",
        )
        headings = {
            "material": "Material",
            "target": "Target",
            "pressure": "Pressure (mbar)",
            "sccm": "Gas flow (sccm)",
            "date_interval": "Date interval",
            "rate": "Base rate (nm/min)",
            "samples": "Samples",
            "measurements": "Measurements",
            "sample_names": "Sample names",
        }
        widths = {
            "material": 90,
            "target": 80,
            "pressure": 120,
            "sccm": 120,
            "date_interval": 150,
            "rate": 145,
            "samples": 90,
            "measurements": 110,
            "sample_names": 560,
        }
        for column in columns:
            self.rate_groups_tree.heading(column, text=headings[column])
            self.rate_groups_tree.column(column, width=widths[column], anchor=tk.W)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.rate_groups_tree.yview)
        self.rate_groups_tree.configure(yscrollcommand=scrollbar.set)
        self.rate_groups_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rate_group_records: dict[str, dict[str, object]] = {}

        self.rate_groups_figure = Figure(figsize=(8.4, 3.8), dpi=170)
        rate_plot_header = ttk.Frame(plot_frame)
        rate_plot_header.pack(fill=tk.X)
        self._pack_download_figure_button(
            rate_plot_header,
            lambda: self.rate_groups_figure,
            "rate_groups",
        )
        self.rate_groups_canvas = FigureCanvasTkAgg(self.rate_groups_figure, plot_frame)
        self.rate_groups_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def _build_calibration_tab(self, parent: ttk.Frame) -> None:
        wrapper = ttk.Frame(parent, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(wrapper, text="Physical calibration from sputtering times", padding=8)
        controls.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Sample group").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.calibration_group_var,
            values=(
                "smooth Si",
                "smooth Si double polished",
                "rough Si",
                "rough Si double polished",
                "rough Ti",
                "smooth",
                "rough",
                "All",
            ),
            width=24,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(row, text="Rate +/- (%)").pack(side=tk.LEFT)
        self._spinbox(row, self.calibration_rate_range_var, 0.5, 20.0, 0.5, lambda: None).pack(
            side=tk.LEFT,
            padx=(6, 16),
        )
        ttk.Label(row, text="Rate points").pack(side=tk.LEFT)
        self._spinbox(row, self.calibration_rate_points_var, 5, 101, 2, lambda: None).pack(
            side=tk.LEFT,
            padx=(6, 16),
        )
        run_calibration_button = ttk.Button(
            row,
            text="Run calibration",
            command=self.run_physical_calibration,
        )
        run_calibration_button.pack(side=tk.LEFT)
        self._add_tooltip(
            run_calibration_button,
            "Tests constants profiles and optical models, then fits shared sputter-rate corrections for the selected calibration group.",
        )
        use_model_button = ttk.Button(
            row,
            text="Use selected model",
            command=self.apply_selected_calibration_model,
        )
        use_model_button.pack(side=tk.LEFT, padx=(6, 0))
        self._add_tooltip(
            use_model_button,
            "Applies the selected calibration row's constants profile and optical model to the main controls for future simulations/fits.",
        )
        ttk.Label(
            controls,
            text=(
                "Tests constants profiles and optical models, then fits one shared sputter-rate "
                "correction per material/target/pressure/gas-flow group. Start with smooth Si."
            ),
            foreground="#52606d",
            wraplength=980,
        ).pack(anchor=tk.W, pady=(6, 0))

        table_frame = ttk.Frame(wrapper)
        table_frame.pack(fill=tk.X, pady=(0, 8))
        columns = (
            "rank",
            "group",
            "profile",
            "model",
            "before",
            "after",
            "improvement",
            "rate_span",
            "groups",
            "measurements",
        )
        self.calibration_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headings = {
            "rank": "#",
            "group": "Group",
            "profile": "Constants",
            "model": "Optical model",
            "before": f"Before {self._delta_e_label()}",
            "after": f"After {self._delta_e_label()}",
            "improvement": "Improvement",
            "rate_span": "Max rate change",
            "groups": "Rate groups",
            "measurements": "Measurements",
        }
        widths = {
            "rank": 45,
            "group": 90,
            "profile": 180,
            "model": 210,
            "before": 100,
            "after": 100,
            "improvement": 100,
            "rate_span": 115,
            "groups": 95,
            "measurements": 105,
        }
        for column in columns:
            self.calibration_tree.heading(column, text=headings[column])
            self.calibration_tree.column(column, width=widths[column], anchor=tk.CENTER)
        self.calibration_tree.pack(fill=tk.X)

        self.calibration_figure = Figure(figsize=(8.4, 6.2), dpi=170)
        calibration_header = ttk.Frame(wrapper)
        calibration_header.pack(fill=tk.X)
        self._pack_download_figure_button(
            calibration_header,
            lambda: self.calibration_figure,
            "calibration",
        )
        self.calibration_canvas = FigureCanvasTkAgg(self.calibration_figure, wrapper)
        self.calibration_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def browse_experiment_folder(self) -> None:
        folder = filedialog.askdirectory(
            initialdir=self.experiment_data_path_var.get(),
            title="Select sample_data folder",
        )
        if folder:
            self.experiment_data_path_var.set(folder)
            self.load_experiment_samples()

    @staticmethod
    def _is_experiment_data_folder(path: Path) -> bool:
        required_files = ("sample_index.csv", "measurement_index.csv", "thickness_estimates.csv")
        return path.is_dir() and all((path / name).exists() for name in required_files)

    def _candidate_experiment_data_paths(self, requested: Path) -> list[Path]:
        project_root = Path(__file__).resolve().parent
        project_parent = project_root.parent
        candidates = [
            requested,
            requested / "sample_data",
            project_parent / "Reflectivity" / "sample_data",
            project_root / "data" / "sample_data",
            Path.home() / "Desktop" / "Reflectivity" / "sample_data",
            Path.home()
            / "OneDrive - Aarhus universitet"
            / "Skrivebord"
            / "Reflectivity"
            / "sample_data",
            Path.home()
            / "OneDrive - Aarhus universitet"
            / "Skrivebord"
            / "Nyt program"
            / "Reflectivity"
            / "sample_data",
        ]
        if requested.name.lower() == "sample_data" and requested.parent.name.lower() == "reflectivity":
            candidates.extend(
                [
                    project_parent / requested.parent.name / requested.name,
                    project_parent.parent / requested.parent.name / requested.name,
                ]
            )
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _resolve_experiment_data_path(self) -> Path:
        requested = Path(self.experiment_data_path_var.get()).expanduser()
        for candidate in self._candidate_experiment_data_paths(requested):
            if self._is_experiment_data_folder(candidate):
                resolved = candidate.resolve()
                if resolved != requested:
                    self.experiment_data_path_var.set(str(resolved))
                return resolved
        required = "sample_index.csv, measurement_index.csv, thickness_estimates.csv"
        raise FileNotFoundError(
            f"Could not find a sample_data folder with {required}. "
            f"Current path: {requested}"
        )

    def load_experiment_samples(self, show_errors: bool = True) -> None:
        try:
            sample_data_root = self._resolve_experiment_data_path()
            self.experiment_store = ExperimentDataStore(sample_data_root)
            self.latest_sputter_rates_cache = None
            self.plots_before_points_cache = None
            self.plots_after_points_cache = None
            self.plots_deposited_samples_cache = {}
            sample_names = self.experiment_store.sample_names(require_spectra=True)
            if sample_names:
                if self.experiment_sample_var.get() not in sample_names:
                    self.experiment_sample_var.set(sample_names[0])
                self._update_experiment_data_overview(sample_data_root)
                self.experiment_info_var.set(
                    f"Loaded {len(sample_names)} samples with spectra. "
                    "Build or load saved results to browse them."
                )
                source_paths = [
                    sample_data_root / "measurement_index.csv",
                    sample_data_root / "thickness_estimates.csv",
                ]
                source_mtime = max(
                    (path.stat().st_mtime for path in source_paths if path.exists()),
                    default=0.0,
                )
                cache_is_current = (
                    self.experiment_cache_path.exists()
                    and self.experiment_cache_path.stat().st_mtime >= source_mtime
                )
                if cache_is_current:
                    self.load_experiment_cache()
                else:
                    self.experiment_cache = None
                    self._clear_experiment_results_tree()
                    if self.experiment_cache_path.exists():
                        self.experiment_info_var.set(
                            "Experiment data changed after the saved cache was made. "
                            "Build the experiment cache again to show every new measurement."
                        )
            else:
                self._update_experiment_data_overview(sample_data_root)
                self.experiment_info_var.set("No samples with linked spectra were found.")
            self.refresh_rate_groups()
            self._update_sputter_time_estimate()
            if hasattr(self, "samples_tree"):
                self.refresh_samples_overview(redraw_only=True)
        except Exception as exc:
            self.experiment_store = None
            self.latest_sputter_rates_cache = None
            self.plots_before_points_cache = None
            self.plots_after_points_cache = None
            self.plots_deposited_samples_cache = {}
            self.experiment_info_var.set(f"Could not load experiment data: {exc}")
            self.refresh_rate_groups()
            self._update_sputter_time_estimate()
            if hasattr(self, "samples_tree"):
                self.refresh_samples_overview(redraw_only=True)
            if show_errors:
                messagebox.showerror("Experiments", str(exc))

    def _update_experiment_data_overview(self, sample_data_root: Path) -> None:
        try:
            sample_index = pd.read_csv(sample_data_root / "sample_index.csv", keep_default_na=False)
            measurement_index = pd.read_csv(sample_data_root / "measurement_index.csv", keep_default_na=False)
            thickness = pd.read_csv(sample_data_root / "thickness_estimates.csv", keep_default_na=False)
            sputtering_doc = sample_data_root.parent / "all sputtering.docx"
            sputtering_stamp = (
                datetime.fromtimestamp(sputtering_doc.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                if sputtering_doc.exists()
                else "not found"
            )
            spectra_count = int(pd.to_numeric(sample_index.get("spectra"), errors="coerce").fillna(0).sum())
            rough_count = (
                int((measurement_index["surface"].astype(str) == "rough").sum())
                if "surface" in measurement_index
                else 0
            )
            overview = (
                f"Reflectivity indexes: {len(sample_index)} samples, "
                f"{len(measurement_index)} measurement rows, {spectra_count} linked spectra, "
                f"{len(thickness)} thickness/rate rows, {rough_count} rough measurements. "
                f"all sputtering.docx: {sputtering_stamp}"
            )
        except Exception as exc:
            overview = f"Reflectivity/sample_data loaded, but overview could not be summarized: {exc}"
        self.experiment_data_overview_var.set(overview)

    def refresh_samples_overview(self, redraw_only: bool = False) -> None:
        """Populate the Samples tab with one row per sample name."""

        if not hasattr(self, "samples_tree"):
            return
        if self.experiment_store is None:
            self._clear_samples_overview("No Reflectivity/sample_data folder is loaded.")
            if not redraw_only:
                self.load_experiment_samples()
            return

        rows = self._sample_overview_rows()
        self.samples_overview_rows = {str(row["sample"]): row for row in rows}
        for item in self.samples_tree.get_children():
            self.samples_tree.delete(item)
        for row in rows:
            tags: list[str] = []
            if int(row["condition_count"]) > 1:
                tags.append("multiple_variants")
            if int(row["loaded_spectra"]) == 0:
                tags.append("no_loaded_spectra")
            self.samples_tree.insert(
                "",
                tk.END,
                iid=str(row["sample"]),
                tags=tuple(tags),
                values=(
                    row["sample"],
                    row["series"],
                    row["layers"],
                    row["stack"],
                    row["measurement_rows"],
                    row["loaded_spectra"],
                    row["condition_count"],
                    row["surfaces"],
                    row["substrates"],
                    row["kinds"],
                    row["indexed_spectra"],
                    row["indexed_colors"],
                ),
            )

        total_samples = len(rows)
        total_measurement_rows = sum(int(row["measurement_rows"]) for row in rows)
        total_loaded_spectra = sum(int(row["loaded_spectra"]) for row in rows)
        total_variants = sum(int(row["condition_count"]) for row in rows)
        samples_with_measurements = sum(int(row["measurement_rows"]) > 0 for row in rows)
        multi_variant_samples = sum(int(row["condition_count"]) > 1 for row in rows)
        self.samples_overview_info_var.set(
            f"{total_samples} sample names; {samples_with_measurements} with measurement rows; "
            f"{total_measurement_rows} measurement rows; {total_loaded_spectra} loadable spectra; "
            f"{total_variants} surface/substrate/kind condition variants. "
            f"{multi_variant_samples} sample names contain more than one condition variant."
        )

        selected = [item for item in self.samples_tree.selection() if item in self.samples_overview_rows]
        if selected:
            self._on_samples_tree_selected()
        elif rows:
            first_item = str(rows[0]["sample"])
            self.samples_tree.selection_set(first_item)
            self.samples_tree.see(first_item)
            self._on_samples_tree_selected()
        else:
            self._clear_sample_measurement_details("No samples were found in sample_index.csv.")

    def _clear_samples_overview(self, message: str) -> None:
        if hasattr(self, "samples_tree"):
            for item in self.samples_tree.get_children():
                self.samples_tree.delete(item)
        if hasattr(self, "sample_measurements_tree"):
            for item in self.sample_measurements_tree.get_children():
                self.sample_measurements_tree.delete(item)
        self.samples_overview_rows = {}
        self.samples_overview_info_var.set(message)
        self.sample_measurements_info_var.set("Select a sample to see each measured spectrum and its condition.")

    def _sample_overview_rows(self) -> list[dict[str, object]]:
        store = self.experiment_store
        if store is None:
            return []

        sample_index, measurement_row_counts = self._sample_overview_index_counts()
        sample_names = set(store.sample_names(require_spectra=False))
        sample_names.update(sample_index)
        sample_names.update(measurement_row_counts)
        rows: list[dict[str, object]] = []
        for sample_name in sorted(sample_names, key=self._sample_name_sort_key):
            sample = store.load_sample(sample_name)
            measurements = list(sample.measurements)
            condition_keys = {self._measurement_condition_key(measurement) for measurement in measurements}
            substrates = [
                getattr(measurement, "substrate_group", "")
                or getattr(measurement, "substrate_hint", "")
                for measurement in measurements
            ]
            row_index = sample_index.get(sample_name, {})
            rows.append(
                {
                    "sample": sample_name,
                    "series": sample_series_from_name(sample_name),
                    "layers": len(sample.layer_estimates),
                    "stack": sample.stack_label,
                    "measurement_rows": measurement_row_counts.get(sample_name, len(measurements)),
                    "loaded_spectra": len(measurements),
                    "condition_count": len(condition_keys),
                    "surfaces": self._join_unique_display_values(
                        getattr(measurement, "surface_class", "") for measurement in measurements
                    ),
                    "substrates": self._join_unique_display_values(substrates),
                    "kinds": self._join_unique_display_values(
                        getattr(measurement, "measurement_kind", "") for measurement in measurements
                    ),
                    "indexed_spectra": int(row_index.get("spectra", 0)),
                    "indexed_colors": int(row_index.get("colors", 0)),
                }
            )
        return rows

    def _sample_overview_index_counts(self) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        if self.experiment_store is None:
            return {}, {}
        sample_data_root = self.experiment_store.sample_data_root
        sample_index: dict[str, dict[str, int]] = {}
        try:
            data = pd.read_csv(sample_data_root / "sample_index.csv", keep_default_na=False)
            if "sample_name" in data.columns:
                for _, row in data.iterrows():
                    sample_name = self._display_text(row.get("sample_name"), "")
                    if not sample_name:
                        continue
                    sample_index[sample_name] = {
                        "spectra": self._integer_cell(row.get("spectra")),
                        "colors": self._integer_cell(row.get("colors")),
                    }
        except Exception:
            pass

        measurement_counts: dict[str, int] = {}
        try:
            data = pd.read_csv(sample_data_root / "measurement_index.csv", keep_default_na=False)
            if "sample_name" in data.columns:
                counts = data["sample_name"].astype(str).value_counts()
                measurement_counts = {str(name): int(count) for name, count in counts.items()}
        except Exception:
            pass
        return sample_index, measurement_counts

    def _on_samples_tree_selected(self, *_args) -> None:
        if self.experiment_store is None or not hasattr(self, "sample_measurements_tree"):
            return
        selection = self.samples_tree.selection()
        if not selection:
            self._clear_sample_measurement_details("Select a sample to see each measured spectrum and its condition.")
            return
        sample_name = str(selection[0])
        sample = self.experiment_store.load_sample(sample_name)
        measurements = list(sample.measurements)
        for item in self.sample_measurements_tree.get_children():
            self.sample_measurements_tree.delete(item)
        if not measurements:
            row = self.samples_overview_rows.get(sample_name, {})
            measurement_rows = int(row.get("measurement_rows", 0))
            self.sample_measurements_info_var.set(
                f"{sample_name}: no loadable spectra. The index has {measurement_rows} measurement rows."
            )
            return

        condition_numbers: dict[tuple[str, str, str, str], int] = {}
        for index, measurement in enumerate(measurements, start=1):
            key = self._measurement_condition_key(measurement)
            condition_numbers.setdefault(key, len(condition_numbers) + 1)
            self.sample_measurements_tree.insert(
                "",
                tk.END,
                iid=f"{sample_name}_{index}",
                values=(
                    condition_numbers[key],
                    key[0],
                    key[1],
                    key[2],
                    self._display_text(getattr(measurement, "source_system", ""), "unknown"),
                    self._display_text(getattr(measurement, "instrument_sample_id", ""), ""),
                    self._shorten_middle(getattr(measurement, "description", ""), 105),
                    self._shorten_middle(self._sample_csv_display(measurement.csv_path), 105),
                ),
            )

        labels = [self._condition_label_from_key(key) for key in condition_numbers]
        self.sample_measurements_info_var.set(
            f"{sample_name}: {len(measurements)} loadable spectra across "
            f"{len(condition_numbers)} condition variants: {', '.join(labels)}."
        )
        first_detail = f"{sample_name}_1"
        if self.sample_measurements_tree.exists(first_detail):
            self.sample_measurements_tree.selection_set(first_detail)
            self.sample_measurements_tree.see(first_detail)
        self.sample_sweep_info_var.set(
            self._sample_sweep_status_text(sample_name, sample, 0, measurements[0])
        )

    def _on_sample_measurement_selected(self, *_args) -> None:
        try:
            sample_name, measurement_index, sample, measurement = self._selected_sample_measurement()
        except Exception:
            return
        self.sample_sweep_info_var.set(
            self._sample_sweep_status_text(sample_name, sample, measurement_index, measurement)
        )

    def run_selected_sample_layer_sweeps(self) -> None:
        self._run_selected_sample_sweeps(include_layers=True, include_angle=False)

    def run_selected_sample_angle_sweep(self) -> None:
        self._run_selected_sample_sweeps(include_layers=False, include_angle=True)

    def run_selected_sample_all_sweeps(self) -> None:
        self._run_selected_sample_sweeps(include_layers=True, include_angle=True)

    def _run_selected_sample_sweeps(self, include_layers: bool, include_angle: bool) -> None:
        try:
            context = self._selected_sample_sweep_context()
            layer_specs = self._sample_sweep_layer_specs(context["stack"], context["model"], context["sample"])
            if include_layers and not layer_specs:
                raise ValueError("The selected sample has no deposited layer estimates to sweep.")
            result_count = (len(layer_specs) if include_layers else 0) + (1 if include_angle else 0)
            if result_count <= 0:
                raise ValueError("Choose at least one sweep type.")

            thickness_min_nm = float(self.sweep_min_var.get())
            thickness_max_nm = float(self.sweep_max_var.get())
            if thickness_max_nm <= thickness_min_nm:
                raise ValueError("Sweep max nm must be larger than min nm.")
            thickness_points = int(self.sweep_points_1d_var.get())
            angle_min_deg = float(self.angle_sweep_min_var.get())
            angle_max_deg = float(self.angle_sweep_max_var.get())
            if angle_max_deg <= angle_min_deg:
                raise ValueError("Angle max must be larger than angle min.")
            angle_points = int(self.angle_sweep_points_var.get())
            quality = self.sweep_quality_var.get()
            thickness_angle_deg = float(context["thickness_angle_deg"])
            sample_name = str(context["sample_name"])

            def task(progress):
                results: list[tuple[str, str, object, float | None]] = []
                done = 0
                if include_layers:
                    for spec in layer_specs:
                        self._wait_if_paused(progress)
                        progress(done, f"{sample_name}: sweeping {spec['label']} thickness")
                        result = run_thickness_sweep_1d(
                            stack=context["stack"],
                            model=context["model"],
                            layer=spec["selector"],
                            layer_occurrence=int(spec["occurrence"]),
                            thickness_min_nm=thickness_min_nm,
                            thickness_max_nm=thickness_max_nm,
                            angle_deg=thickness_angle_deg,
                            num_points=thickness_points,
                            quality=quality,
                        )
                        results.append(("layer", str(spec["label"]), result, float(spec["nominal_nm"])))
                        done += 1
                        progress(done, f"{sample_name}: {spec['label']} thickness sweep complete")
                if include_angle:
                    self._wait_if_paused(progress)
                    progress(done, f"{sample_name}: sweeping angle")
                    result = run_angle_sweep(
                        stack=context["stack"],
                        model=context["model"],
                        angle_min_deg=angle_min_deg,
                        angle_max_deg=angle_max_deg,
                        num_points=angle_points,
                        quality=quality,
                    )
                    results.append(("angle", "angle", result, float(context["thickness_angle_deg"])))
                    done += 1
                    progress(done, f"{sample_name}: angle sweep complete")
                return {**context, "results": results}

            def on_success(payload) -> str:
                last_tab = None
                sample_label = self._sample_sweep_plot_label(payload)
                for kind, label, result, marker_value in payload["results"]:
                    tab_title = f"{payload['sample_name']} {label}"
                    tab, figure, canvas = self._new_sweep_plot(
                        tab_title[:18],
                        plot_kind="1d" if kind == "layer" else "angle",
                    )
                    if kind == "layer":
                        self._draw_sample_1d_sweep_result(
                            result,
                            figure,
                            canvas,
                            sample_label,
                            nominal_thickness_nm=marker_value,
                            angle_deg=float(payload["thickness_angle_deg"]),
                        )
                    else:
                        self._draw_sample_angle_sweep_result(
                            result,
                            figure,
                            canvas,
                            sample_label,
                            reference_angle_deg=marker_value,
                        )
                    last_tab = tab
                if last_tab is not None:
                    self._select_sweep_tab(last_tab)
                self.sample_sweep_info_var.set(
                    f"Created {len(payload['results'])} sweep plot(s) for {payload['sample_name']}."
                )
                return f"Created {len(payload['results'])} selected-sample sweep plot(s)."

            title = "Sample sweeps"
            busy = f"running sample sweeps for {sample_name}"
            self._run_background(task, on_success, title=title, busy_message=busy, progress_max=result_count)
        except Exception as exc:
            messagebox.showerror("Sample sweeps", str(exc))

    def _selected_sample_measurement(self):
        if self.experiment_store is None:
            raise ValueError("Load Reflectivity/sample_data first.")
        if not hasattr(self, "samples_tree"):
            raise ValueError("Samples tab is not ready.")
        selection = self.samples_tree.selection()
        if not selection:
            raise ValueError("Select a sample first.")
        sample_name = str(selection[0])
        sample = self.experiment_store.load_sample(sample_name)
        if not sample.measurements:
            raise ValueError(f"{sample_name} has no loadable measurement spectra.")
        measurement_index = 0
        detail_selection = self.sample_measurements_tree.selection() if hasattr(self, "sample_measurements_tree") else ()
        if detail_selection:
            try:
                candidate_index = int(str(detail_selection[0]).rsplit("_", 1)[1]) - 1
            except (IndexError, ValueError):
                candidate_index = 0
            if 0 <= candidate_index < len(sample.measurements):
                measurement_index = candidate_index
        return sample_name, measurement_index, sample, sample.measurements[measurement_index]

    def _selected_sample_sweep_context(self) -> dict[str, object]:
        sample_name, measurement_index, sample, measurement = self._selected_sample_measurement()
        substrate_name = (
            getattr(measurement, "substrate_hint", None)
            or normalize_substrate_name(getattr(measurement, "substrate_group", ""))
            or self.substrate_var.get()
        )
        if substrate_name not in self.materials:
            raise ValueError(f"Substrate {substrate_name!r} is not available in the current constants profile.")
        native_oxide = self._native_oxide_from_controls(substrate_name)
        stack = build_stack_from_estimates(
            sample,
            materials=self.materials,
            substrate_name=substrate_name,
            native_oxide=native_oxide,
            use_effective_interfaces=self._use_effective_interfaces(),
            interface_thickness_nm=float(self.roughness_thickness_var.get()),
            interface_fraction=float(self.roughness_fraction_var.get()),
        )
        measurement_angle = getattr(measurement, "measurement_angle_deg", None)
        if measurement_angle is None or not np.isfinite(float(measurement_angle)):
            thickness_angle_deg = float(self.angle_var.get())
        else:
            thickness_angle_deg = float(measurement_angle)
        return {
            "sample_name": sample_name,
            "measurement_index": measurement_index,
            "sample": sample,
            "measurement": measurement,
            "substrate_name": substrate_name,
            "stack": stack,
            "model": self._model_from_controls(),
            "model_label": self.model_mode_var.get(),
            "material_profile": self.material_profile_var.get(),
            "thickness_angle_deg": thickness_angle_deg,
        }

    def _sample_sweep_layer_specs(self, stack, model, sample) -> list[dict[str, object]]:
        if not sample.layer_estimates:
            return []
        prepared = model.prepare_stack(stack, np.asarray([550.0], dtype=float))
        display_indices = list(prepared.display_layer_indices)
        material_totals: dict[str, int] = {}
        for layer in sample.layer_estimates:
            material_totals[layer.material_name] = material_totals.get(layer.material_name, 0) + 1
        material_seen: dict[str, int] = {}
        specs: list[dict[str, object]] = []
        for index, layer in enumerate(sample.layer_estimates):
            occurrence = material_seen.get(layer.material_name, 0)
            material_seen[layer.material_name] = occurrence + 1
            label = (
                f"{layer.material_name} #{occurrence + 1}"
                if material_totals.get(layer.material_name, 0) > 1
                else layer.material_name
            )
            if index < len(display_indices):
                selector: int | str = int(display_indices[index])
                selector_occurrence = 0
            else:
                selector = layer.material_name
                selector_occurrence = occurrence
            specs.append(
                {
                    "label": label,
                    "selector": selector,
                    "occurrence": selector_occurrence,
                    "nominal_nm": float(layer.thickness_nm),
                }
            )
        return specs

    def _sample_sweep_status_text(self, sample_name: str, sample, measurement_index: int, measurement) -> str:
        context = self._sample_measurement_context_label(measurement)
        layer_count = len(sample.layer_estimates)
        stack_label = sample.stack_label
        return (
            f"Selected {sample_name}, measurement {measurement_index + 1}: {context}. "
            f"{layer_count} deposited layer(s): {stack_label}."
        )

    def _sample_measurement_context_label(self, measurement) -> str:
        surface = self._display_text(getattr(measurement, "surface_class", ""), "unknown surface")
        substrate = self._display_text(
            getattr(measurement, "substrate_group", "")
            or getattr(measurement, "substrate_hint", ""),
            "unknown substrate",
        )
        kind = self._display_text(getattr(measurement, "measurement_kind", ""), "unknown measurement")
        return f"{surface}, {substrate}, {kind}"

    def _sample_sweep_plot_label(self, payload: dict[str, object]) -> str:
        measurement = payload["measurement"]
        return (
            f"{payload['sample_name']} ({self._sample_measurement_context_label(measurement)}); "
            f"{payload['material_profile']} constants; {payload['model_label']}"
        )

    def _draw_sample_1d_sweep_result(
        self,
        result,
        figure: Figure,
        canvas: FigureCanvasTkAgg,
        sample_label: str,
        nominal_thickness_nm: float | None,
        angle_deg: float,
    ) -> None:
        figure.clear()
        figure.set_size_inches(7.2, 2.65, forward=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(
            result.rgb_values[np.newaxis, :, :],
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_nm[0]),
                float(result.thickness_values_nm[-1]),
                0.0,
                1.0,
            ],
        )
        if nominal_thickness_nm is not None:
            ax.axvline(
                float(nominal_thickness_nm),
                color="#111827",
                linestyle="--",
                linewidth=1.2,
                label=f"estimate {float(nominal_thickness_nm):g} nm",
            )
            ax.legend(loc="upper right", fontsize=10, frameon=True)
        ax.set_yticks([])
        ax.set_xlabel(f"{result.layer_name} thickness (nm)")
        ax.set_title(
            self._sweep_report_title(
                f"{self._sample_sweep_short_name(sample_label)}: {result.layer_name} thickness sweep",
                result.stack_label,
                context=f"Angle {angle_deg:g} deg",
            )
        )
        self._style_report_sweep_axis(ax)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.62)
        canvas.draw_idle()

    def _draw_sample_angle_sweep_result(
        self,
        result,
        figure: Figure,
        canvas: FigureCanvasTkAgg,
        sample_label: str,
        reference_angle_deg: float | None,
    ) -> None:
        figure.clear()
        figure.set_size_inches(7.2, 2.65, forward=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(
            result.rgb_values[np.newaxis, :, :],
            aspect="auto",
            origin="lower",
            extent=[
                float(result.angle_values_deg[0]),
                float(result.angle_values_deg[-1]),
                0.0,
                1.0,
            ],
        )
        if reference_angle_deg is not None:
            ax.axvline(
                float(reference_angle_deg),
                color="#111827",
                linestyle="--",
                linewidth=1.2,
                label=f"reference {float(reference_angle_deg):g} deg",
            )
            ax.legend(loc="upper right", fontsize=10, frameon=True)
        ax.set_yticks([])
        ax.set_xlabel("Angle of incidence (deg)")
        ax.set_title(
            self._sweep_report_title(
                f"{self._sample_sweep_short_name(sample_label)}: angle sweep",
                result.stack_label,
            )
        )
        self._style_report_sweep_axis(ax)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.67)
        canvas.draw_idle()

    def _clear_sample_measurement_details(self, message: str) -> None:
        if hasattr(self, "sample_measurements_tree"):
            for item in self.sample_measurements_tree.get_children():
                self.sample_measurements_tree.delete(item)
        self.sample_measurements_info_var.set(message)
        self.sample_sweep_info_var.set("Select a sample and measurement row before running sample sweeps.")

    def _sample_csv_display(self, csv_path: Path) -> str:
        if self.experiment_store is None:
            return str(csv_path)
        try:
            return str(Path(csv_path).relative_to(self.experiment_store.reflectivity_root))
        except ValueError:
            return str(csv_path)

    @staticmethod
    def _measurement_condition_key(measurement) -> tuple[str, str, str, str]:
        surface = ThinFilmDesignerApp._display_text(getattr(measurement, "surface_class", ""), "unknown")
        substrate = ThinFilmDesignerApp._display_text(
            getattr(measurement, "substrate_group", "")
            or getattr(measurement, "substrate_hint", ""),
            "unknown",
        )
        kind = ThinFilmDesignerApp._display_text(getattr(measurement, "measurement_kind", ""), "unknown")
        note = ThinFilmDesignerApp._display_text(getattr(measurement, "sample_condition_note", ""), "")
        return surface, substrate, kind, note

    @staticmethod
    def _condition_label_from_key(key: tuple[str, str, str, str]) -> str:
        surface, substrate, kind, note = key
        parts = [surface, substrate, kind]
        if note:
            parts.append(note)
        return " / ".join(parts)

    @staticmethod
    def _join_unique_display_values(values, fallback: str = "none") -> str:
        cleaned = {
            ThinFilmDesignerApp._display_text(value, "")
            for value in values
            if ThinFilmDesignerApp._display_text(value, "")
        }
        if not cleaned:
            return fallback
        return ", ".join(sorted(cleaned, key=str.lower))

    @staticmethod
    def _display_text(value, fallback: str = "unknown") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        if text.lower() in {"nan", "none"}:
            return fallback
        return text if text else fallback

    @staticmethod
    def _integer_cell(value) -> int:
        try:
            text = str(value).strip().replace(",", ".")
            if not text:
                return 0
            return int(float(text))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _try_float_variable(variable) -> float | None:
        try:
            return float(variable.get())
        except (tk.TclError, TypeError, ValueError):
            return None

    @staticmethod
    def _sample_name_sort_key(sample_name: str) -> tuple[str, int, str]:
        text = str(sample_name)
        series = sample_series_from_name(text)
        suffix = text[len(series) :].lstrip("-_ ")
        number_text = ""
        for character in suffix:
            if not character.isdigit():
                break
            number_text += character
        number = int(number_text) if number_text else 10**9
        return series.lower(), number, text.lower()

    @staticmethod
    def _sample_group_sort_key(sample_name: object) -> tuple[int, str, int, str]:
        """Sort colour-distance figures by sample family before fit quality."""

        text = str(sample_name or "").strip()
        series, number, lowered = ThinFilmDesignerApp._sample_name_sort_key(text)
        preferred = {"s": 0, "b": 1, "c": 2, "a": 3, "d": 4, "au": 5, "other": 99}
        return preferred.get(series, 90), series, number, lowered

    @staticmethod
    def _shorten_middle(value, max_len: int = 90) -> str:
        text = str(value or "").strip()
        if len(text) <= max_len:
            return text
        keep = max(max_len - 5, 8)
        head = keep // 2
        tail = keep - head
        return f"{text[:head]} ... {text[-tail:]}"

    def refresh_rate_groups(self) -> None:
        if not hasattr(self, "rate_groups_tree"):
            return
        for item in self.rate_groups_tree.get_children():
            self.rate_groups_tree.delete(item)
        self.rate_group_records = {}
        if self.experiment_store is None:
            return
        for index, record in enumerate(self._rate_group_records()):
            item_id = f"group_{index}"
            self.rate_group_records[item_id] = record
            sample_names = record["sample_names"]
            self.rate_groups_tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    record["material"],
                    record["target"],
                    "" if record["pressure"] is None else f"{float(record['pressure']):g}",
                    "" if record["sccm"] is None else f"{float(record['sccm']):g}",
                    record["date_interval"],
                    f"{float(record['rate']):.4g}",
                    record["sample_count"],
                    record["measurement_count"],
                    ", ".join(sample_names[:18]) + (" ..." if len(sample_names) > 18 else ""),
                ),
            )

    def _rate_group_records(self) -> list[dict[str, object]]:
        groups: dict[tuple[str, str, float | None, float | None], dict[str, object]] = {}
        for sample_name in self.experiment_store.sample_names(require_spectra=True):  # type: ignore[union-attr]
            sample = self.experiment_store.load_sample(sample_name)  # type: ignore[union-attr]
            if not self._sample_matches_fit_filters(sample_name, sample):
                continue
            for layer in sample.layer_estimates:
                if layer.time_min is None or layer.time_min <= 0:
                    continue
                if not layer.target or layer.rate_nm_per_min is None:
                    continue
                key = (
                    layer.material_name,
                    layer.target,
                    None if layer.pressure_mbar is None else round(float(layer.pressure_mbar), 7),
                    None if layer.sccm is None else round(float(layer.sccm), 3),
                )
                record = groups.setdefault(
                    key,
                    {
                        "key": key,
                        "material": key[0],
                        "target": key[1],
                        "pressure": key[2],
                        "sccm": key[3],
                        "rates": [],
                        "sample_names": set(),
                        "dates": [],
                        "measurement_count": 0,
                    },
                )
                record["rates"].append(float(layer.rate_nm_per_min))  # type: ignore[index, union-attr]
                record["sample_names"].add(sample_name)  # type: ignore[index, union-attr]
                if layer.deposition_date:
                    record["dates"].append(layer.deposition_date)  # type: ignore[index, union-attr]
                record["measurement_count"] = int(record["measurement_count"]) + len(sample.measurements)
        records = []
        for record in groups.values():
            sample_names = tuple(sorted(record["sample_names"]))  # type: ignore[arg-type]
            rates = np.asarray(record["rates"], dtype=float)  # type: ignore[arg-type]
            records.append(
                {
                    **record,
                    "rate": float(np.median(rates)),
                    "sample_names": sample_names,
                    "sample_count": len(sample_names),
                    "date_interval": self._format_date_interval(record["dates"]),  # type: ignore[arg-type]
                }
            )
        return sorted(
            records,
            key=lambda item: (
                str(item["material"]),
                str(item["target"]),
                float(item["pressure"] or 0.0),
                float(item["sccm"] or 0.0),
            ),
        )

    @staticmethod
    def _format_date_interval(values) -> str:
        dates = sorted({str(value)[:10] for value in values if str(value or "").strip()})
        if not dates:
            return ""
        if len(dates) == 1:
            return dates[0]
        return f"{dates[0]} to {dates[-1]}"

    def _selected_rate_group_keys(self) -> set[tuple[str, str, float | None, float | None]] | None:
        if not hasattr(self, "rate_groups_tree"):
            return None
        selected = self.rate_groups_tree.selection()
        if not selected:
            return None
        return {
            self.rate_group_records[item_id]["key"]  # type: ignore[index]
            for item_id in selected
            if item_id in self.rate_group_records
        }

    def _on_fit_filter_changed(self, *_args) -> None:
        self._populate_experiment_results_tree()
        self.refresh_rate_groups()
        if hasattr(self, "colour_distance_tree"):
            self.refresh_colour_distance_plot(redraw_only=True)

    def _selected_fit_sample_names(self) -> list[str]:
        if self.experiment_store is None:
            return []
        names: list[str] = []
        for sample_name in self.experiment_store.sample_names(require_spectra=True):
            sample = self.experiment_store.load_sample(sample_name)
            if not sample.layer_estimates or not sample.measurements:
                continue
            if self._sample_matches_fit_filters(sample_name, sample):
                names.append(sample_name)
        limit = max(int(self.fit_sample_limit_var.get()), 0)
        if limit:
            names = names[:limit]
        return names

    def _selected_fit_measurement_pairs(self) -> list[tuple[str, int]]:
        if self.experiment_store is None:
            return []
        pairs: list[tuple[str, int]] = []
        for sample_name in self._selected_fit_sample_names():
            sample = self.experiment_store.load_sample(sample_name)
            for measurement_index in self._fit_measurement_indices(sample_name, sample):
                pairs.append((sample_name, measurement_index))
        return pairs

    def _sample_matches_fit_filters(self, sample_name: str, sample) -> bool:
        if not self._sample_matches_composition(sample):
            return False
        return bool(self._fit_measurement_indices(sample_name, sample))

    def _fit_measurement_indices(self, sample_name: str, sample) -> list[int]:
        return [
            index
            for index, measurement in enumerate(sample.measurements)
            if self._measurement_matches_experiment_sorting(sample_name, measurement)
        ]

    def _sample_matches_composition(self, sample) -> bool:
        composition = self.fit_composition_filter_var.get()
        materials = [layer.material_name for layer in sample.layer_estimates]
        material_set = set(materials)
        if composition == "All":
            return True
        if composition == "Single layer":
            return len(material_set) == 1 and len(materials) == 1
        if composition.endswith(" single layer"):
            material_name = composition.removesuffix(" single layer")
            return len(materials) == 1 and material_set == {material_name}
        if composition == "Multilayer":
            return len(materials) > 1
        if composition == "Ag containing":
            return "Ag" in material_set or "Au" in material_set
        if composition == "TiO2/SiO2/Ag":
            return {"TiO2", "SiO2", "Ag"}.issubset(material_set)
        if composition.endswith(" only"):
            material_name = composition.removesuffix(" only")
            return material_set == {material_name}
        return True

    def _on_experiment_sample_selected(self, *_args) -> None:
        if self.experiment_store is None:
            return
        try:
            sample = self.experiment_store.load_sample(self.experiment_sample_var.get())
            self.current_experiment_sample = sample
            labels = [
                f"{index + 1}: {measurement.description}"
                for index, measurement in enumerate(sample.measurements)
            ]
            if labels:
                self.experiment_measurement_var.set(labels[0])
                self.experiment_info_var.set(
                    f"{sample.sample_name}: {sample.stack_label}; {len(labels)} linked spectra."
                )
            else:
                self.experiment_measurement_var.set("")
                self.experiment_info_var.set(f"{sample.sample_name}: no linked spectra found.")
        except Exception as exc:
            self.experiment_info_var.set(f"Could not load sample: {exc}")

    def _experiment_cache_path(self) -> Path:
        base_path = default_experiment_cache_path(Path(__file__).resolve().parent)
        profile = getattr(self, "material_profile_var", tk.StringVar(value="current")).get()
        metric = self._current_colour_metric()
        suffix_parts: list[str] = []
        if profile != "current":
            suffix_parts.append(profile)
        if metric != COLOUR_METRIC_CIE76:
            suffix_parts.append(metric)
        if not suffix_parts:
            return base_path
        return base_path.with_name(f"{base_path.stem}_{'_'.join(suffix_parts)}{base_path.suffix}")

    def _current_colour_metric(self) -> str:
        variable = getattr(self, "colour_metric_var", None)
        return normalise_colour_metric(variable.get() if variable is not None else COLOUR_METRIC_CIE76)

    def _current_colour_metric_label(self) -> str:
        return colour_metric_label(self._current_colour_metric())

    def _delta_e_label(self) -> str:
        return "Delta E00" if self._current_colour_metric() == COLOUR_METRIC_CIEDE2000 else "Delta E*"

    def _colour_delta_e(self, xyz_1, xyz_2) -> float:
        return delta_e_colour(xyz_1, xyz_2, metric=self._current_colour_metric())

    def _material_profile_choices(self) -> tuple[str, ...]:
        choices = [
            profile
            for profile in material_profile_names()
            if (
                profile not in {"fitted_single_films", "best_refractiveindex_candidates"}
                or (
                    profile == "fitted_single_films"
                    and self.fitted_constants_path.exists()
                )
                or (
                    profile == "best_refractiveindex_candidates"
                    and self.best_candidate_profile_path.exists()
                )
            )
        ]
        choices.extend(self._group_candidate_profile_names())
        return tuple(dict.fromkeys(choices))

    def _group_candidate_profile_names(self) -> list[str]:
        output_dir = self.best_candidate_profile_path.parent
        if not output_dir.exists():
            return []
        names: list[str] = []
        prefix = "best_refractiveindex_candidates_"
        for path in sorted(output_dir.glob(f"{prefix}*.json")):
            suffix = path.stem.removeprefix(prefix)
            if suffix:
                names.append(f"best_candidates_{suffix}")
        return names

    def _refresh_material_profile_choices(self) -> None:
        choices = self._material_profile_choices()
        if hasattr(self, "profile_combo"):
            self.profile_combo.configure(values=choices)
        if hasattr(self, "experiment_profile_combo"):
            self.experiment_profile_combo.configure(values=choices)

    def _group_candidate_profile_path_from_name(self, profile_name: str) -> Path:
        suffix = profile_name.removeprefix("best_candidates_")
        return grouped_best_candidate_profile_path(Path(__file__).resolve().parent, suffix)

    def _missing_material_profile_message(self, profile: str, expected_path: Path) -> str:
        if profile == "fitted_single_films":
            action = "Run 'Fit from single films' first."
        elif profile == "best_refractiveindex_candidates":
            action = "Run 'Fit refractiveindex.info candidates' first."
        else:
            action = "Set the experiment filters and run 'Fit refractiveindex.info candidates' first."
        return f"{profile} has not been created yet. {action}\nExpected file: {expected_path}"

    def _fallback_to_current_profile(self, message: str, show_warning: bool) -> None:
        self.material_profile_var.set("current")
        self.materials = built_in_materials("current")
        if hasattr(self, "status_var"):
            self.status_var.set(message)
        if show_warning:
            messagebox.showwarning("Material profile", message)

    def _clear_experiment_results_tree(self) -> None:
        if not hasattr(self, "experiment_results_tree"):
            return
        for item in self.experiment_results_tree.get_children():
            self.experiment_results_tree.delete(item)

    def _progress_widgets(self) -> list[ttk.Progressbar]:
        widgets: list[ttk.Progressbar] = []
        for name in ("experiment_progress", "global_progress"):
            widget = getattr(self, name, None)
            if widget is not None:
                widgets.append(widget)
        return widgets

    def _start_busy(self, message: str) -> None:
        self.status_var.set(message)
        self.busy_text_var.set(f"Calculating: {message}")
        for progress_bar in self._progress_widgets():
            progress_bar.configure(mode="indeterminate")
            progress_bar.start(12)
        if hasattr(self, "pause_button"):
            self.pause_button.configure(state=tk.NORMAL, text="Pause")
        if hasattr(self, "abort_button"):
            self.abort_button.configure(state=tk.NORMAL)
        self.root.update_idletasks()

    def _stop_busy(self, message: str = "Ready") -> None:
        for progress_bar in self._progress_widgets():
            progress_bar.stop()
            progress_bar.configure(mode="determinate")
            progress_bar["value"] = 0
        if hasattr(self, "pause_button"):
            self.pause_button.configure(state=tk.DISABLED, text="Pause")
        if hasattr(self, "abort_button"):
            self.abort_button.configure(state=tk.DISABLED)
        self.pause_requested.clear()
        self.abort_requested.clear()
        self.busy_text_var.set("")
        self.status_var.set(message)
        self.root.update_idletasks()

    def _start_progress(self, message: str, maximum: int) -> None:
        self.status_var.set(message)
        self.busy_text_var.set(f"Caching: {message}")
        for progress_bar in self._progress_widgets():
            progress_bar.stop()
            progress_bar.configure(mode="determinate", maximum=max(int(maximum), 1))
            progress_bar["value"] = 0
        if hasattr(self, "pause_button"):
            self.pause_button.configure(state=tk.NORMAL, text="Pause")
        if hasattr(self, "abort_button"):
            self.abort_button.configure(state=tk.NORMAL)
        self.root.update_idletasks()

    def _update_progress(self, value: int, message: str) -> None:
        self.status_var.set(message)
        self.busy_text_var.set(f"Caching: {message}")
        for progress_bar in self._progress_widgets():
            numeric_value = max(int(value), 0)
            try:
                maximum = int(float(progress_bar["maximum"]))
            except Exception:
                maximum = numeric_value
            if numeric_value > maximum:
                progress_bar.configure(maximum=numeric_value)
            progress_bar["value"] = numeric_value
        self.root.update_idletasks()

    def toggle_pause_calculation(self) -> None:
        if not self.background_task_running:
            return
        if self.pause_requested.is_set():
            self.pause_requested.clear()
            self.pause_button.configure(text="Pause")
            self.busy_text_var.set("Caching: resumed")
        else:
            self.pause_requested.set()
            self.pause_button.configure(text="Resume")
            self.busy_text_var.set("Paused after current trial")

    def abort_calculation(self) -> None:
        if not self.background_task_running:
            return
        self.abort_requested.set()
        self.pause_requested.clear()
        if hasattr(self, "pause_button"):
            self.pause_button.configure(text="Pause", state=tk.DISABLED)
        if hasattr(self, "abort_button"):
            self.abort_button.configure(state=tk.DISABLED)
        self.busy_text_var.set("Aborting at next checkpoint")

    def _wait_if_paused(self, progress) -> None:
        while self.pause_requested.is_set():
            if self.abort_requested.is_set():
                raise InterruptedError("Calculation aborted by user.")
            progress(
                int(self.experiment_progress["value"]) if hasattr(self, "experiment_progress") else 0,
                "paused - press Resume to continue",
            )
            self.pause_requested.wait(0.25)
        if self.abort_requested.is_set():
            raise InterruptedError("Calculation aborted by user.")

    def _run_background(
        self,
        task,
        on_success,
        title: str,
        busy_message: str,
        progress_max: int | None = None,
    ) -> None:
        if self.background_task_running:
            messagebox.showinfo(title, "A calculation is already running.")
            return

        self.background_task_running = True
        self.pause_requested.clear()
        self.abort_requested.clear()
        events: queue.Queue = queue.Queue()

        def progress(value: int, message: str) -> None:
            events.put(("progress", int(value), message))

        def worker() -> None:
            try:
                events.put(("done", task(progress)))
            except InterruptedError as exc:
                events.put(("aborted", str(exc)))
            except Exception as exc:
                events.put(("error", exc))

        if progress_max is None:
            self._start_busy(busy_message)
        else:
            self._start_progress(busy_message, progress_max)
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, lambda: self._poll_background(events, on_success, title))

    def _poll_background(self, events: queue.Queue, on_success, title: str) -> None:
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "progress":
                self._update_progress(event[1], event[2])
            elif kind == "done":
                self.background_task_running = False
                try:
                    message = on_success(event[1])
                except Exception as exc:
                    self._stop_busy(f"{title} display failed.")
                    messagebox.showerror(title, str(exc))
                    return
                self._stop_busy(message or "Done")
                return
            elif kind == "error":
                self.background_task_running = False
                self._stop_busy(f"{title} failed.")
                messagebox.showerror(title, str(event[1]))
                return
            elif kind == "aborted":
                self.background_task_running = False
                self._stop_busy(f"{title} aborted.")
                self.experiment_info_var.set(str(event[1]) or "Calculation aborted by user.")
                return

        self.root.after(80, lambda: self._poll_background(events, on_success, title))

    def _draw_progress_dog(self, canvas: tk.Canvas | None = None) -> None:
        if canvas is None:
            canvas = self.experiment_progress_dog_canvas
        canvas.delete("all")
        canvas.create_oval(8, 4, 38, 28, fill="#ffffff", outline="#9fb3c8", width=1.2)
        canvas.create_arc(8, 4, 38, 28, start=30, extent=120, outline="#256f7f", width=3)
        canvas.create_arc(8, 4, 38, 28, start=210, extent=80, outline="#f97316", width=3)
        canvas.create_text(23, 16, text="V3", fill="#1f2933", font=("Segoe UI Semibold", 8))
        canvas.create_line(42, 9, 51, 9, fill="#9fb3c8", width=1.2)
        canvas.create_line(42, 16, 53, 16, fill="#256f7f", width=1.5)
        canvas.create_line(42, 23, 49, 23, fill="#f97316", width=1.5)

    def build_experiment_cache(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Experiments", "Load experiment data first.")
            return
        if self.background_task_running:
            messagebox.showinfo("Experiments", "A calculation is already running.")
            return
        try:
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            self.experiment_cache_path = self._experiment_cache_path()
            store = self.experiment_store
            materials = dict(self.materials)
            model = self._model_from_controls()
            angle_deg = float(self.angle_var.get())
            substrate_name = self.substrate_var.get()
            colour_metric = self._current_colour_metric()
            use_effective_interfaces = self._use_effective_interfaces()
            interface_thickness_nm = float(self.roughness_thickness_var.get())
            interface_fraction = float(self.roughness_fraction_var.get())
            native_oxide_enabled = bool(self.native_oxide_enabled_var.get())
            native_oxide_thickness_nm = float(self.native_oxide_thickness_var.get())
            cache_path = self.experiment_cache_path

            def native_oxide_for_name(name: str) -> NativeOxide | None:
                if not native_oxide_enabled:
                    return None
                default_oxide = native_oxide_for_substrate(materials, name)
                if default_oxide is None:
                    return None
                return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

            def task(_progress):
                cache = store.build_cached_results(
                    materials=materials,
                    model=model,
                    wavelengths_nm=wavelengths_nm,
                    angle_deg=angle_deg,
                    substrate_name=substrate_name,
                    native_oxide_factory=native_oxide_for_name,
                    use_effective_interfaces=use_effective_interfaces,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                    max_measurements_per_sample=None,
                    colour_metric=colour_metric,
                )
                save_cached_results(cache, cache_path)
                return cache

            def on_success(cache: CachedExperimentResults) -> str:
                self.experiment_cache = cache
                self._refresh_experiment_filter_choices()
                self._populate_experiment_results_tree()
                if hasattr(self, "colour_distance_tree"):
                    self.refresh_colour_distance_plot(redraw_only=True)
                self.plots_before_points_cache = None
                self.plots_after_points_cache = None
                self.experiment_info_var.set(
                    f"Saved {cache.count} measurement comparisons to {cache_path}"
                )
                return f"Saved {cache.count} experiment comparisons."

            self.experiment_info_var.set("Calculating experiment cache...")
            self._run_background(
                task,
                on_success,
                title="Experiments",
                busy_message="building experiment cache",
            )
        except Exception as exc:
            messagebox.showerror("Experiments", str(exc))

    def benchmark_all_models_and_constants(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Benchmark", "Load experiment data first.")
            return
        if self.background_task_running:
            messagebox.showinfo("Benchmark", "A calculation is already running.")
            return

        profiles: list[tuple[str, dict[str, Material]]] = []
        for profile in self._material_profile_choices():
            try:
                profiles.append((profile, self._materials_for_profile(profile)))
            except Exception:
                continue
        if not profiles:
            messagebox.showerror("Benchmark", "No constants profiles could be loaded.")
            return

        model_labels = self._optical_model_labels()
        sample_names = list(self.experiment_store.sample_names(require_spectra=True))
        total_measurements = sum(
            len(self.experiment_store.load_sample(name).measurements)
            for name in sample_names
        )
        total_steps = max(len(profiles) * len(model_labels) * total_measurements, 1)

        store = self.experiment_store
        default_substrate = self.substrate_var.get()
        angle_deg = float(self.angle_var.get())
        interface_thickness_nm = float(self.roughness_thickness_var.get())
        interface_fraction = float(self.roughness_fraction_var.get())
        native_oxide_enabled = bool(self.native_oxide_enabled_var.get())
        native_oxide_thickness_nm = float(self.native_oxide_thickness_var.get())
        roughness_settings = {
            "rms_roughness_nm": float(self.rms_roughness_var.get()),
            "scatter_scale": float(self.scatter_scale_var.get()),
            "scatter_exponent": float(self.scatter_exponent_var.get()),
            "max_scatter_fraction": float(self.scatter_max_var.get()),
        }
        colour_metric = self._current_colour_metric()
        wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
        output_dir = Path(__file__).resolve().parent / "outputs" / "model_constant_benchmark"

        def native_oxide_for_name(materials: dict[str, Material], substrate_name: str) -> NativeOxide | None:
            if not native_oxide_enabled:
                return None
            default_oxide = native_oxide_for_substrate(materials, substrate_name)
            if default_oxide is None:
                return None
            return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

        def task(progress):
            rows: list[dict[str, object]] = []
            done = 0
            for profile_name, materials in profiles:
                for model_label in model_labels:
                    model = self._model_for_label(model_label, roughness_settings)
                    use_effective = self._use_effective_interfaces_for_label(model_label)
                    for sample_name in sample_names:
                        self._wait_if_paused(progress)
                        sample = store.load_sample(sample_name)
                        if not sample.layer_estimates or not sample.measurements:
                            continue
                        for measurement_index, measurement in enumerate(sample.measurements):
                            done += 1
                            progress(
                                done,
                                f"{profile_name}; {model_label}; {sample_name} "
                                f"({done:,}/{total_steps:,})",
                            )
                            substrate = measurement.substrate_hint or default_substrate
                            try:
                                comparison = store.compare_sample(
                                    sample_name=sample_name,
                                    measurement_index=measurement_index,
                                    materials=materials,
                                    model=model,
                                    wavelengths_nm=wavelengths_nm,
                                    angle_deg=angle_deg,
                                    substrate_name=substrate,
                                    native_oxide=native_oxide_for_name(materials, substrate),
                                    use_effective_interfaces=use_effective,
                                    interface_thickness_nm=interface_thickness_nm,
                                    interface_fraction=interface_fraction,
                                )
                            except Exception as exc:
                                rows.append(
                                    {
                                        "sample_name": sample_name,
                                        "measurement_index": measurement_index,
                                        "measurement_description": measurement.description,
                                        "constants_profile": profile_name,
                                        "optical_model": model_label,
                                        "error": str(exc),
                                    }
                                )
                                continue
                            rows.append(
                                {
                                    "sample_name": sample_name,
                                    "series": sample_series_from_name(sample_name),
                                    "measurement_index": measurement_index,
                                    "measurement_description": measurement.description,
                                    "surface": measurement.surface_class,
                                    "substrate": measurement.substrate_group or measurement.substrate_hint or substrate,
                                    "experiment_group": self._experiment_group_label(
                                        measurement.surface_class,
                                        measurement.substrate_group or measurement.substrate_hint or substrate,
                                    ),
                                    "measurement_kind": measurement.measurement_kind,
                                    "source_system": measurement.source_system,
                                    "constants_profile": profile_name,
                                    "optical_model": model_label,
                                    "angle_deg": angle_deg,
                                    "interface_nm": interface_thickness_nm if use_effective else 0.0,
                                    "mix_fraction": interface_fraction if use_effective else np.nan,
                                    "rms_roughness_nm": roughness_settings["rms_roughness_nm"],
                                    "scatter_scale": roughness_settings["scatter_scale"],
                                    "scatter_exponent": roughness_settings["scatter_exponent"],
                                    "max_scatter_fraction": roughness_settings["max_scatter_fraction"],
                                    "colour_metric": colour_metric,
                                    "delta_e": delta_e_colour(
                                        comparison.measured_color.xyz,
                                        comparison.simulated_color.xyz,
                                        metric=colour_metric,
                                    ),
                                    "measured_hex": comparison.measured_color.hex,
                                    "simulated_hex": comparison.simulated_color.hex,
                                    "error": "",
                                }
                            )

            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = output_dir / f"model_constant_delta_e_{stamp}.csv"
            summary_path = output_dir / f"model_constant_delta_e_summary_{stamp}.csv"
            plot_path = output_dir / f"model_constant_delta_e_summary_{stamp}.png"
            raw = pd.DataFrame(rows)
            raw.to_csv(raw_path, index=False)
            valid = raw[raw["error"].fillna("").eq("")].copy() if not raw.empty else raw
            if valid.empty:
                raise ValueError("No benchmark comparisons could be calculated.")
            overall = (
                valid.assign(experiment_group="All samples")
                .groupby(["experiment_group", "constants_profile", "optical_model"], dropna=False)
                .agg(
                    mean_delta_e=("delta_e", "mean"),
                    median_delta_e=("delta_e", "median"),
                    max_delta_e=("delta_e", "max"),
                    count=("delta_e", "count"),
                )
                .reset_index()
            )
            by_group = (
                valid.groupby(["experiment_group", "constants_profile", "optical_model"], dropna=False)
                .agg(
                    mean_delta_e=("delta_e", "mean"),
                    median_delta_e=("delta_e", "median"),
                    max_delta_e=("delta_e", "max"),
                    count=("delta_e", "count"),
                )
                .reset_index()
            )
            summary = pd.concat([overall, by_group], ignore_index=True).sort_values("mean_delta_e")
            summary.to_csv(summary_path, index=False)
            self._save_benchmark_summary_plot(summary, plot_path)
            return {
                "raw": raw,
                "summary": summary,
                "raw_path": raw_path,
                "summary_path": summary_path,
                "plot_path": plot_path,
            }

        def on_success(result: dict[str, object]) -> str:
            self._draw_benchmark_summary(result["summary"], result["plot_path"])  # type: ignore[index]
            best = result["summary"].iloc[0]  # type: ignore[index]
            self.experiment_info_var.set(
                "Benchmark saved: "
                f"{result['summary_path']} | best mean {self._delta_e_label()} "
                f"{best['mean_delta_e']:.2f} ({best['constants_profile']} / {best['optical_model']})"
            )
            return "Benchmark finished."

        self._run_background(
            task,
            on_success,
            title="Benchmark",
            busy_message="benchmarking models and constants",
            progress_max=total_steps,
        )

    def _cache_with_current_substrate_groups(self, cache: CachedExperimentResults) -> CachedExperimentResults:
        if self.experiment_store is None or cache.count == 0:
            return cache
        groups: list[str] = []
        changed = False
        for sample_name, description, old_group in zip(
            cache.sample_names.astype(str),
            cache.measurement_descriptions.astype(str),
            cache.substrate_classes.astype(str),
        ):
            try:
                group = self.experiment_store.substrate_group_for_measurement(
                    sample_name,
                    description,
                    old_group,
                )
            except Exception:
                group = old_group
            group = str(group or old_group)
            groups.append(group)
            if group != old_group:
                changed = True
        if not changed:
            return cache
        return replace(cache, substrate_classes=np.asarray(groups, dtype=str))

    def load_experiment_cache(self) -> None:
        try:
            self.experiment_cache_path = self._experiment_cache_path()
            if not self.experiment_cache_path.exists():
                raise FileNotFoundError(self.experiment_cache_path)
            cache = load_cached_results(self.experiment_cache_path)
            if normalise_colour_metric(cache.colour_metric) != self._current_colour_metric():
                raise ValueError(
                    f"Saved cache uses {colour_metric_label(cache.colour_metric)}, "
                    f"but the GUI is set to {self._current_colour_metric_label()}."
                )
            self.experiment_cache = self._cache_with_current_substrate_groups(cache)
            self._refresh_experiment_filter_choices()
            self._populate_experiment_results_tree()
            if hasattr(self, "colour_distance_tree"):
                self.refresh_colour_distance_plot(redraw_only=True, hydrate_fit_colours=False)
            self.plots_before_points_cache = None
            self.plots_after_points_cache = None
            self.experiment_info_var.set(
                f"Loaded {self.experiment_cache.count} saved measurement comparisons "
                f"for {self.material_profile_var.get()} constants."
            )
        except Exception as exc:
            self.experiment_cache = None
            self._clear_experiment_results_tree()
            self.experiment_info_var.set(f"No saved experiment cache loaded: {exc}")

    def _experiment_cache_for_ml(self) -> CachedExperimentResults:
        if self.experiment_cache is not None:
            return self.experiment_cache
        self.experiment_cache_path = self._experiment_cache_path()
        if not self.experiment_cache_path.exists():
            raise FileNotFoundError(
                f"No experiment cache found at {self.experiment_cache_path}. "
                "Build / refresh saved results first."
            )
        cache = load_cached_results(self.experiment_cache_path)
        if normalise_colour_metric(cache.colour_metric) != self._current_colour_metric():
            raise ValueError(
                f"Saved cache uses {colour_metric_label(cache.colour_metric)}, "
                f"but the GUI is set to {self._current_colour_metric_label()}."
            )
        self.experiment_cache = self._cache_with_current_substrate_groups(cache)
        self._refresh_experiment_filter_choices()
        self._populate_experiment_results_tree()
        return self.experiment_cache


    def _refresh_experiment_filter_choices(self) -> None:
        if self.experiment_cache is None:
            return
        series_values = ["All"] + sorted({str(value) for value in self.experiment_cache.sample_series})
        substrate_values = ["All"] + sorted(
            {"Si", "Ti", *(str(value) for value in self.experiment_cache.substrate_classes)}
        )
        surface_values = ["All"] + sorted({str(value) for value in self.experiment_cache.surface_classes})
        kind_values = ["All"] + sorted({str(value) for value in self.experiment_cache.measurement_kinds})
        if hasattr(self, "experiment_series_filter_combo"):
            self.experiment_series_filter_combo.configure(values=series_values)
            self.experiment_substrate_filter_combo.configure(values=substrate_values)
            self.experiment_surface_filter_combo.configure(values=surface_values)
            self.experiment_kind_filter_combo.configure(values=kind_values)
        for combo_set in getattr(self, "fit_filter_combo_sets", []):
            combo_set["series"].configure(values=series_values)
            combo_set["substrate"].configure(values=substrate_values)
            combo_set["surface"].configure(values=surface_values)
            combo_set["kind"].configure(values=kind_values)
        if self.experiment_series_filter_var.get() not in series_values:
            self.experiment_series_filter_var.set("All")
        if self.experiment_substrate_filter_var.get() not in substrate_values:
            self.experiment_substrate_filter_var.set("All")
        if self.experiment_surface_filter_var.get() not in surface_values:
            self.experiment_surface_filter_var.set("All")
        if self.experiment_kind_filter_var.get() not in kind_values:
            self.experiment_kind_filter_var.set("All")

    def _on_experiment_filter_changed(self, *_args) -> None:
        self._populate_experiment_results_tree()
        self.refresh_rate_groups()
        if hasattr(self, "colour_distance_tree"):
            self.refresh_colour_distance_plot(redraw_only=True)

    def _filtered_experiment_indices(self) -> list[int]:
        if self.experiment_cache is None:
            return []
        series_filter = self.experiment_series_filter_var.get()
        substrate_filter = self.experiment_substrate_filter_var.get()
        surface_filter = self.experiment_surface_filter_var.get()
        kind_filter = self.experiment_kind_filter_var.get()
        allowed_pairs: set[tuple[str, str]] = set()
        if self.experiment_store is not None and (
            self.fit_composition_filter_var.get() != "All" or int(self.fit_sample_limit_var.get()) > 0
        ):
            for sample_name in self._selected_fit_sample_names():
                sample = self.experiment_store.load_sample(sample_name)
                for measurement_index in self._fit_measurement_indices(sample_name, sample):
                    allowed_pairs.add((sample_name, sample.measurements[measurement_index].description))
        indices: list[int] = []
        for index in range(self.experiment_cache.count):
            if series_filter != "All" and str(self.experiment_cache.sample_series[index]) != series_filter:
                continue
            if substrate_filter != "All" and str(self.experiment_cache.substrate_classes[index]) != substrate_filter:
                continue
            if surface_filter != "All" and str(self.experiment_cache.surface_classes[index]) != surface_filter:
                continue
            if kind_filter != "All" and str(self.experiment_cache.measurement_kinds[index]) != kind_filter:
                continue
            if allowed_pairs and (
                str(self.experiment_cache.sample_names[index]),
                str(self.experiment_cache.measurement_descriptions[index]),
            ) not in allowed_pairs:
                continue
            indices.append(index)
        return indices

    def _populate_experiment_results_tree(self) -> None:
        if self.experiment_cache is None:
            return
        for item in self.experiment_results_tree.get_children():
            self.experiment_results_tree.delete(item)
        visible_indices = self._filtered_experiment_indices()
        for index in visible_indices:
            sample = str(self.experiment_cache.sample_names[index])
            measurement = str(self.experiment_cache.measurement_descriptions[index])
            label = f"{sample} - {measurement}"
            measured_hex = self._rgb_tuple_to_hex(self.experiment_cache.measured_rgb[index])
            simulated_hex = self._rgb_tuple_to_hex(self.experiment_cache.simulated_rgb[index])
            self.experiment_results_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    label,
                    str(self.experiment_cache.sample_series[index]),
                    str(self.experiment_cache.substrate_classes[index]),
                    str(self.experiment_cache.surface_classes[index]),
                    str(self.experiment_cache.measurement_kinds[index]),
                    f"{self.experiment_cache.delta_e[index]:.2f}",
                    measured_hex,
                    simulated_hex,
                ),
            )
        self.experiment_info_var.set(
            f"Showing {len(visible_indices)} of {self.experiment_cache.count} cached measurements."
        )
        children = self.experiment_results_tree.get_children()
        if children:
            self.experiment_results_tree.selection_set(children[0])
            self.experiment_results_tree.focus(children[0])
            first_index = int(children[0])
            self._draw_cached_experiment_result(first_index)
            self._refresh_cached_thickness_fit_choices(first_index)

    def _on_experiment_result_selected(self, *_args) -> None:
        selection = self.experiment_results_tree.selection()
        if selection:
            index = int(selection[0])
            self._draw_cached_experiment_result(index)
            self._refresh_cached_thickness_fit_choices(index)

    def _on_experiment_tree_motion(self, event) -> None:
        if self.experiment_cache is None or self.experiment_store is None:
            self.tooltip.hide()
            self.experiment_tree_hover_item = None
            return
        region = self.experiment_results_tree.identify_region(event.x, event.y)
        item_id = self.experiment_results_tree.identify_row(event.y)
        if region not in {"cell", "tree"} or not item_id:
            self.tooltip.hide()
            self.experiment_tree_hover_item = None
            return
        self.experiment_tree_hover_item = item_id
        text = self._experiment_tree_tooltip_text(item_id)
        self._show_widget_tooltip(event, text)

    def _on_experiment_tree_leave(self, _event=None) -> None:
        self.experiment_tree_hover_item = None
        self.tooltip.hide()

    def _experiment_tree_tooltip_text(self, item_id: str) -> str:
        if self.experiment_cache is None or self.experiment_store is None:
            return ""
        try:
            index = int(item_id)
        except (TypeError, ValueError):
            return ""
        sample_name = str(self.experiment_cache.sample_names[index])
        measurement_description = str(self.experiment_cache.measurement_descriptions[index])
        try:
            sample = self.experiment_store.load_sample(sample_name)
        except Exception as exc:
            return f"{sample_name}\nCould not load sample configuration: {exc}"

        lines = [
            f"{sample_name} configuration",
            f"Series: {sample.sample_series}",
            f"Stack: {sample.stack_label}",
            (
                "Group: "
                f"{self.experiment_cache.substrate_classes[index]}, "
                f"{self.experiment_cache.surface_classes[index]}, "
                f"{self.experiment_cache.measurement_kinds[index]}"
            ),
            f"Measurement: {measurement_description}",
            "Layers, top to bottom:",
        ]
        for layer_number, layer in enumerate(sample.layer_estimates, start=1):
            lines.append(f"  {layer_number}. {self._experiment_layer_summary(layer)}")
        if not sample.layer_estimates:
            lines.append("  No layer estimates found.")
        return "\n".join(lines)

    @staticmethod
    def _experiment_layer_summary(layer) -> str:
        parts = [
            f"{layer.material_name}",
            f"{float(layer.thickness_nm):.2f} nm",
        ]
        if layer.time_min is not None:
            parts.append(f"time {float(layer.time_min):.2f} min")
        if layer.rate_nm_per_min is not None:
            parts.append(f"rate {float(layer.rate_nm_per_min):.4g} nm/min")
        if layer.target:
            parts.append(f"target {layer.target}")
        if layer.pressure_mbar is not None:
            parts.append(f"pressure {float(layer.pressure_mbar):.3g} mbar")
        if layer.sccm is not None:
            parts.append(f"{float(layer.sccm):.3g} sccm")
        if layer.deposition_date:
            parts.append(f"date {layer.deposition_date}")
        if layer.confidence:
            parts.append(f"confidence {layer.confidence}")
        return "; ".join(parts)

    def _refresh_cached_thickness_fit_choices(self, index: int) -> None:
        self.cached_thickness_fit_records = self._find_cached_thickness_fits(index)
        labels = [str(record["label"]) for record in self.cached_thickness_fit_records]
        if hasattr(self, "cached_thickness_fit_combo"):
            self.cached_thickness_fit_combo.configure(values=labels)
        if labels:
            self.cached_thickness_fit_var.set(labels[0])
            self.show_selected_cached_thickness_fit()
        else:
            self.cached_thickness_fit_var.set("")

    def _find_cached_thickness_fits(self, index: int) -> list[dict[str, object]]:
        if self.experiment_cache is None:
            return []
        sample_name = str(self.experiment_cache.sample_names[index])
        measurement_description = str(self.experiment_cache.measurement_descriptions[index])
        cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)
        if not cache_dir.exists():
            return []
        safe_sample = self._safe_cache_prefix(sample_name)
        records: list[dict[str, object]] = []
        for path in cache_dir.glob(f"{safe_sample}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                if str(metadata.get("sample_name", "")) != sample_name:
                    continue
                cached_description = str(metadata.get("measurement_description", ""))
                if cached_description != measurement_description:
                    continue
                cached_metric = normalise_colour_metric(metadata.get("colour_metric", COLOUR_METRIC_CIE76))
                if cached_metric != self._current_colour_metric():
                    continue
                best = data.get("evaluations", {}).get(data.get("best_key"), {})
                if not isinstance(best, dict):
                    continue
                delta = float(best.get("delta_e", float("nan")))
                label = (
                    f"{metadata.get('profile_name', 'unknown')} | "
                    f"{metadata.get('model_label', 'unknown')} | "
                    f"{colour_metric_label(cached_metric)} | "
                    f"+/-{float(metadata.get('range_percent_last_run', 0.0)):g}% / "
                    f"{float(metadata.get('step_percent', 0.0)):g}% step | "
                    f"{'Delta E00' if cached_metric == COLOUR_METRIC_CIEDE2000 else 'Delta E*'} {delta:.2f} | "
                    f"{path.stat().st_mtime_ns}"
                )
                display_label = label.rsplit(" | ", maxsplit=1)[0]
                records.append(
                    {
                        "label": display_label,
                        "path": path,
                        "mtime": path.stat().st_mtime,
                        "delta_e": delta,
                    }
                )
            except Exception:
                continue
        return sorted(records, key=lambda record: (float(record["delta_e"]), -float(record["mtime"])))

    def _on_cached_thickness_fit_selected(self, *_args) -> None:
        self.show_selected_cached_thickness_fit()

    def show_selected_cached_thickness_fit(self) -> None:
        label = self.cached_thickness_fit_var.get()
        record = next(
            (item for item in self.cached_thickness_fit_records if item["label"] == label),
            None,
        )
        if record is None:
            return
        try:
            result = self._load_cached_thickness_fit_result(Path(record["path"]))
            self.last_thickness_optimization_result = result
            self._draw_thickness_optimization_result(result)
            self.experiment_info_var.set(
                f"Loaded cached thickness fit from {Path(record['path']).name}: "
                f"{colour_metric_label(result.colour_metric)} {result.optimized_delta_e:.2f}, "
                f"reused {result.reused_count:,} cached trials."
            )
        except Exception as exc:
            self.experiment_info_var.set(f"Could not load cached thickness fit: {exc}")

    def refresh_colour_distance_plot(
        self,
        redraw_only: bool = False,
        hydrate_fit_colours: bool = True,
    ) -> None:
        if not hasattr(self, "colour_distance_tree"):
            return
        if self.experiment_cache is None:
            try:
                self.load_experiment_cache()
            except Exception:
                pass
        if self.experiment_cache is None:
            if not redraw_only:
                messagebox.showinfo("Colour Distance", "Build or load an experiment model cache first.")
            return
        if self.experiment_store is None:
            self.load_experiment_samples(show_errors=not redraw_only)

        rows = self._colour_distance_rows()
        self._draw_colour_distance_rows(rows, hydrate_fit_colours=hydrate_fit_colours)
        self._select_colour_distance_if_requested(redraw_only)

    def _colour_distance_rows(self) -> list[dict[str, object]]:
        if self.experiment_cache is None:
            return []
        allowed_pairs: set[tuple[str, str]] = set()
        if self.experiment_store is not None:
            for sample_name in self._selected_fit_sample_names():
                sample = self.experiment_store.load_sample(sample_name)
                for measurement_index in self._fit_measurement_indices(sample_name, sample):
                    allowed_pairs.add((sample_name, sample.measurements[measurement_index].description))

        rows: list[dict[str, object]] = []
        for index in range(self.experiment_cache.count):
            sample_name = str(self.experiment_cache.sample_names[index])
            measurement = str(self.experiment_cache.measurement_descriptions[index])
            if allowed_pairs and (sample_name, measurement) not in allowed_pairs:
                continue
            model_delta = float(self.experiment_cache.delta_e[index])
            fit_record = (
                self._best_cached_thickness_fit_for_measurement(sample_name, measurement)
                if self.colour_distance_source_var.get() == "Best cached thickness fit"
                else None
            )
            fit_delta = None
            fit_label = ""
            fit_path = None
            fit_rgb = None
            fit_stage_label = ""
            fit_profile_name = ""
            fit_model_label = ""
            if fit_record is not None:
                fit_delta = float(fit_record["delta_e"])
                fit_path = Path(fit_record["path"])
                fit_label = self._cached_fit_label_from_path(fit_path)
                fit_stage_label = str(fit_record.get("stage_label", ""))
                fit_profile_name = str(fit_record.get("profile_name", ""))
                fit_model_label = str(fit_record.get("model_label", ""))
                if fit_delta > model_delta:
                    fit_delta = None
                    fit_path = None
                    fit_label = "Cached fit not better than active before model"
                    fit_stage_label = ""
                    fit_profile_name = ""
                    fit_model_label = ""
            measured_rgb = np.clip(np.asarray(self.experiment_cache.measured_rgb[index], dtype=float), 0.0, 1.0)
            model_rgb = np.clip(np.asarray(self.experiment_cache.simulated_rgb[index], dtype=float), 0.0, 1.0)
            rows.append(
                {
                    "index": index,
                    "sample_name": sample_name,
                    "measurement": measurement,
                    "measurement_label": self._measurement_axis_label(sample_name, measurement),
                    "series": str(self.experiment_cache.sample_series[index]),
                    "substrate": str(self.experiment_cache.substrate_classes[index]),
                    "surface": str(self.experiment_cache.surface_classes[index]),
                    "kind": str(self.experiment_cache.measurement_kinds[index]),
                    "measured_rgb": measured_rgb,
                    "model_rgb": model_rgb,
                    "fit_rgb": fit_rgb,
                    "model_delta": model_delta,
                    "fit_delta": fit_delta,
                    "improvement": None if fit_delta is None else model_delta - fit_delta,
                    "fit_label": fit_label,
                    "fit_path": fit_path,
                    "fit_stage_label": fit_stage_label,
                    "fit_profile_name": fit_profile_name,
                    "fit_model_label": fit_model_label,
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                self._sample_group_sort_key(row["sample_name"]),
                str(row["surface"]),
                str(row["substrate"]),
                str(row["measurement"]),
            ),
        )

    @staticmethod
    def _measurement_axis_label(sample_name: str, measurement: str, max_len: int = 22) -> str:
        text = str(measurement or "").strip()
        for marker in (" Reflectance", " reflectance", "Transmission", "["):
            if marker in text:
                text = text.split(marker, maxsplit=1)[0].strip()
        label = f"{sample_name}: {text}" if text else sample_name
        if len(label) > max_len:
            return label[: max_len - 1] + "..."
        return label

    @staticmethod
    def _colour_distance_display_rows(rows: list[dict[str, object]], max_count: int = 80) -> list[dict[str, object]]:
        fit_rows = [row for row in rows if row["fit_delta"] is not None]
        source = fit_rows if fit_rows else rows
        ordered = sorted(
            source,
            key=lambda row: (
                ThinFilmDesignerApp._sample_group_sort_key(row["sample_name"]),
                str(row["surface"]),
                str(row["substrate"]),
                str(row["measurement"]),
            ),
        )
        return ordered[:max_count]

    def _hydrate_colour_distance_fit_colours(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row.get("fit_rgb") is not None or row.get("fit_path") is None:
                continue
            cache_key = str(row["fit_path"])
            cached = self.colour_distance_fit_colour_cache.get(cache_key)
            if cached is not None:
                row["fit_rgb"] = cached[0]
                row["fit_delta"] = cached[1]
                continue
            try:
                result = self._load_cached_thickness_fit_result(Path(row["fit_path"]))
            except Exception:
                continue
            row["fit_delta"] = float(result.optimized_delta_e)
            row["fit_rgb"] = np.clip(np.asarray(result.optimized_color.srgb, dtype=float), 0.0, 1.0)
            self.colour_distance_fit_colour_cache[cache_key] = (row["fit_rgb"], float(row["fit_delta"]))

    def refresh_plots_map_choices(self, redraw_only: bool = False) -> None:
        if not hasattr(self, "plots_map_combo"):
            return
        if self.experiment_store is None:
            self.load_experiment_samples(show_errors=not redraw_only)
        if self.experiment_cache is None:
            try:
                self.load_experiment_cache()
            except Exception:
                pass
        if self.experiment_store is None or self.experiment_cache is None:
            self.plots_map_combo.configure(values=())
            self.plots_info_var.set("Load or build experiment results before drawing plot maps.")
            return

        points = self._cached_experiment_plot_points(include_after=False, force_refresh=True)
        counts: dict[tuple[str, tuple[str, ...], str, str, str], dict[str, object]] = {}
        for point in points:
            materials = tuple(point["materials"])
            if len(materials) not in (1, 2, 3):
                continue
            if len(materials) == 3 and not self._is_opaque_ag_triple(materials, point["thicknesses"]):
                continue
            substrate_label = str(point["substrate"])
            substrate_key = str(point["substrate_key"])
            if len(materials) == 1:
                kind = "1D"
            elif len(materials) == 2:
                kind = "2D"
            else:
                kind = "3L"
            surface_label = str(point["surface"] or "unknown")
            key = (kind, materials, substrate_key, substrate_label, surface_label)
            record = counts.setdefault(key, {"measurement_count": 0, "sample_names": set()})
            record["measurement_count"] = int(record["measurement_count"]) + 1
            sample_names = record["sample_names"]
            if isinstance(sample_names, set):
                sample_names.add(str(point["sample_name"]))

        choices = [
            self._plots_choice_label(
                kind,
                materials,
                substrate_label,
                surface_label,
                int(record["measurement_count"]),
                len(record["sample_names"]) if isinstance(record["sample_names"], set) else 0,
                len(self._deposited_plot_samples(materials)),
            )
            for (kind, materials, _substrate_key, substrate_label, surface_label), record in sorted(
                counts.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][3], item[0][4]),
            )
        ]
        self.plots_map_combo.configure(values=choices)
        if choices and self.plots_map_var.get() not in choices:
            self.plots_map_var.set(choices[0])
        if choices:
            self.plots_info_var.set(
                f"Found {len(choices)} one/two/triple-layer experiment maps. "
                "Use Draw map to overlay measured colours on simulated sweep colours."
            )
        else:
            self.plots_info_var.set("No one-, two-, or supported triple-layer measured samples were found in the current cache.")

    def refresh_configuration_fit_choices(self) -> None:
        if not hasattr(self, "configuration_fit_combo"):
            return
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_cache is None:
            try:
                self.load_experiment_cache()
            except Exception:
                pass
        if self.experiment_store is None or self.experiment_cache is None:
            self.configuration_fit_combo.configure(values=())
            self.configuration_fit_info_var.set("Load or build experiment results before choosing a configuration.")
            return
        choices = self._experiment_map_choice_values()
        self.configuration_fit_combo.configure(values=choices)
        if choices and self.configuration_fit_var.get() not in choices:
            self.configuration_fit_var.set(choices[0])
        self.configuration_fit_info_var.set(
            f"Found {len(choices)} measured configurations. Pick one, then run the staged fit search."
        )

    def _experiment_map_choice_values(self) -> list[str]:
        points = self._cached_experiment_plot_points(include_after=False, force_refresh=True)
        counts: dict[tuple[str, tuple[str, ...], str, str, str], dict[str, object]] = {}
        for point in points:
            materials = tuple(point["materials"])
            if len(materials) not in (1, 2, 3):
                continue
            if len(materials) == 3 and not self._is_opaque_ag_triple(materials, point["thicknesses"]):
                continue
            if len(materials) == 1:
                kind = "1D"
            elif len(materials) == 2:
                kind = "2D"
            else:
                kind = "3L"
            substrate_label = str(point["substrate"])
            substrate_key = str(point["substrate_key"])
            surface_label = str(point["surface"] or "unknown")
            key = (kind, materials, substrate_key, substrate_label, surface_label)
            record = counts.setdefault(key, {"measurement_count": 0, "sample_names": set()})
            record["measurement_count"] = int(record["measurement_count"]) + 1
            sample_names = record["sample_names"]
            if isinstance(sample_names, set):
                sample_names.add(str(point["sample_name"]))
        return [
            self._plots_choice_label(
                kind,
                materials,
                substrate_label,
                surface_label,
                int(record["measurement_count"]),
                len(record["sample_names"]) if isinstance(record["sample_names"], set) else 0,
                len(self._deposited_plot_samples(materials)),
            )
            for (kind, materials, _substrate_key, substrate_label, surface_label), record in sorted(
                counts.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][3], item[0][4]),
            )
        ]

    @staticmethod
    def _plots_choice_label(
        kind: str,
        materials: tuple[str, ...],
        substrate_label: str,
        surface_label: str,
        measurement_count: int,
        measured_sample_count: int,
        deposited_sample_count: int | None = None,
    ) -> str:
        if len(materials) == 1:
            material_text = f"top {materials[0]}"
        elif kind == "3L" and len(materials) == 3:
            material_text = f"top {materials[0]} / middle {materials[1]} / opaque {materials[2]}"
        else:
            material_text = f"top {materials[0]} / under {materials[1]}"
        count_text = f"{measurement_count} measurements, {measured_sample_count} samples"
        if deposited_sample_count is not None:
            count_text += f", {deposited_sample_count} deposited"
        surface_text = str(surface_label or "unknown").strip()
        substrate_text = str(substrate_label or "").strip()
        if surface_text and surface_text.lower() not in {"all", "unknown", "none", "nan"}:
            location_text = f"{surface_text} {substrate_text}".strip()
        else:
            location_text = substrate_text or surface_text or "unknown substrate"
        return f"{kind}: {material_text} on {location_text} ({count_text})"

    def _parse_plots_choice(self) -> tuple[str, tuple[str, ...], str, str, str | None]:
        return self._parse_plots_choice_label(self.plots_map_var.get())

    def _parse_plots_choice_label(self, label: str) -> tuple[str, tuple[str, ...], str, str, str | None]:
        label = str(label).strip()
        if not label or ":" not in label:
            raise ValueError("Select an experiment map.")
        kind, rest = label.split(":", maxsplit=1)
        choice_text = rest.split("(", maxsplit=1)[0].strip()
        if " on " in choice_text:
            material_text, substrate_label = choice_text.rsplit(" on ", maxsplit=1)
            substrate_label = substrate_label.strip()
        else:
            material_text = choice_text
            substrate_label = self.substrate_var.get()
        materials = tuple(
            part.strip()
            .removeprefix("top ")
            .removeprefix("under ")
            .removeprefix("middle ")
            .removeprefix("opaque ")
            for part in material_text.split("/")
            if part.strip()
        )
        if kind not in {"1D", "2D", "3L"} or len(materials) not in (1, 2, 3):
            raise ValueError(f"Could not parse plot choice: {label}")
        if (kind, len(materials)) not in {("1D", 1), ("2D", 2), ("3L", 3)}:
            raise ValueError(f"Could not parse plot choice: {label}")
        surface_filter = None
        substrate_text = substrate_label
        lowered = substrate_label.lower()
        for surface in ("smooth", "rough"):
            prefix = f"{surface} "
            if lowered.startswith(prefix):
                surface_filter = surface
                substrate_text = substrate_label[len(prefix):].strip()
                break
        return kind, materials, self._plots_substrate_key(substrate_text), substrate_text, surface_filter

    @staticmethod
    def _is_opaque_ag_triple(materials: tuple[str, ...], thicknesses: object) -> bool:
        if materials != ("TiO2", "SiO2", "Ag"):
            return False
        try:
            values = tuple(float(value) for value in thicknesses)
        except Exception:
            return False
        return len(values) == 3 and values[2] >= 50.0

    def _deposited_plot_samples(self, materials: tuple[str, ...]) -> list[dict[str, object]]:
        cached = self.plots_deposited_samples_cache.get(materials)
        if cached is not None:
            return list(cached)
        if self.experiment_store is None:
            return []

        samples: list[dict[str, object]] = []
        for sample_name in self.experiment_store.sample_names(require_spectra=False):
            try:
                sample = self.experiment_store.load_sample(sample_name)
            except Exception:
                continue
            sample_materials = tuple(layer.material_name for layer in sample.layer_estimates)
            if sample_materials != materials:
                continue
            thicknesses = tuple(float(layer.thickness_nm) for layer in sample.layer_estimates)
            if len(materials) == 3 and not self._is_opaque_ag_triple(materials, thicknesses):
                continue
            samples.append(
                {
                    "sample_name": str(sample.sample_name),
                    "materials": sample_materials,
                    "thicknesses": thicknesses,
                    "measurement_count": len(sample.measurements),
                }
            )
        samples.sort(key=lambda item: str(item["sample_name"]))
        self.plots_deposited_samples_cache[materials] = samples
        return list(samples)

    def _plots_sample_coverage_lines(
        self,
        materials: tuple[str, ...],
        points: list[dict[str, object]],
        include_after: bool,
    ) -> list[str]:
        deposited_samples = self._deposited_plot_samples(materials)
        measured_sample_names = {str(point["sample_name"]) for point in points}
        measured_count = len(measured_sample_names)
        lines = [
            (
                "Deposited stack count: "
                f"{len(deposited_samples)} samples; {measured_count} shown here."
            )
        ]
        missing_measurements = [
            str(sample["sample_name"])
            for sample in deposited_samples
            if int(sample["measurement_count"]) <= 0
        ]
        hidden_after_fit = [
            str(sample["sample_name"])
            for sample in deposited_samples
            if int(sample["measurement_count"]) > 0
            and str(sample["sample_name"]) not in measured_sample_names
        ]
        if missing_measurements:
            lines.append("No loaded measurement: " + ", ".join(missing_measurements))
        if include_after and hidden_after_fit:
            lines.append("No cached after-fit point: " + ", ".join(hidden_after_fit))
        return lines

    def _plots_after_position_lines(
        self,
        points: list[dict[str, object]],
        include_after: bool,
    ) -> list[str]:
        if not include_after:
            return []
        moved_count = sum(1 for point in points if bool(point.get("thickness_moved")))
        fixed_opaque_count = sum(1 for point in points if bool(point.get("fixed_opaque_metal")))
        lines = [
            f"After-fit positions use cached optimized thicknesses; moved {moved_count}/{len(points)} points."
        ]
        if moved_count:
            lines.append("Open grey markers show before-fit positions.")
        if fixed_opaque_count:
            lines.append(
                f"Ag/Au layers >= 50 nm fixed as opaque in {fixed_opaque_count} point(s)."
            )
        return lines

    def run_configuration_fit_pipeline(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_cache is None:
            self.load_experiment_cache()
        if self.experiment_store is None or self.experiment_cache is None:
            messagebox.showerror("Configuration Fit", "Load or build experiment results first.")
            return
        if self.background_task_running:
            messagebox.showinfo("Configuration Fit", "A calculation is already running.")
            return
        try:
            if not self.configuration_fit_var.get():
                self.refresh_configuration_fit_choices()
            kind, materials_tuple, substrate_name, substrate_label, surface_filter = self._parse_plots_choice_label(
                self.configuration_fit_var.get()
            )
            points = self._configuration_fit_points(materials_tuple, substrate_name, surface_filter)
            if not points:
                raise ValueError("No measurements were found for the selected configuration.")

            profiles: list[tuple[str, dict[str, Material]]] = []
            for profile in self._material_profile_choices():
                try:
                    profiles.append((profile, self._materials_for_profile(profile)))
                except Exception:
                    continue
            if not profiles:
                raise ValueError("No constants profiles could be loaded.")

            model_candidates: list[tuple[str, str, dict[str, float]]] = []
            for model_label in self._optical_model_labels():
                model_candidates.extend(self._configuration_model_candidates(model_label))
            total_candidates = max(len(profiles) * len(model_candidates) * len(points), 1)
            thickness_steps = len(points) if bool(self.configuration_fit_run_thickness_var.get()) else 0
            progress_max = total_candidates + thickness_steps

            store = self.experiment_store
            angle_deg = float(self.angle_var.get())
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            interface_thickness_nm = float(self.roughness_thickness_var.get())
            interface_fraction = float(self.roughness_fraction_var.get())
            native_oxide_enabled = bool(self.native_oxide_enabled_var.get())
            native_oxide_thickness_nm = float(self.native_oxide_thickness_var.get())
            colour_metric = self._current_colour_metric()
            run_thickness = bool(self.configuration_fit_run_thickness_var.get())
            range_percent = float(self.thickness_opt_range_percent_var.get())
            step_percent = float(self.thickness_opt_step_percent_var.get())
            cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)
            fit_reflectance_scale = bool(self.thickness_fit_scale_enabled_var.get())
            reflectance_scale_min = float(self.thickness_fit_scale_min_var.get())
            reflectance_scale_max = float(self.thickness_fit_scale_max_var.get())
            project_root = Path(__file__).resolve().parent
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_slug = self._safe_cache_prefix(
                f"{kind}_{'_'.join(materials_tuple)}_{surface_filter or 'all'}_{substrate_label}"
            )
            output_dir = project_root / "outputs" / "configuration_fit" / f"{config_slug}_{stamp}"

            def native_oxide_for(materials: dict[str, Material], substrate: str) -> NativeOxide | None:
                if not native_oxide_enabled:
                    return None
                default_oxide = native_oxide_for_substrate(materials, substrate)
                if default_oxide is None:
                    return None
                return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

            def task(progress):
                done = 0
                candidate_rows: list[dict[str, object]] = []
                for profile_name, profile_materials in profiles:
                    for model_label, parameter_label, settings in model_candidates:
                        model = self._model_for_label(model_label, settings)
                        use_effective = self._use_effective_interfaces_for_label(model_label)
                        deltas: list[float] = []
                        for point in points:
                            self._wait_if_paused(progress)
                            done += 1
                            progress(
                                done,
                                f"{profile_name}; {model_label} {parameter_label}; "
                                f"{point['sample_name']} ({done:,}/{progress_max:,})",
                            )
                            sample_name = str(point["sample_name"])
                            measurement_index = int(point["measurement_index"])
                            measurement_substrate = str(point["substrate_key"] or substrate_name)
                            try:
                                comparison = store.compare_sample(
                                    sample_name=sample_name,
                                    measurement_index=measurement_index,
                                    materials=profile_materials,
                                    model=model,
                                    wavelengths_nm=wavelengths_nm,
                                    angle_deg=angle_deg,
                                    substrate_name=measurement_substrate,
                                    native_oxide=native_oxide_for(profile_materials, measurement_substrate),
                                    use_effective_interfaces=use_effective,
                                    interface_thickness_nm=interface_thickness_nm,
                                    interface_fraction=interface_fraction,
                                )
                                delta = delta_e_colour(
                                    comparison.measured_color.xyz,
                                    comparison.simulated_color.xyz,
                                    metric=colour_metric,
                                )
                            except Exception:
                                delta = float("nan")
                            deltas.append(float(delta))
                        finite = np.asarray([value for value in deltas if np.isfinite(value)], dtype=float)
                        if finite.size:
                            candidate_rows.append(
                                {
                                    "constants_profile": profile_name,
                                    "optical_model": model_label,
                                    "model_parameters": parameter_label,
                                    "mean_delta_e": float(np.mean(finite)),
                                    "median_delta_e": float(np.median(finite)),
                                    "max_delta_e": float(np.max(finite)),
                                    "measurement_count": int(finite.size),
                                    "settings": settings,
                                }
                            )
                if not candidate_rows:
                    raise ValueError("No constants/model candidate could be evaluated.")
                candidate_rows.sort(key=lambda row: float(row["mean_delta_e"]))
                best = candidate_rows[0]
                best_profile = str(best["constants_profile"])
                best_model_label = str(best["optical_model"])
                best_settings = dict(best.get("settings", {}))
                best_materials = self._materials_for_profile(best_profile)
                best_model = self._model_for_label(best_model_label, best_settings)
                best_use_effective = self._use_effective_interfaces_for_label(best_model_label)

                thickness_results: list[ThicknessOptimizationResult] = []
                if run_thickness:
                    for point in points:
                        self._wait_if_paused(progress)
                        done += 1
                        sample_name = str(point["sample_name"])
                        measurement_index = int(point["measurement_index"])
                        measurement_substrate = str(point["substrate_key"] or substrate_name)
                        progress(
                            min(done, progress_max),
                            f"Thickness fit {sample_name} ({done:,}/{progress_max:,})",
                        )

                        def trial_progress(_trial_done: int, _trial_total: int) -> None:
                            self._wait_if_paused(progress)

                        result = optimize_experiment_thicknesses(
                            store=store,
                            sample_name=sample_name,
                            measurement_index=measurement_index,
                            materials=best_materials,
                            model=best_model,
                            wavelengths_nm=wavelengths_nm,
                            angle_deg=angle_deg,
                            substrate_name=measurement_substrate,
                            native_oxide=native_oxide_for(best_materials, measurement_substrate),
                            use_effective_interfaces=best_use_effective,
                            interface_thickness_nm=interface_thickness_nm,
                            interface_fraction=interface_fraction,
                            range_percent=range_percent,
                            step_percent=step_percent,
                            cache_dir=cache_dir,
                            profile_name=best_profile,
                            model_label=best_model_label,
                            model_settings=best_settings,
                            group_by_material=False,
                            fixed_metal_threshold_nm=50.0,
                            colour_metric=colour_metric,
                            fit_reflectance_scale=fit_reflectance_scale,
                            reflectance_scale_min=reflectance_scale_min,
                            reflectance_scale_max=reflectance_scale_max,
                            progress_callback=trial_progress,
                        )
                        thickness_results.append(result)

                sweep_result = self._configuration_fit_sweep_result(
                    kind=kind,
                    materials_tuple=materials_tuple,
                    substrate_name=substrate_name,
                    points=points,
                    thickness_results=thickness_results,
                    materials=best_materials,
                    model=best_model,
                    use_effective_interfaces=best_use_effective,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                    angle_deg=angle_deg,
                )

                output_dir.mkdir(parents=True, exist_ok=True)
                candidate_csv = output_dir / "constants_model_candidates.csv"
                pd.DataFrame(
                    [
                        {key: value for key, value in row.items() if key != "settings"}
                        for row in candidate_rows
                    ]
                ).to_csv(candidate_csv, index=False)
                stage_rows = self._configuration_fit_stage_rows(points, candidate_rows, thickness_results)
                stage_csv = output_dir / "stage_delta_e_summary.csv"
                pd.DataFrame(stage_rows).to_csv(stage_csv, index=False)
                summary_path = output_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "configuration": self.configuration_fit_var.get(),
                            "kind": kind,
                            "materials": list(materials_tuple),
                            "substrate": substrate_label,
                            "surface": surface_filter,
                            "colour_metric": colour_metric,
                            "best_constants_profile": best_profile,
                            "best_optical_model": best_model_label,
                            "best_model_parameters": str(best["model_parameters"]),
                            "best_mean_delta_e_before_thickness": float(best["mean_delta_e"]),
                            "mean_delta_e_after_thickness": (
                                float(np.mean([result.optimized_delta_e for result in thickness_results]))
                                if thickness_results
                                else None
                            ),
                            "candidate_csv": str(candidate_csv),
                            "stage_csv": str(stage_csv),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return {
                    "kind": kind,
                    "materials": materials_tuple,
                    "substrate_name": substrate_name,
                    "substrate_label": substrate_label,
                    "surface_filter": surface_filter,
                    "points": points,
                    "candidate_rows": candidate_rows,
                    "thickness_results": thickness_results,
                    "sweep_result": sweep_result,
                    "stage_rows": stage_rows,
                    "output_dir": output_dir,
                    "summary_path": summary_path,
                }

            def on_success(payload) -> str:
                self._draw_configuration_fit_result(payload)
                self.notebook.select(self.configuration_fit_tab)
                best = payload["candidate_rows"][0]
                after = (
                    float(np.mean([result.optimized_delta_e for result in payload["thickness_results"]]))
                    if payload["thickness_results"]
                    else float(best["mean_delta_e"])
                )
                return (
                    "Configuration fit complete: "
                    f"{best['constants_profile']} / {best['optical_model']} "
                    f"mean {self._delta_e_label()} {after:.2f}."
                )

            self._run_background(
                task,
                on_success,
                title="Configuration Fit",
                busy_message="running staged configuration fit",
                progress_max=progress_max,
            )
        except Exception as exc:
            messagebox.showerror("Configuration Fit", str(exc))

    def run_empirical_refractive_index_fit(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Empirical Fit", "Load Reflectivity/sample_data first.")
            return
        try:
            pairs = self._selected_fit_measurement_pairs()
            if not pairs:
                raise ValueError("No filtered measurement rows were selected.")
            material_names = self._empirical_fit_material_names()
            if not material_names:
                raise ValueError("Choose at least one material to fit.")
            wavelengths_nm = wavelength_grid(400.0, 700.0, 61)
            project_root = Path(__file__).resolve().parent
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            material_slug = self._safe_cache_prefix("_".join(material_names))
            output_dir = project_root / "outputs" / "empirical_fit" / f"{material_slug}_{stamp}"
            base_materials = dict(self.materials)
            model = self._model_from_controls()
            angle_deg = float(self.angle_var.get())
            substrate_name = self.substrate_var.get()
            native_oxide_enabled = bool(self.native_oxide_enabled_var.get())
            native_oxide_thickness_nm = float(self.native_oxide_thickness_var.get())
            use_effective_interfaces = bool(self._use_effective_interfaces())
            interface_thickness_nm = float(self.roughness_thickness_var.get())
            interface_fraction = float(self.roughness_fraction_var.get())
            fit_k = bool(self.empirical_fit_k_var.get())
            use_thickness = bool(self.empirical_fit_thickness_dependence_var.get())
            use_time = bool(self.empirical_fit_time_dependence_var.get())
            validation_fraction = float(self.empirical_validation_fraction_var.get())
            lab_weight = float(self.empirical_lab_weight_var.get())
            max_evals = int(self.empirical_max_evals_var.get())
            colour_metric = self._current_colour_metric()

            def task(progress):
                def fit_progress(done: int, total: int, message: str) -> None:
                    self._wait_if_paused(progress)
                    progress(done, message)

                return fit_empirical_refractive_index_model(
                    store=self.experiment_store,
                    measurement_pairs=pairs,
                    base_materials=base_materials,
                    model=model,
                    wavelengths_nm=wavelengths_nm,
                    material_names=material_names,
                    angle_deg=angle_deg,
                    substrate_name=substrate_name,
                    native_oxide_enabled=native_oxide_enabled,
                    native_oxide_thickness_nm=native_oxide_thickness_nm,
                    use_effective_interfaces=use_effective_interfaces,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                    fit_k=fit_k,
                    use_thickness_dependence=use_thickness,
                    use_time_dependence=use_time,
                    validation_fraction=validation_fraction,
                    lab_weight=lab_weight,
                    max_nfev=max_evals,
                    colour_metric=colour_metric,
                    output_dir=output_dir,
                    progress_callback=fit_progress,
                )

            def on_success(result: EmpiricalFitResult) -> str:
                self._draw_empirical_fit_result(result)
                self.notebook.select(self.empirical_fit_tab)
                validation_text = (
                    f"; validation {self._delta_e_label()} "
                    f"{result.mean_validation_delta_e_after:.2f}"
                    if result.mean_validation_delta_e_after is not None
                    else ""
                )
                self.empirical_fit_info_var.set(
                    f"Saved empirical fit to {result.output_dir}. "
                    f"Train {self._delta_e_label()} {result.mean_train_delta_e_before:.2f} -> "
                    f"{result.mean_train_delta_e_after:.2f}{validation_text}."
                )
                return (
                    "Empirical n/k fit complete: "
                    f"train {self._delta_e_label()} {result.mean_train_delta_e_after:.2f}."
                )

            self.empirical_fit_info_var.set(
                f"Running empirical fit on {len(pairs)} filtered measurement rows."
            )
            self._run_background(
                task,
                on_success,
                title="Empirical Fit",
                busy_message="running empirical n/k fit",
                progress_max=max(max_evals, 1),
            )
        except Exception as exc:
            messagebox.showerror("Empirical Fit", str(exc))

    def _empirical_fit_material_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for raw in self.empirical_fit_materials_var.get().split(","):
            name = raw.strip()
            if not name:
                continue
            if name not in self.materials:
                raise ValueError(f"Material {name!r} is not available in the current constants profile.")
            if name not in names:
                names.append(name)
        return tuple(names)

    def _draw_empirical_fit_result(self, result: EmpiricalFitResult) -> None:
        figure = self.empirical_fit_figure
        figure.clear()
        grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28)
        summary_ax = figure.add_subplot(grid[0, 0])
        scatter_ax = figure.add_subplot(grid[0, 1])
        parameter_ax = figure.add_subplot(grid[1, :])

        labels = ["train before", "train after"]
        values = [result.mean_train_delta_e_before, result.mean_train_delta_e_after]
        colours = ["#8aa8bd", "#f97316"]
        if result.mean_validation_delta_e_before is not None and result.mean_validation_delta_e_after is not None:
            labels.extend(["validation before", "validation after"])
            values.extend([result.mean_validation_delta_e_before, result.mean_validation_delta_e_after])
            colours.extend(["#b7c7d4", "#fb923c"])
        summary_ax.bar(labels, values, color=colours)
        summary_ax.set_ylabel(self._delta_e_label())
        summary_ax.set_title("Mean colour error", fontsize=10, fontweight="semibold")
        summary_ax.tick_params(axis="x", rotation=20)
        summary_ax.grid(True, axis="y", alpha=0.25)

        predictions = list(result.predictions)
        for split, colour, marker in (("train", "#0f766e", "o"), ("validation", "#dc2626", "^")):
            subset = [prediction for prediction in predictions if prediction.split == split]
            if not subset:
                continue
            scatter_ax.scatter(
                [prediction.baseline_delta_e for prediction in subset],
                [prediction.fitted_delta_e for prediction in subset],
                label=split,
                color=colour,
                marker=marker,
                alpha=0.78,
                edgecolors="#ffffff",
                linewidths=0.5,
            )
        all_values = [
            value
            for prediction in predictions
            for value in (prediction.baseline_delta_e, prediction.fitted_delta_e)
            if np.isfinite(value)
        ]
        if all_values:
            limit = max(all_values) * 1.05
            scatter_ax.plot([0, limit], [0, limit], color="#111827", linestyle="--", linewidth=1)
            scatter_ax.set_xlim(0, limit)
            scatter_ax.set_ylim(0, limit)
        scatter_ax.set_xlabel(f"Before {self._delta_e_label()}")
        scatter_ax.set_ylabel(f"After {self._delta_e_label()}")
        scatter_ax.set_title("Prediction check", fontsize=10, fontweight="semibold")
        scatter_ax.legend(loc="best", fontsize=8)
        scatter_ax.grid(True, alpha=0.25)

        parameters = list(result.parameters)
        if parameters:
            labels = [f"{parameter.material_name} {parameter.parameter}" for parameter in parameters]
            values = [parameter.value for parameter in parameters]
            y_pos = np.arange(len(parameters), dtype=float)
            parameter_ax.barh(y_pos, values, color="#256f7f", alpha=0.82)
            parameter_ax.axvline(0, color="#111827", linewidth=1)
            parameter_ax.set_yticks(y_pos)
            parameter_ax.set_yticklabels(labels, fontsize=8)
            parameter_ax.set_xlabel("Fitted correction value")
            parameter_ax.set_title("Loose effective n/k parameters", fontsize=10, fontweight="semibold")
            parameter_ax.grid(True, axis="x", alpha=0.25)
        else:
            parameter_ax.text(0.5, 0.5, "No fitted parameters", ha="center", va="center")
            parameter_ax.set_axis_off()

        figure.suptitle(
            "Empirical equipment-calibrated n/k fit\n"
            f"{', '.join(result.material_names)}; train n={result.training_count}, "
            f"validation n={result.validation_count}; saved to {result.output_dir}",
            fontsize=11,
            fontweight="semibold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        self.empirical_fit_canvas.draw_idle()

    def _configuration_fit_points(
        self,
        materials_tuple: tuple[str, ...],
        substrate_name: str,
        surface_filter: str | None,
    ) -> list[dict[str, object]]:
        source_points = self._cached_experiment_plot_points(include_after=False, force_refresh=True)
        points: list[dict[str, object]] = []
        for point in source_points:
            if tuple(point["materials"]) != materials_tuple:
                continue
            if str(point["substrate_key"]) != substrate_name:
                continue
            if surface_filter is not None and str(point["surface"]) != surface_filter:
                continue
            try:
                measurement_index = self._measurement_index_for_description(
                    str(point["sample_name"]),
                    str(point["measurement"]),
                )
            except Exception:
                continue
            enriched = dict(point)
            enriched["measurement_index"] = int(measurement_index)
            points.append(enriched)
        return points

    def _measurement_index_for_description(self, sample_name: str, measurement_description: str) -> int:
        if self.experiment_store is None:
            raise ValueError("Load experiment data first.")
        sample = self.experiment_store.load_sample(sample_name)
        for index, measurement in enumerate(sample.measurements):
            if measurement.description == measurement_description:
                return index
        raise ValueError(f"Measurement {measurement_description!r} was not found for {sample_name}.")

    def _configuration_model_candidates(self, model_label: str) -> list[tuple[str, str, dict[str, float]]]:
        base = {
            "rms_roughness_nm": float(self.rms_roughness_var.get()),
            "scatter_scale": float(self.scatter_scale_var.get()),
            "scatter_exponent": float(self.scatter_exponent_var.get()),
            "max_scatter_fraction": float(self.scatter_max_var.get()),
        }
        lowered = model_label.lower()
        if "rms" not in lowered and "diffuse redistribution" not in lowered:
            return [(model_label, "default", {})]
        candidates: list[tuple[str, str, dict[str, float]]] = []
        rms = max(float(base["rms_roughness_nm"]), 0.0)
        rms_values = sorted({0.0, rms, max(rms * 0.5, 0.0), max(rms * 2.0, 0.1)})
        for value in rms_values:
            settings = dict(base)
            settings["rms_roughness_nm"] = value
            candidates.append((model_label, f"RMS {value:.3g} nm", settings))
        if "diffuse redistribution" in lowered:
            for scale in sorted(
                {base["scatter_scale"], max(base["scatter_scale"] * 0.5, 0.0), base["scatter_scale"] * 1.5}
            ):
                settings = dict(base)
                settings["scatter_scale"] = float(scale)
                candidates.append((model_label, f"scatter scale {scale:.3g}", settings))
            for max_fraction in sorted({base["max_scatter_fraction"], 0.5, 0.85, 1.0}):
                settings = dict(base)
                settings["max_scatter_fraction"] = float(max_fraction)
                candidates.append((model_label, f"max scatter {max_fraction:.3g}", settings))
        unique: dict[tuple[str, tuple[tuple[str, float], ...]], tuple[str, str, dict[str, float]]] = {}
        for label, parameter_label, settings in candidates:
            key = (
                parameter_label,
                tuple(sorted((name, round(float(value), 6)) for name, value in settings.items())),
            )
            unique[key] = (label, parameter_label, settings)
        return list(unique.values())

    def _configuration_fit_sweep_result(
        self,
        *,
        kind: str,
        materials_tuple: tuple[str, ...],
        substrate_name: str,
        points: list[dict[str, object]],
        thickness_results: list[ThicknessOptimizationResult],
        materials: dict[str, Material],
        model,
        use_effective_interfaces: bool,
        interface_thickness_nm: float,
        interface_fraction: float,
        angle_deg: float,
    ):
        point_thicknesses = [tuple(float(value) for value in point["thicknesses"]) for point in points]
        if thickness_results:
            optimized_by_key = {
                (result.sample_name, result.measurement_description): tuple(
                    float(layer.optimized_thickness_nm) for layer in result.layer_results
                )
                for result in thickness_results
            }
            point_thicknesses = [
                optimized_by_key.get((str(point["sample_name"]), str(point["measurement"])), values)
                for point, values in zip(points, point_thicknesses)
            ]
        columns = list(zip(*point_thicknesses)) if point_thicknesses else []
        if not columns:
            raise ValueError("No thickness coordinates are available for the selected configuration.")

        def bounds(values: tuple[float, ...]) -> tuple[float, float]:
            span = max(max(values) - min(values), 20.0)
            return max(0.0, min(values) - 0.12 * span), max(values) + 0.12 * span

        def stack_for(thicknesses_nm: tuple[float, ...]):
            substrate = materials[substrate_name]
            native_oxide = None
            if bool(self.native_oxide_enabled_var.get()):
                default_oxide = native_oxide_for_substrate(materials, substrate_name)
                if default_oxide is not None:
                    native_oxide = NativeOxide(default_oxide.material, float(self.native_oxide_thickness_var.get()))
            deposited_layers = [
                Layer(materials[material], float(thickness))
                for material, thickness in zip(materials_tuple, thicknesses_nm)
            ]
            if use_effective_interfaces:
                return make_stack_with_interfaces(
                    incident_medium=materials["air"],
                    deposited_layers=deposited_layers,
                    substrate=substrate,
                    native_oxide=native_oxide,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                    name="configuration fit sweep",
                )
            layers = list(deposited_layers)
            if native_oxide is not None:
                layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
            return make_stack(
                incident_medium=materials["air"],
                substrate=substrate,
                layers=layers,
                name="configuration fit sweep",
                display_layers=deposited_layers,
            )

        if kind == "1D":
            x_min, x_max = bounds(tuple(float(value) for value in columns[0]))
            return run_thickness_sweep_1d(
                stack=stack_for((max(x_min, 1.0),)),
                model=model,
                layer=materials_tuple[0],
                thickness_min_nm=x_min,
                thickness_max_nm=x_max,
                angle_deg=angle_deg,
                num_points=int(self.sweep_points_1d_var.get()),
                quality=self.sweep_quality_var.get(),
            )
        x_min, x_max = bounds(tuple(float(value) for value in columns[0]))
        y_min, y_max = bounds(tuple(float(value) for value in columns[1]))
        stack_thicknesses = (max(x_min, 1.0), max(y_min, 1.0))
        if kind == "3L":
            stack_thicknesses = (stack_thicknesses[0], stack_thicknesses[1], 1000.0)
        return run_thickness_sweep_2d(
            stack=stack_for(stack_thicknesses),
            model=model,
            layer_1=materials_tuple[0],
            layer_2=materials_tuple[1],
            thickness_1_min_nm=x_min,
            thickness_1_max_nm=x_max,
            thickness_2_min_nm=y_min,
            thickness_2_max_nm=y_max,
            angle_deg=angle_deg,
            num_points_1=int(self.sweep_points_2d_var.get()),
            num_points_2=int(self.sweep_points_2d_var.get()),
            quality=self.sweep_quality_var.get(),
            layer_2_occurrence=1 if materials_tuple[0] == materials_tuple[1] else 0,
        )

    @staticmethod
    def _configuration_fit_stage_rows(
        points: list[dict[str, object]],
        candidate_rows: list[dict[str, object]],
        thickness_results: list[ThicknessOptimizationResult],
    ) -> list[dict[str, object]]:
        rows = [
            {
                "stage": "Best constants + model",
                "mean_delta_e": float(candidate_rows[0]["mean_delta_e"]) if candidate_rows else float("nan"),
                "measurement_count": int(candidate_rows[0]["measurement_count"]) if candidate_rows else len(points),
            }
        ]
        if thickness_results:
            rows.append(
                {
                    "stage": "Individual thickness fit",
                    "mean_delta_e": float(np.mean([result.optimized_delta_e for result in thickness_results])),
                    "measurement_count": len(thickness_results),
                }
            )
        return rows

    def _draw_configuration_fit_result(self, payload: dict[str, object]) -> None:
        points = list(payload["points"])
        thickness_results = list(payload["thickness_results"])
        result_by_key = {
            (result.sample_name, result.measurement_description): result
            for result in thickness_results
        }
        candidate_rows = list(payload["candidate_rows"])
        stage_rows = list(payload["stage_rows"])
        sweep_result = payload["sweep_result"]
        materials_tuple = tuple(payload["materials"])
        kind = str(payload["kind"])
        delta_label = self._delta_e_label()

        self.configuration_fit_figure.clear()
        grid = self.configuration_fit_figure.add_gridspec(
            2,
            2,
            width_ratios=[1.45, 1.0],
            height_ratios=[1.25, 1.0],
            hspace=0.42,
            wspace=0.30,
        )
        map_ax = self.configuration_fit_figure.add_subplot(grid[:, 0])
        stage_ax = self.configuration_fit_figure.add_subplot(grid[0, 1])
        impact_ax = self.configuration_fit_figure.add_subplot(grid[1, 1])

        if kind == "1D":
            strip = np.repeat(sweep_result.rgb_values[np.newaxis, :, :], 24, axis=0)
            map_ax.imshow(
                strip,
                aspect="auto",
                origin="lower",
                extent=[
                    float(sweep_result.thickness_values_nm[0]),
                    float(sweep_result.thickness_values_nm[-1]),
                    0.0,
                    1.0,
                ],
            )
            for index, point in enumerate(points):
                key = (str(point["sample_name"]), str(point["measurement"]))
                thicknesses = tuple(float(value) for value in point["thicknesses"])
                if key in result_by_key:
                    thicknesses = tuple(float(layer.optimized_thickness_nm) for layer in result_by_key[key].layer_results)
                y_value = 0.35 + 0.3 * ((index % 7) / 6.0)
                map_ax.scatter([thicknesses[0]], [y_value], s=62, facecolors=[point["rgb"]], edgecolors="#111827")
            map_ax.set_yticks([])
            map_ax.set_xlabel(f"{materials_tuple[0]} thickness (nm)")
        else:
            map_ax.imshow(
                sweep_result.rgb_grid,
                aspect="auto",
                origin="lower",
                extent=[
                    float(sweep_result.thickness_values_1_nm[0]),
                    float(sweep_result.thickness_values_1_nm[-1]),
                    float(sweep_result.thickness_values_2_nm[0]),
                    float(sweep_result.thickness_values_2_nm[-1]),
                ],
            )
            marker_by_surface = {"smooth": "o", "rough": "^"}
            for point in points:
                key = (str(point["sample_name"]), str(point["measurement"]))
                thicknesses = tuple(float(value) for value in point["thicknesses"])
                if key in result_by_key:
                    thicknesses = tuple(float(layer.optimized_thickness_nm) for layer in result_by_key[key].layer_results)
                map_ax.scatter(
                    [thicknesses[0]],
                    [thicknesses[1]],
                    s=62,
                    marker=marker_by_surface.get(str(point["surface"]), "s"),
                    facecolors=[point["rgb"]],
                    edgecolors="#111827",
                    linewidths=0.8,
                    zorder=4,
                )
            y_label = "middle-layer" if kind == "3L" else "underlayer"
            map_ax.set_xlabel(f"{materials_tuple[0]} top-layer thickness (nm)")
            map_ax.set_ylabel(f"{materials_tuple[1]} {y_label} thickness (nm)")
        map_ax.grid(True, color="white", alpha=0.25)
        best = candidate_rows[0]
        after_delta = (
            float(np.mean([result.optimized_delta_e for result in thickness_results]))
            if thickness_results
            else float(best["mean_delta_e"])
        )
        map_ax.set_title(
            f"{self.configuration_fit_var.get()}\n"
            f"{best['constants_profile']} / {best['optical_model']} / {best['model_parameters']}; "
            f"mean {delta_label} {after_delta:.2f}",
            fontsize=9.5,
            fontweight="semibold",
        )

        stage_labels = [str(row["stage"]) for row in stage_rows]
        stage_values = [float(row["mean_delta_e"]) for row in stage_rows]
        stage_ax.plot(np.arange(len(stage_values)), stage_values, marker="o", color="#147d77")
        stage_ax.set_xticks(np.arange(len(stage_labels)), stage_labels, rotation=25, ha="right")
        stage_ax.set_ylabel(delta_label)
        stage_ax.set_title("Delta E minimization", fontsize=9, fontweight="semibold")
        stage_ax.grid(True, alpha=0.25)

        top = candidate_rows[:10]
        labels = [
            f"{row['constants_profile']}\n{row['optical_model']}\n{row['model_parameters']}"
            for row in top
        ]
        values = [float(row["mean_delta_e"]) for row in top]
        impact_ax.bar(np.arange(len(top)), values, color="#8aa6b8")
        impact_ax.set_xticks(np.arange(len(top)), labels, rotation=70, ha="right", fontsize=5.7)
        impact_ax.set_ylabel(delta_label)
        impact_ax.set_title("Parameter impact: best candidates", fontsize=9, fontweight="semibold")
        impact_ax.grid(True, axis="y", alpha=0.25)

        output_dir = Path(payload["output_dir"])
        self.configuration_fit_info_var.set(
            f"Saved configuration fit to {output_dir}. "
            f"Best before thickness {delta_label} {float(best['mean_delta_e']):.2f}; after {after_delta:.2f}."
        )
        self.configuration_fit_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90)
        self.configuration_fit_canvas.draw_idle()
        try:
            self.configuration_fit_figure.savefig(output_dir / "configuration_fit_report.png", dpi=180)
        except Exception:
            pass


    def draw_selected_experiment_plot_map(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            if self.experiment_store is None:
                self.load_experiment_samples()
            if self.experiment_cache is None:
                self.load_experiment_cache()
            if self.experiment_store is None or self.experiment_cache is None:
                raise ValueError("Build or load experiment results first.")
            if not self.plots_map_combo.cget("values"):
                self.refresh_plots_map_choices(redraw_only=True)
            kind, materials, substrate_name, substrate_label, surface_filter = self._parse_plots_choice()
            include_after = self.plots_fit_state_var.get().startswith("After")
            fit_mode = self._plots_fit_mode_filter()
            source_points = self._cached_experiment_plot_points(
                include_after=include_after,
                fit_mode=fit_mode,
            )
            points = [
                point
                for point in source_points
                if tuple(point["materials"]) == materials
                and str(point["substrate_key"]) == substrate_name
                and (surface_filter is None or str(point["surface"]) == surface_filter)
            ]
            if not points:
                fit_text = "cached optimized" if include_after else "estimated"
                location_text = f"{surface_filter} {substrate_label}" if surface_filter else substrate_label
                raise ValueError(
                    f"No {fit_text} points were found for {' / '.join(materials)} on {location_text}."
                )

            model = self._model_from_controls()
            angle_deg = float(self.angle_var.get())
            quality = self.sweep_quality_var.get()
            if kind == "1D":
                material_name = materials[0]
                thickness_min, thickness_max = self._map_bounds(
                    [float(point["thicknesses"][0]) for point in points]
                )
                stack = self._plots_stack_for_materials(
                    (material_name,),
                    (max(thickness_min, 1.0),),
                    substrate_name,
                )
                num_points = int(self.sweep_points_1d_var.get())

                def task(_progress):
                    return run_thickness_sweep_1d(
                        stack=stack,
                        model=model,
                        layer=material_name,
                        thickness_min_nm=thickness_min,
                        thickness_max_nm=thickness_max,
                        angle_deg=angle_deg,
                        num_points=num_points,
                        quality=quality,
                    )

                def on_success(result) -> str:
                    self._draw_plots_1d_map(
                        result,
                        points,
                        material_name,
                        substrate_name,
                        substrate_label,
                        include_after,
                        self._plots_fit_state_label(),
                    )
                    return f"Plotted {len(points)} {material_name} measurements."

            elif kind == "2D":
                material_1, material_2 = materials
                x_min, x_max = self._map_bounds([float(point["thicknesses"][0]) for point in points])
                y_min, y_max = self._map_bounds([float(point["thicknesses"][1]) for point in points])
                stack = self._plots_stack_for_materials(
                    materials,
                    (max(x_min, 1.0), max(y_min, 1.0)),
                    substrate_name,
                )
                num_points = int(self.sweep_points_2d_var.get())
                occurrence_2 = 1 if material_1 == material_2 else 0

                def task(_progress):
                    return run_thickness_sweep_2d(
                        stack=stack,
                        model=model,
                        layer_1=material_1,
                        layer_2=material_2,
                        thickness_1_min_nm=x_min,
                        thickness_1_max_nm=x_max,
                        thickness_2_min_nm=y_min,
                        thickness_2_max_nm=y_max,
                        angle_deg=angle_deg,
                        num_points_1=num_points,
                        num_points_2=num_points,
                        quality=quality,
                        layer_2_occurrence=occurrence_2,
                    )

                def on_success(result) -> str:
                    self._draw_plots_2d_map(
                        result,
                        points,
                        materials,
                        substrate_name,
                        substrate_label,
                        include_after,
                        self._plots_fit_state_label(),
                        map_kind="2D",
                    )
                    return f"Plotted {len(points)} {' / '.join(materials)} measurements."

            else:
                material_1, material_2, opaque_material = materials
                x_min, x_max = self._map_bounds([float(point["thicknesses"][0]) for point in points])
                y_min, y_max = self._map_bounds([float(point["thicknesses"][1]) for point in points])
                stack = self._plots_stack_for_materials(
                    materials,
                    (max(x_min, 1.0), max(y_min, 1.0), 1000.0),
                    substrate_name,
                )
                num_points = int(self.sweep_points_2d_var.get())

                def task(_progress):
                    return run_thickness_sweep_2d(
                        stack=stack,
                        model=model,
                        layer_1=material_1,
                        layer_2=material_2,
                        thickness_1_min_nm=x_min,
                        thickness_1_max_nm=x_max,
                        thickness_2_min_nm=y_min,
                        thickness_2_max_nm=y_max,
                        angle_deg=angle_deg,
                        num_points_1=num_points,
                        num_points_2=num_points,
                        quality=quality,
                    )

                def on_success(result) -> str:
                    self._draw_plots_2d_map(
                        result,
                        points,
                        materials,
                        substrate_name,
                        substrate_label,
                        include_after,
                        self._plots_fit_state_label(),
                        map_kind="3L",
                    )
                    return (
                        f"Plotted {len(points)} {material_1} / {material_2} measurements "
                        f"with {opaque_material} fixed as opaque."
                    )

            self._run_background(
                task,
                on_success,
                title="Experiment plots",
                busy_message="calculating experiment sweep plot",
            )
        except Exception as exc:
            messagebox.showerror("Plots", str(exc))

    def _cached_experiment_plot_points(
        self,
        include_after: bool,
        force_refresh: bool = False,
        fit_mode: str | None = None,
    ) -> list[dict[str, object]]:
        if include_after:
            if fit_mode is not None:
                return self._experiment_plot_points(include_after=True, fit_mode=fit_mode)
            if force_refresh or self.plots_after_points_cache is None:
                self.plots_after_points_cache = self._experiment_plot_points(include_after=True)
            return list(self.plots_after_points_cache)
        if force_refresh or self.plots_before_points_cache is None:
            self.plots_before_points_cache = self._experiment_plot_points(include_after=False)
        return list(self.plots_before_points_cache)

    def _experiment_plot_points(
        self,
        include_after: bool,
        fit_mode: str | None = None,
    ) -> list[dict[str, object]]:
        if self.experiment_store is None or self.experiment_cache is None:
            return []
        points: list[dict[str, object]] = []
        sample_cache: dict[str, object] = {}
        for index in range(self.experiment_cache.count):
            sample_name = str(self.experiment_cache.sample_names[index])
            measurement = str(self.experiment_cache.measurement_descriptions[index])
            try:
                sample = sample_cache.get(sample_name)
                if sample is None:
                    sample = self.experiment_store.load_sample(sample_name)
                    sample_cache[sample_name] = sample
            except Exception:
                continue
            if len(sample.layer_estimates) not in (1, 2, 3):
                continue
            materials = tuple(layer.material_name for layer in sample.layer_estimates)
            if any(material not in self.materials for material in materials):
                continue
            base_thicknesses = tuple(float(layer.thickness_nm) for layer in sample.layer_estimates)
            thicknesses = list(base_thicknesses)
            if len(materials) == 3 and not self._is_opaque_ag_triple(materials, thicknesses):
                continue
            if len(materials) not in (1, 2, 3):
                continue
            delta_e = float(self.experiment_cache.delta_e[index])
            fit_path = None
            fit_label = ""
            fit_stage_label = ""
            fit_profile_name = ""
            fit_model_label = ""
            fixed_opaque_metal = False
            if include_after:
                cached = self._best_cached_thickness_fit_for_measurement(
                    sample_name,
                    measurement,
                    optimization_mode=fit_mode,
                )
                if cached is None:
                    continue
                try:
                    fit_result = self._load_cached_thickness_fit_result(Path(cached["path"]))
                except Exception:
                    continue
                if len(fit_result.layer_results) != len(sample.layer_estimates):
                    continue
                thicknesses = [
                    float(layer.optimized_thickness_nm)
                    for layer in fit_result.layer_results
                ]
                fixed_opaque_metal = any(
                    layer.material_name in {"Ag", "Au"}
                    and float(layer.base_thickness_nm) >= 50.0
                    and abs(
                        float(layer.optimized_thickness_nm)
                        - float(layer.base_thickness_nm)
                    )
                    <= 1e-9
                    for layer in fit_result.layer_results
                )
                delta_e = float(fit_result.optimized_delta_e)
                fit_path = Path(cached["path"])
                fit_label = str(cached.get("stage_label", ""))
                fit_stage_label = str(cached.get("stage_label", ""))
                fit_profile_name = str(cached.get("profile_name", ""))
                fit_model_label = str(cached.get("model_label", ""))
            substrate_label = str(self.experiment_cache.substrate_classes[index])
            thickness_tuple = tuple(float(value) for value in thicknesses)
            thickness_moved = any(
                abs(float(after) - float(before)) > 1e-6
                for after, before in zip(thickness_tuple, base_thicknesses)
            )
            points.append(
                {
                    "index": index,
                    "sample_name": sample_name,
                    "measurement": measurement,
                    "materials": materials,
                    "thicknesses": thickness_tuple,
                    "base_thicknesses": base_thicknesses,
                    "uses_optimized_thicknesses": include_after,
                    "thickness_moved": thickness_moved,
                    "fixed_opaque_metal": fixed_opaque_metal,
                    "rgb": np.clip(self.experiment_cache.measured_rgb[index], 0.0, 1.0),
                    "hex": self._rgb_tuple_to_hex(self.experiment_cache.measured_rgb[index]),
                    "delta_e": delta_e,
                    "series": str(self.experiment_cache.sample_series[index]),
                    "substrate": substrate_label,
                    "substrate_key": self._plots_substrate_key(substrate_label),
                    "surface": str(self.experiment_cache.surface_classes[index]),
                    "kind": str(self.experiment_cache.measurement_kinds[index]),
                    "fit_path": fit_path,
                    "fit_label": fit_label,
                    "fit_stage_label": fit_stage_label,
                    "fit_profile_name": fit_profile_name,
                    "fit_model_label": fit_model_label,
                }
            )
        return points

    def _plots_substrate_key(self, substrate_label: object) -> str:
        text = str(substrate_label or "").strip()
        substrate_name = normalize_substrate_name(text)
        lowered = text.lower()
        if substrate_name is None:
            if lowered == "si" or lowered.startswith("si ") or "silicon" in lowered:
                substrate_name = "Si"
            elif lowered == "ti" or lowered.startswith("ti ") or "titanium" in lowered:
                substrate_name = "Ti"
            elif "substrate" in lowered or "glass" in lowered:
                substrate_name = "substrate"
        if substrate_name in self.materials:
            return substrate_name
        fallback = self.substrate_var.get()
        if fallback in self.materials:
            return fallback
        substrate_names = self._substrate_names()
        return substrate_names[0] if substrate_names else "Si"

    def _plots_stack_for_materials(
        self,
        materials: tuple[str, ...],
        thicknesses_nm: tuple[float, ...],
        substrate_name: str,
    ):
        substrate = self.materials[substrate_name]
        native_oxide = self._native_oxide_from_controls(substrate_name)
        deposited_layers = [
            Layer(self.materials[material], float(thickness))
            for material, thickness in zip(materials, thicknesses_nm)
        ]
        if self._use_effective_interfaces():
            return make_stack_with_interfaces(
                incident_medium=self.materials["air"],
                deposited_layers=deposited_layers,
                substrate=substrate,
                native_oxide=native_oxide,
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
                name="experiment plot sweep",
            )
        layers = list(deposited_layers)
        if native_oxide is not None:
            layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
        return make_stack(
            incident_medium=self.materials["air"],
            substrate=substrate,
            layers=layers,
            name="experiment plot sweep",
            display_layers=deposited_layers,
        )

    def _plots_configuration_text(
        self,
        materials: tuple[str, ...],
        substrate_name: str,
        substrate_label: str,
        stack_label: str,
    ) -> str:
        layer_parts = []
        for index, material in enumerate(materials):
            if index == 0:
                layer_parts.append(f"{material} (top, air-facing, x-axis)")
            elif index == 1 and len(materials) == 2:
                layer_parts.append(f"{material} (underlayer, y-axis)")
            elif index == 1:
                layer_parts.append(f"{material} (middle layer, y-axis)")
            elif index == 2 and material == "Ag":
                layer_parts.append(f"{material} (opaque fixed layer, simulated as 1000 nm)")
            else:
                layer_parts.append(material)
        stack_text = "air -> " + " -> ".join(layer_parts) + f" -> {substrate_name}"
        if len(materials) == 3:
            order_text = (
                f"{materials[0]} is top, {materials[1]} is middle, "
                f"{materials[2]} is fixed opaque because measured Ag >= 50 nm."
            )
        elif len(materials) == 2:
            order_text = f"{materials[0]} is the top layer; {materials[1]} is below it."
        else:
            order_text = f"{materials[0]} is the top layer."

        native_text = "Native oxide: off"
        if bool(self.native_oxide_enabled_var.get()):
            native = native_oxide_for_substrate(self.materials, substrate_name)
            if native is not None:
                native_text = (
                    f"Native oxide: {float(self.native_oxide_thickness_var.get()):.2g} nm "
                    f"{native.material.name} above {substrate_name}"
                )

        interface_lines: list[str] = []
        if self._use_effective_interfaces():
            interface_lines.append(
                "Interface mix: "
                f"{float(self.roughness_thickness_var.get()):.2g} nm, "
                f"fraction {float(self.roughness_fraction_var.get()):.2g}"
            )
        if self.model_mode_var.get() in {
            "Roughness corrected TMM",
            "Effective interface + diffuse redistribution",
        }:
            interface_lines.append(
                "Rough/diffuse: "
                f"RMS {float(self.rms_roughness_var.get()):.2g} nm, "
                f"scale {float(self.scatter_scale_var.get()):.2g}, "
                f"max {float(self.scatter_max_var.get()):.2g}"
            )

        lines = [
            "Background stack:",
            stack_text,
            order_text,
            f"Substrate group: {substrate_label}",
            f"Sweep stack label: {stack_label}",
            f"Constants: {self.material_profile_var.get()}",
            f"Optical model: {self.model_mode_var.get()}",
            f"Angle: {float(self.angle_var.get()):.2f} deg",
            native_text,
            *interface_lines,
        ]
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=42) or [""])
        return "\n".join(wrapped)

    @staticmethod
    def _plots_point_thickness_text(
        materials: tuple[str, ...],
        thicknesses_nm: tuple[float, ...],
    ) -> str:
        if len(materials) == 1 and thicknesses_nm:
            return f"top {materials[0]} {thicknesses_nm[0]:.1f} nm"
        if len(materials) >= 3 and len(thicknesses_nm) >= 3:
            return (
                f"top {materials[0]} {thicknesses_nm[0]:.1f} nm / "
                f"middle {materials[1]} {thicknesses_nm[1]:.1f} nm / "
                f"{materials[2]} {thicknesses_nm[2]:.1f} nm"
            )
        if len(materials) >= 2 and len(thicknesses_nm) >= 2:
            return (
                f"top {materials[0]} {thicknesses_nm[0]:.1f} nm / "
                f"under {materials[1]} {thicknesses_nm[1]:.1f} nm"
            )
        return " / ".join(f"{value:.1f} nm" for value in thicknesses_nm)

    def _plots_fit_mode_filter(self) -> str | None:
        state = self.plots_fit_state_var.get().lower()
        if "individual" in state:
            return "layer"
        if "same-material" in state:
            return "material_rate"
        return None

    def _plots_fit_state_label(self) -> str:
        state = self.plots_fit_state_var.get().strip()
        if state:
            return state
        return "After best cached thickness fit"

    def _draw_plots_1d_map(
        self,
        result,
        points: list[dict[str, object]],
        material_name: str,
        substrate_name: str,
        substrate_label: str,
        include_after: bool,
        mode_label: str,
    ) -> None:
        self.plots_figure.clear()
        grid = self.plots_figure.add_gridspec(
            3,
            2,
            height_ratios=[0.32, 1.0, 0.46],
            width_ratios=[2.9, 1.0],
            hspace=0.02,
            wspace=0.28,
        )
        ax = self.plots_figure.add_subplot(grid[1, 0])
        info_ax = self.plots_figure.add_subplot(grid[:, 1])
        strip = np.repeat(result.rgb_values[np.newaxis, :, :], 24, axis=0)
        ax.imshow(
            strip,
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_nm[0]),
                float(result.thickness_values_nm[-1]),
                0.0,
                1.0,
            ],
        )
        marker_by_surface = {"smooth": "o", "rough": "^"}
        for point_index, point in enumerate(points):
            y_value = 0.35 + 0.3 * ((point_index % 7) / 6.0)
            x_value = float(point["thicknesses"][0])
            base_thicknesses = point.get("base_thicknesses")
            if include_after and base_thicknesses is not None:
                try:
                    base_x = float(tuple(base_thicknesses)[0])
                except (TypeError, ValueError, IndexError):
                    base_x = x_value
                if abs(base_x - x_value) > 1e-6:
                    marker = marker_by_surface.get(str(point["surface"]), "s")
                    ax.plot(
                        [base_x, x_value],
                        [y_value, y_value],
                        color="#64748b",
                        alpha=0.55,
                        linewidth=0.8,
                        zorder=3,
                    )
                    ax.scatter(
                        [base_x],
                        [y_value],
                        s=26,
                        marker=marker,
                        facecolors="none",
                        edgecolors="#64748b",
                        linewidths=0.7,
                        alpha=0.75,
                        zorder=3,
                    )
            ax.scatter(
                [x_value],
                [y_value],
                s=72,
                marker=marker_by_surface.get(str(point["surface"]), "s"),
                facecolors=[point["rgb"]],
                edgecolors="#111827",
                linewidths=0.9,
                zorder=4,
            )
        ax.set_yticks([])
        ax.set_xlabel(f"{material_name} top-layer thickness (nm)")
        ax.grid(True, axis="x", color="white", alpha=0.35, linewidth=0.7)
        self._finish_plots_map_figure(
            ax,
            info_ax,
            points,
            title=f"{material_name} top-layer one-layer measured colours on simulated thickness sweep",
            mode_label=mode_label,
            configuration_text=self._plots_configuration_text(
                (material_name,),
                substrate_name,
                substrate_label,
                result.stack_label,
            ),
            extra_lines=[
                *self._plots_after_position_lines(points, include_after),
                *self._plots_sample_coverage_lines((material_name,), points, include_after),
            ],
        )

    def _draw_plots_2d_map(
        self,
        result,
        points: list[dict[str, object]],
        materials: tuple[str, ...],
        substrate_name: str,
        substrate_label: str,
        include_after: bool,
        mode_label: str,
        map_kind: str = "2D",
    ) -> None:
        self.plots_figure.clear()
        grid = self.plots_figure.add_gridspec(1, 2, width_ratios=[2.9, 1.0], wspace=0.28)
        ax = self.plots_figure.add_subplot(grid[0, 0])
        info_ax = self.plots_figure.add_subplot(grid[0, 1])
        ax.imshow(
            result.rgb_grid,
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_1_nm[0]),
                float(result.thickness_values_1_nm[-1]),
                float(result.thickness_values_2_nm[0]),
                float(result.thickness_values_2_nm[-1]),
            ],
        )
        marker_by_surface = {"smooth": "o", "rough": "^"}
        plotted_labels: set[str] = set()
        for point in points:
            surface = str(point["surface"])
            label = f"{surface}, {point['substrate']}"
            x_value = float(point["thicknesses"][0])
            y_value = float(point["thicknesses"][1])
            base_thicknesses = point.get("base_thicknesses")
            if include_after and base_thicknesses is not None:
                try:
                    base_x = float(tuple(base_thicknesses)[0])
                    base_y = float(tuple(base_thicknesses)[1])
                except (TypeError, ValueError, IndexError):
                    base_x = x_value
                    base_y = y_value
                if abs(base_x - x_value) > 1e-6 or abs(base_y - y_value) > 1e-6:
                    marker = marker_by_surface.get(surface, "s")
                    ax.plot(
                        [base_x, x_value],
                        [base_y, y_value],
                        color="#64748b",
                        alpha=0.55,
                        linewidth=0.8,
                        zorder=3,
                    )
                    ax.scatter(
                        [base_x],
                        [base_y],
                        s=26,
                        marker=marker,
                        facecolors="none",
                        edgecolors="#64748b",
                        linewidths=0.7,
                        alpha=0.75,
                        zorder=3,
                    )
            ax.scatter(
                [x_value],
                [y_value],
                s=72,
                marker=marker_by_surface.get(surface, "s"),
                facecolors=[point["rgb"]],
                edgecolors="#111827",
                linewidths=0.9,
                label=None if label in plotted_labels else label,
                zorder=4,
            )
            plotted_labels.add(label)
        if map_kind == "3L":
            y_role = "middle-layer"
            title = (
                f"{materials[0]} top / {materials[1]} middle with opaque {materials[2]} "
                "measured colours on simulated sweep"
            )
        else:
            y_role = "underlayer"
            title = f"{materials[0]} top / {materials[1]} underlayer measured colours on simulated thickness sweep"
        extra_lines = [
            *self._plots_after_position_lines(points, include_after),
            *self._plots_sample_coverage_lines(materials, points, include_after),
        ]
        ax.set_xlabel(f"{materials[0]} top-layer thickness (nm)")
        ax.set_ylabel(f"{materials[1]} {y_role} thickness (nm)")
        ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
        if plotted_labels:
            ax.legend(loc="best", fontsize=8)
        self._finish_plots_map_figure(
            ax,
            info_ax,
            points,
            title=title,
            mode_label=mode_label,
            configuration_text=self._plots_configuration_text(
                materials,
                substrate_name,
                substrate_label,
                result.stack_label,
            ),
            extra_lines=extra_lines,
        )

    def _finish_plots_map_figure(
        self,
        ax,
        info_ax,
        points: list[dict[str, object]],
        title: str,
        mode_label: str,
        configuration_text: str,
        extra_lines: list[str] | None = None,
    ) -> None:
        delta_values = np.asarray([float(point["delta_e"]) for point in points], dtype=float)
        average = float(np.nanmean(delta_values)) if delta_values.size else float("nan")
        delta_label = self._delta_e_label()
        ax.set_title(
            f"{title}\n{mode_label}; mean {delta_label} {average:.2f} over {len(points)} measurements",
            fontsize=11,
            fontweight="semibold",
        )
        info_ax.axis("off")
        info_ax.set_title("Configuration and samples", fontweight="semibold")
        ordered = sorted(points, key=lambda point: float(point["delta_e"]))
        sample_lines = []
        for point in ordered[:24]:
            thickness_text = self._plots_point_thickness_text(
                tuple(point["materials"]),
                tuple(float(value) for value in point["thicknesses"]),
            )
            sample_lines.append(
                f"{point['sample_name']}: {thickness_text}, "
                f"{delta_label} {float(point['delta_e']):.1f}, {point['hex']}"
            )
        if len(ordered) > 24:
            sample_lines.append(f"... and {len(ordered) - 24} more")
        lines = [configuration_text]
        if extra_lines:
            lines.extend(["", *extra_lines])
        lines.extend(["", f"Samples sorted by {delta_label}:", *sample_lines])
        info_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=7.5,
            family="Consolas",
        )
        self.plots_info_var.set(
            f"{mode_label}: plotted {len(points)} measurements; mean {delta_label} {average:.2f}."
        )
        self.plots_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88)
        self.plots_canvas.draw_idle()

    def draw_experiment_plot_colour_distances(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            if self.experiment_store is None:
                self.load_experiment_samples()
            if self.experiment_cache is None:
                self.load_experiment_cache()
            if self.experiment_store is None or self.experiment_cache is None:
                raise ValueError("Build or load experiment results first.")
            before_points = self._cached_experiment_plot_points(include_after=False)
            after_points = self._cached_experiment_plot_points(include_after=True)
            if not before_points:
                raise ValueError("No one- or two-layer measured samples were found.")
            self._draw_plots_colour_distances(before_points, after_points)
        except Exception as exc:
            messagebox.showerror("Plots", str(exc))

    def draw_1d_thickness_colour_difference(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            model = self._model_from_controls()
            layer, occurrence = self._selected_sweep_layer(self.sweep_layer_1_var)
            current_thickness = self._current_layer_thickness_for_sweep_label(self.sweep_layer_1_var)
            stack = self._sensitivity_stack_for_selected_layers(
                ((layer, occurrence),),
                (current_thickness,),
            )
            thickness_min_nm = float(self.sweep_min_var.get())
            thickness_max_nm = float(self.sweep_max_var.get())
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_1d_var.get())
            quality = self.sweep_quality_var.get()
            metric = self._current_colour_metric()

            def task(_progress):
                result = run_thickness_sweep_1d(
                    stack=stack,
                    model=model,
                    layer=layer,
                    layer_occurrence=occurrence,
                    thickness_min_nm=thickness_min_nm,
                    thickness_max_nm=thickness_max_nm,
                    angle_deg=angle_deg,
                    num_points=num_points,
                    quality=quality,
                    save_reflectance=True,
                )
                reference_xyz = self._reference_stack_xyz(stack, model, result.wavelengths_nm, angle_deg)
                delta_values = self._delta_e_series_from_reflectance(
                    result.reflectance_spectra,
                    reference_xyz,
                    result.wavelengths_nm,
                    metric,
                )
                return result, delta_values

            def on_success(payload) -> str:
                result, delta_values = payload
                self._draw_1d_thickness_colour_difference(result, delta_values)
                return f"Plotted 1-layer colour change for {result.layer_name}."

            self._run_background(
                task,
                on_success,
                title="Colour change",
                busy_message="calculating 1-layer colour change",
            )
        except Exception as exc:
            messagebox.showerror("Colour change plot", str(exc))

    def draw_2d_thickness_colour_difference(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            model = self._model_from_controls()
            layer_1, occurrence_1 = self._selected_sweep_layer(self.sweep_layer_1_var)
            layer_2, occurrence_2 = self._selected_sweep_layer(self.sweep_layer_2_var)
            current_1 = self._current_layer_thickness_for_sweep_label(self.sweep_layer_1_var)
            current_2 = self._current_layer_thickness_for_sweep_label(self.sweep_layer_2_var)
            stack = self._sensitivity_stack_for_selected_layers(
                ((layer_1, occurrence_1), (layer_2, occurrence_2)),
                (current_1, current_2),
            )
            thickness_min_nm = float(self.sweep_min_var.get())
            thickness_max_nm = float(self.sweep_max_var.get())
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_2d_var.get())
            quality = self.sweep_quality_var.get()
            metric = self._current_colour_metric()

            def task(_progress):
                result = run_thickness_sweep_2d(
                    stack=stack,
                    model=model,
                    layer_1=layer_1,
                    layer_1_occurrence=occurrence_1,
                    layer_2=layer_2,
                    layer_2_occurrence=occurrence_2,
                    thickness_1_min_nm=thickness_min_nm,
                    thickness_1_max_nm=thickness_max_nm,
                    thickness_2_min_nm=thickness_min_nm,
                    thickness_2_max_nm=thickness_max_nm,
                    angle_deg=angle_deg,
                    num_points_1=num_points,
                    num_points_2=num_points,
                    quality=quality,
                    save_reflectance=True,
                )
                reference_xyz = self._reference_stack_xyz(stack, model, result.wavelengths_nm, angle_deg)
                delta_grid = self._delta_e_grid_from_reflectance(
                    result.reflectance_data,
                    reference_xyz,
                    result.wavelengths_nm,
                    metric,
                )
                return result, delta_grid

            def on_success(payload) -> str:
                result, delta_grid = payload
                self._draw_2d_thickness_colour_difference(result, delta_grid)
                return f"Plotted 2-layer colour change for {result.layer_name_1} / {result.layer_name_2}."

            self._run_background(
                task,
                on_success,
                title="Colour change",
                busy_message="calculating 2-layer colour change map",
            )
        except Exception as exc:
            messagebox.showerror("Colour change map", str(exc))

    def _sensitivity_stack_for_selected_layers(
        self,
        selected_layers: tuple[tuple[str, int], ...],
        current_thicknesses_nm: tuple[float, ...],
    ):
        materials = tuple(material for material, _occurrence in selected_layers)
        substrate_name = self.substrate_var.get()
        return self._plots_stack_for_materials(materials, current_thicknesses_nm, substrate_name)

    def _reference_stack_xyz(self, stack, model, wavelengths_nm, angle_deg: float):
        reference = model.simulate(stack, wavelengths_nm, angle_deg)
        color_cache = prepare_color_conversion(wavelengths_nm)
        return reflectance_to_xyz(reference.reflectance, cache=color_cache)

    def _delta_e_series_from_reflectance(
        self,
        reflectance_spectra,
        reference_xyz,
        wavelengths_nm,
        metric: str,
    ) -> np.ndarray:
        if reflectance_spectra is None:
            raise ValueError("No reflectance spectra were saved for the colour-change plot.")
        color_cache = prepare_color_conversion(wavelengths_nm)
        return np.asarray(
            [
                delta_e_colour(reference_xyz, reflectance_to_xyz(spectrum, cache=color_cache), metric=metric)
                for spectrum in reflectance_spectra
            ],
            dtype=float,
        )

    def _delta_e_grid_from_reflectance(
        self,
        reflectance_data,
        reference_xyz,
        wavelengths_nm,
        metric: str,
    ) -> np.ndarray:
        if reflectance_data is None:
            raise ValueError("No reflectance spectra were saved for the colour-change map.")
        color_cache = prepare_color_conversion(wavelengths_nm)
        delta_grid = np.empty(reflectance_data.shape[:2], dtype=float)
        for y_index in range(reflectance_data.shape[0]):
            for x_index in range(reflectance_data.shape[1]):
                xyz = reflectance_to_xyz(reflectance_data[y_index, x_index], cache=color_cache)
                delta_grid[y_index, x_index] = delta_e_colour(reference_xyz, xyz, metric=metric)
        return delta_grid

    def _thickness_sample_points_for_sensitivity(
        self,
        selected_layers: tuple[tuple[str, int], ...],
    ) -> list[dict[str, object]]:
        if self.experiment_store is None:
            try:
                self.load_experiment_samples(show_errors=False)
            except Exception:
                pass
        if self.experiment_store is None:
            return []

        points: list[dict[str, object]] = []
        for sample_name in self.experiment_store.sample_names(require_spectra=False):
            try:
                sample = self.experiment_store.load_sample(sample_name)
            except Exception:
                continue
            sample_materials = tuple(layer.material_name for layer in sample.layer_estimates)
            requested_materials = tuple(material for material, _occurrence in selected_layers)
            if sample_materials != requested_materials:
                continue
            thicknesses: list[float] = []
            for material_name, occurrence in selected_layers:
                value = self._sample_layer_occurrence_thickness(sample.layer_estimates, material_name, occurrence)
                if value is None:
                    break
                thicknesses.append(value)
            if len(thicknesses) != len(selected_layers):
                continue
            points.append(
                {
                    "sample_name": str(sample.sample_name),
                    "thicknesses": tuple(thicknesses),
                    "measurement_count": len(sample.measurements),
                }
            )
        return points

    @staticmethod
    def _sample_layer_occurrence_thickness(layers, material_name: str, occurrence: int) -> float | None:
        seen = 0
        for layer in layers:
            if layer.material_name != material_name:
                continue
            if seen == occurrence:
                return float(layer.thickness_nm)
            seen += 1
        return None

    @staticmethod
    def _sample_count_summary(points: list[dict[str, object]]) -> tuple[int, int]:
        sample_count = len({str(point["sample_name"]) for point in points})
        measurement_count = int(sum(int(point["measurement_count"]) for point in points))
        return sample_count, measurement_count

    @staticmethod
    def _sample_thickness_bounds(
        points: list[dict[str, object]],
        axis_index: int,
    ) -> tuple[float, float] | None:
        values = [
            float(point["thicknesses"][axis_index])
            for point in points
            if len(tuple(point["thicknesses"])) > axis_index
        ]
        if not values:
            return None
        return float(min(values)), float(max(values))

    def _draw_1d_thickness_colour_difference(self, result, delta_values: np.ndarray) -> None:
        self.plots_figure.clear()
        sample_points = self._thickness_sample_points_for_sensitivity(
            (self._selected_sweep_layer(self.sweep_layer_1_var),)
        )
        grid = self.plots_figure.add_gridspec(3, 1, height_ratios=[3.0, 0.55, 0.45], hspace=0.18)
        ax = self.plots_figure.add_subplot(grid[0, 0])
        strip_ax = self.plots_figure.add_subplot(grid[1, 0], sharex=ax)
        count_ax = self.plots_figure.add_subplot(grid[2, 0], sharex=ax)
        x_values = np.asarray(result.thickness_values_nm, dtype=float)
        sample_bounds = self._sample_thickness_bounds(sample_points, 0)
        if sample_bounds is not None:
            sample_min, sample_max = sample_bounds
            in_sample_range = (x_values >= sample_min) & (x_values <= sample_max)
            if np.any(x_values < sample_min):
                ax.axvspan(float(x_values[0]), sample_min, color="#e5e7eb", alpha=0.55, zorder=0)
            if np.any(x_values > sample_max):
                ax.axvspan(sample_max, float(x_values[-1]), color="#e5e7eb", alpha=0.55, zorder=0)
            plotted_delta = np.where(in_sample_range, delta_values, np.nan)
            ax.axvline(sample_min, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.85)
            ax.axvline(sample_max, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.85)
        else:
            plotted_delta = np.full_like(delta_values, np.nan, dtype=float)
            ax.text(
                0.5,
                0.5,
                "No exact deposited samples for this stack",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#52606d",
            )
        ax.plot(x_values, plotted_delta, color="#0f766e", linewidth=2.0)
        ax.fill_between(x_values, plotted_delta, color="#0f766e", alpha=0.16)
        ax.axvline(
            self._current_layer_thickness_for_sweep_label(self.sweep_layer_1_var),
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="current thickness",
        )
        ax.set_ylabel(self._delta_e_label())
        ax.set_title(
            f"Colour change vs {result.layer_name} thickness\n"
            f"Only shown inside deposited sample range; {result.stack_label}",
            fontsize=10.5,
            fontweight="semibold",
        )
        ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
        ax.legend(loc="best", fontsize=7.5)
        strip_ax.imshow(
            result.rgb_values[np.newaxis, :, :],
            aspect="auto",
            origin="lower",
            extent=[float(x_values[0]), float(x_values[-1]), 0.0, 1.0],
        )
        strip_ax.set_yticks([])
        strip_ax.set_ylabel("colour", fontsize=8)
        sample_count, measurement_count = self._sample_count_summary(sample_points)
        if sample_points:
            thicknesses = np.asarray([float(point["thicknesses"][0]) for point in sample_points], dtype=float)
            bins = min(max(int(np.sqrt(thicknesses.size)) + 2, 6), 24)
            count_ax.hist(
                thicknesses,
                bins=bins,
                range=(float(x_values[0]), float(x_values[-1])),
                color="#5b8db8",
                alpha=0.72,
                edgecolor="#243b53",
                linewidth=0.4,
            )
            count_ax.scatter(
                thicknesses,
                np.full_like(thicknesses, 0.12),
                s=[24 + 6 * int(point["measurement_count"]) for point in sample_points],
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.7,
                alpha=0.75,
            )
        count_ax.set_ylabel("samples", fontsize=8)
        count_ax.set_xlabel(f"{result.layer_name} thickness (nm)")
        count_ax.tick_params(axis="y", labelsize=7)
        count_ax.grid(True, axis="x", color="#bcccdc", alpha=0.25, linewidth=0.7)
        self.plots_figure.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.86)
        self.plots_canvas.draw_idle()
        max_delta = float(np.nanmax(plotted_delta)) if np.any(np.isfinite(plotted_delta)) else float("nan")
        self.plots_info_var.set(
            f"1-layer colour change inside sample range: max {self._delta_e_label()} {max_delta:.2f}; "
            f"{sample_count} deposited samples, {measurement_count} measurements for this layer."
        )

    def _draw_2d_thickness_colour_difference(self, result, delta_grid: np.ndarray) -> None:
        self.plots_figure.clear()
        sample_points = self._thickness_sample_points_for_sensitivity(
            (
                self._selected_sweep_layer(self.sweep_layer_1_var),
                self._selected_sweep_layer(self.sweep_layer_2_var),
            )
        )
        grid = self.plots_figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24)
        delta_ax = self.plots_figure.add_subplot(grid[0, 0])
        colour_ax = self.plots_figure.add_subplot(grid[0, 1])
        extent = [
            float(result.thickness_values_1_nm[0]),
            float(result.thickness_values_1_nm[-1]),
            float(result.thickness_values_2_nm[0]),
            float(result.thickness_values_2_nm[-1]),
        ]
        x_values = np.asarray(result.thickness_values_1_nm, dtype=float)
        y_values = np.asarray(result.thickness_values_2_nm, dtype=float)
        x_bounds = self._sample_thickness_bounds(sample_points, 0)
        y_bounds = self._sample_thickness_bounds(sample_points, 1)
        if x_bounds is not None and y_bounds is not None:
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds
            x_mask = (x_values >= x_min) & (x_values <= x_max)
            y_mask = (y_values >= y_min) & (y_values <= y_max)
            coverage_mask = np.outer(y_mask, x_mask)
            plotted_delta = np.where(coverage_mask, delta_grid, np.nan)
        else:
            plotted_delta = np.full_like(delta_grid, np.nan, dtype=float)
        image = delta_ax.imshow(plotted_delta, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        delta_ax.set_facecolor("#e5e7eb")
        cbar = self.plots_figure.colorbar(image, ax=delta_ax, fraction=0.046, pad=0.04)
        cbar.set_label(self._delta_e_label())
        delta_ax.scatter(
            [self._current_layer_thickness_for_sweep_label(self.sweep_layer_1_var)],
            [self._current_layer_thickness_for_sweep_label(self.sweep_layer_2_var)],
            s=54,
            facecolors="none",
            edgecolors="#f97316",
            linewidths=1.5,
            label="current stack",
        )
        if sample_points:
            xs = np.asarray([float(point["thicknesses"][0]) for point in sample_points], dtype=float)
            ys = np.asarray([float(point["thicknesses"][1]) for point in sample_points], dtype=float)
            sizes = [26 + 8 * int(point["measurement_count"]) for point in sample_points]
            delta_ax.scatter(
                xs,
                ys,
                s=sizes,
                facecolors="none",
                edgecolors="#ffffff",
                linewidths=1.5,
                alpha=0.95,
                label="sample count",
            )
            delta_ax.scatter(
                xs,
                ys,
                s=sizes,
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.7,
                alpha=0.9,
            )
        delta_ax.legend(loc="best", fontsize=7.5)
        colour_grid = np.asarray(result.rgb_grid, dtype=float).copy()
        if x_bounds is not None and y_bounds is not None:
            colour_grid[~coverage_mask] = np.asarray([0.88, 0.89, 0.91], dtype=float)
        colour_ax.imshow(colour_grid, aspect="auto", origin="lower", extent=extent)
        for ax in (delta_ax, colour_ax):
            ax.set_xlabel(f"{result.layer_name_1} thickness (nm)")
            ax.set_ylabel(f"{result.layer_name_2} thickness (nm)")
        max_delta = float(np.nanmax(plotted_delta)) if np.any(np.isfinite(plotted_delta)) else float("nan")
        delta_ax.set_title(f"Colour difference inside sample range\nmax {self._delta_e_label()} {max_delta:.2f}")
        colour_ax.set_title("Predicted colour inside sample range")
        self.plots_figure.suptitle(
            f"{result.layer_name_1} / {result.layer_name_2} thickness sensitivity; outside sample range is blank",
            fontsize=10.5,
            fontweight="semibold",
        )
        self.plots_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.83)
        self.plots_canvas.draw_idle()
        sample_count, measurement_count = self._sample_count_summary(sample_points)
        self.plots_info_var.set(
            f"2-layer colour change inside sample range: max {self._delta_e_label()} {max_delta:.2f}; "
            f"{sample_count} deposited samples, {measurement_count} measurements for these layers."
        )

    def _draw_plots_colour_distances(
        self,
        before_points: list[dict[str, object]],
        after_points: list[dict[str, object]],
    ) -> None:
        self.plots_figure.clear()
        ax = self.plots_figure.add_subplot(1, 1, 1)
        after_by_key = {
            (str(point["sample_name"]), str(point["measurement"])): point
            for point in after_points
        }
        ordered = sorted(
            before_points,
            key=lambda point: (
                self._sample_group_sort_key(point["sample_name"]),
                str(point["surface"]),
                str(point["substrate"]),
                str(point["measurement"]),
            ),
        )
        x = np.arange(len(ordered))
        before_values = np.asarray([float(point["delta_e"]) for point in ordered], dtype=float)
        after_values = np.asarray(
            [
                float(after_by_key[(str(point["sample_name"]), str(point["measurement"]))]["delta_e"])
                if (str(point["sample_name"]), str(point["measurement"])) in after_by_key
                else np.nan
                for point in ordered
            ],
            dtype=float,
        )
        ax.bar(x, before_values, color="#5b8db8", alpha=0.74, label="Before thickness fit")
        finite_after = np.isfinite(after_values)
        if np.any(finite_after):
            ax.scatter(
                x[finite_after],
                after_values[finite_after],
                s=34,
                c="#f97316",
                edgecolors="#111827",
                linewidths=0.4,
                label="After best cached thickness fit",
                zorder=4,
            )
        label_count = min(len(ordered), 80)
        tick_step = max(1, int(np.ceil(label_count / 32)))
        tick_positions = x[:label_count:tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(ordered[int(position)]["sample_name"]) for position in tick_positions],
            rotation=60,
            ha="right",
            fontsize=6.8,
        )
        if len(ordered) > label_count:
            ax.set_xlim(-0.75, label_count - 0.25)
        delta_label = self._delta_e_label()
        before_mean = float(np.nanmean(before_values))
        title = f"Colour distance by measurement: mean before {delta_label} {before_mean:.2f}"
        if np.any(finite_after):
            title += f", mean after {np.nanmean(after_values):.2f}"
            title += f"\n{self._plots_colour_distance_fit_basis_summary(list(after_by_key.values()))}"
        ax.set_xlabel("Sample designation, grouped by series")
        ax.set_ylabel(delta_label)
        ax.set_title(title, fontsize=9, fontweight="semibold", pad=6)
        ax.grid(True, axis="y", alpha=0.28)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(fontsize=7)
        self.plots_info_var.set(
            f"Colour-distance plot includes {len(ordered)} before-fit measurements and "
            f"{int(np.count_nonzero(finite_after))} cached after-fit measurements."
        )
        self.plots_figure.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.86)
        self.plots_canvas.draw_idle()

    def _plots_colour_distance_fit_basis_summary(
        self,
        after_points: list[dict[str, object]],
        width: int = 135,
    ) -> str:
        if not after_points:
            return "Fit shown: no cached after-fit points"
        stage = self._summarize_row_values(after_points, "fit_stage_label", "Thickness fit")
        profile = self._summarize_row_values(after_points, "fit_profile_name", "unknown constants")
        model = self._summarize_row_values(after_points, "fit_model_label", "unknown optical model")
        text = f"Fit shown: {stage}; constants: {profile}; optical model: {model}"
        return textwrap.shorten(text, width=width, placeholder="...")

    def draw_individual_thickness_adjustment_plot(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            if self.experiment_store is None:
                self.load_experiment_samples(show_errors=False)
            if self.experiment_cache is None:
                try:
                    self.load_experiment_cache()
                except Exception:
                    pass
            if self.experiment_store is None:
                raise ValueError("Load experiment data first.")

            materials_filter: tuple[str, ...] | None = None
            substrate_filter: str | None = None
            surface_filter: str | None = None
            filter_label = "all cached individual thickness fits"
            try:
                if hasattr(self, "plots_map_combo") and self.plots_map_combo.cget("values"):
                    _kind, materials_filter, substrate_filter, substrate_label, surface_filter = self._parse_plots_choice()
                    surface_text = f"{surface_filter} " if surface_filter else ""
                    filter_label = f"{' / '.join(materials_filter)} on {surface_text}{substrate_label}".strip()
            except Exception:
                materials_filter = None
                substrate_filter = None
                surface_filter = None

            cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)
            if not cache_dir.exists():
                raise FileNotFoundError(
                    f"No thickness-fit cache folder found yet.\nExpected: {cache_dir}"
                )
            paths = list(cache_dir.glob("*.json"))
            if not paths:
                raise ValueError("No cached thickness-fit JSON files were found yet.")

            store = self.experiment_store
            active_metric = self._current_colour_metric()
            context_lookup = self._plot_measurement_context_lookup()
            filters = {
                "materials": materials_filter,
                "substrate": substrate_filter,
                "surface": surface_filter,
            }

            def task(progress):
                return self._individual_thickness_adjustment_rows(
                    paths=paths,
                    store=store,
                    active_metric=active_metric,
                    context_lookup=context_lookup,
                    filters=filters,
                    progress=progress,
                )

            def on_success(rows: list[dict[str, object]]) -> str:
                if not rows:
                    raise ValueError(
                        "No individual thickness-fit adjustments matched the current map/filter. "
                        "Run individual thickness fits first, or choose a broader map."
                    )
                self._draw_individual_thickness_adjustment_rows(rows, filter_label)
                measurement_count = len({(row["sample_name"], row["measurement"]) for row in rows})
                return (
                    f"Plotted {len(rows)} individual layer adjustments from "
                    f"{measurement_count} measurements."
                )

            self._run_background(
                task,
                on_success,
                title="Thickness adjustments",
                busy_message="reading individual thickness-fit caches",
                progress_max=len(paths),
            )
        except Exception as exc:
            messagebox.showerror("Thickness adjustments", str(exc))

    def _plot_measurement_context_lookup(self) -> dict[tuple[str, str], dict[str, str]]:
        lookup: dict[tuple[str, str], dict[str, str]] = {}
        if self.experiment_cache is None:
            return lookup
        for index in range(self.experiment_cache.count):
            sample_name = str(self.experiment_cache.sample_names[index])
            measurement = str(self.experiment_cache.measurement_descriptions[index])
            substrate_label = str(self.experiment_cache.substrate_classes[index])
            lookup[(sample_name, measurement)] = {
                "substrate": substrate_label,
                "substrate_key": self._plots_substrate_key(substrate_label),
                "surface": str(self.experiment_cache.surface_classes[index]),
            }
        return lookup

    def _individual_thickness_adjustment_rows(
        self,
        *,
        paths: list[Path],
        store: ExperimentDataStore,
        active_metric: str,
        context_lookup: dict[tuple[str, str], dict[str, str]],
        filters: dict[str, object],
        progress,
    ) -> list[dict[str, object]]:
        best_by_measurement: dict[tuple[str, str], dict[str, object]] = {}
        materials_filter = filters.get("materials")
        substrate_filter = filters.get("substrate")
        surface_filter = filters.get("surface")

        for index, path in enumerate(paths, start=1):
            self._wait_if_paused(progress)
            if index == 1 or index % 25 == 0 or index == len(paths):
                progress(index, f"reading thickness-fit cache {index:,}/{len(paths):,}")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                if str(metadata.get("optimization_mode", "layer")) != "layer":
                    continue
                colour_metric = normalise_colour_metric(metadata.get("colour_metric", COLOUR_METRIC_CIE76))
                if colour_metric != active_metric:
                    continue
                best = data.get("evaluations", {}).get(data.get("best_key"), {})
                if not isinstance(best, dict):
                    continue

                sample_name = str(metadata.get("sample_name", ""))
                measurement = str(metadata.get("measurement_description", ""))
                if not sample_name or not measurement:
                    continue

                layer_names, base_thicknesses = self._cache_layer_names_and_thicknesses(
                    metadata,
                    store,
                    sample_name,
                )
                if not layer_names or len(layer_names) != len(base_thicknesses):
                    continue
                if materials_filter is not None and tuple(layer_names) != tuple(materials_filter):
                    continue

                key = (sample_name, measurement)
                if not self._cache_record_matches_plot_filter(
                    key=key,
                    metadata=metadata,
                    context_lookup=context_lookup,
                    substrate_filter=None if substrate_filter is None else str(substrate_filter),
                    surface_filter=None if surface_filter is None else str(surface_filter),
                ):
                    continue

                delta = float(best.get("delta_e", float("inf")))
                previous = best_by_measurement.get(key)
                if previous is not None:
                    previous_delta = float(previous["delta_e"])
                    previous_mtime = float(previous["mtime"])
                    if (delta, -path.stat().st_mtime) >= (previous_delta, -previous_mtime):
                        continue
                best_by_measurement[key] = {
                    "path": path,
                    "data": data,
                    "metadata": metadata,
                    "best": best,
                    "delta_e": delta,
                    "mtime": path.stat().st_mtime,
                    "layer_names": tuple(layer_names),
                    "base_thicknesses": tuple(base_thicknesses),
                }
            except Exception:
                continue

        rows: list[dict[str, object]] = []
        for (sample_name, measurement), record in best_by_measurement.items():
            layer_names = tuple(str(value) for value in record["layer_names"])
            base_thicknesses = tuple(float(value) for value in record["base_thicknesses"])
            best = record["best"]
            metadata = record["metadata"]
            offsets = [float(value) for value in best.get("offsets_percent", [])]
            variable_labels = best.get("variable_labels") or metadata.get("variable_labels") or []
            percents = self._layer_percents_from_cache_labels(
                layer_names,
                offsets,
                variable_labels,
            )
            for layer_index, (material, base_thickness, percent) in enumerate(
                zip(layer_names, base_thicknesses, percents),
                start=1,
            ):
                if not np.isfinite(base_thickness) or base_thickness <= 0:
                    continue
                adjustment_nm = base_thickness * percent / 100.0
                rows.append(
                    {
                        "sample_name": sample_name,
                        "measurement": measurement,
                        "material": material,
                        "layer_index": layer_index,
                        "base_thickness_nm": float(base_thickness),
                        "adjustment_nm": float(adjustment_nm),
                        "percent_change": float(percent),
                        "optimized_delta_e": float(record["delta_e"]),
                        "profile_name": str(metadata.get("profile_name", "")),
                        "model_label": str(metadata.get("model_label", "")),
                        "fit_reflectance_scale": bool(metadata.get("fit_reflectance_scale", False)),
                        "cache_path": str(record["path"]),
                    }
                )
        return rows

    @staticmethod
    def _cache_layer_names_and_thicknesses(
        metadata: dict[str, object],
        store: ExperimentDataStore,
        sample_name: str,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        layer_names_raw = metadata.get("layer_names", [])
        thicknesses_raw = metadata.get("base_thicknesses_nm", [])
        try:
            layer_names = tuple(str(value) for value in layer_names_raw)
            thicknesses = tuple(float(value) for value in thicknesses_raw)
        except Exception:
            layer_names = ()
            thicknesses = ()
        if layer_names and len(layer_names) == len(thicknesses):
            return layer_names, thicknesses
        sample = store.load_sample(sample_name)
        return (
            tuple(layer.material_name for layer in sample.layer_estimates),
            tuple(float(layer.thickness_nm) for layer in sample.layer_estimates),
        )

    @staticmethod
    def _layer_percents_from_cache_labels(
        layer_names: tuple[str, ...],
        offsets: list[float],
        variable_labels: object,
    ) -> list[float]:
        labels = [str(label) for label in variable_labels] if isinstance(variable_labels, list) else []
        if len(offsets) == len(layer_names):
            return [float(value) for value in offsets]
        percents: list[float] = []
        for layer_index, material in enumerate(layer_names, start=1):
            exact_label = f"{material} #{layer_index}"
            label_index = next(
                (
                    index
                    for index, label in enumerate(labels)
                    if label == exact_label
                ),
                None,
            )
            if label_index is None:
                label_index = next(
                    (
                        index
                        for index, label in enumerate(labels)
                        if label == material
                    ),
                    None,
                )
            percents.append(
                float(offsets[label_index])
                if label_index is not None and label_index < len(offsets)
                else 0.0
            )
        return percents

    @staticmethod
    def _cache_record_matches_plot_filter(
        *,
        key: tuple[str, str],
        metadata: dict[str, object],
        context_lookup: dict[tuple[str, str], dict[str, str]],
        substrate_filter: str | None,
        surface_filter: str | None,
    ) -> bool:
        if substrate_filter is None and surface_filter is None:
            return True
        context = context_lookup.get(key)
        if context is not None:
            if substrate_filter is not None and context.get("substrate_key") != substrate_filter:
                return False
            if surface_filter is not None and context.get("surface") != surface_filter:
                return False
            return True

        substrate_name = normalize_substrate_name(str(metadata.get("substrate_name", "")))
        if substrate_filter is not None and substrate_name != substrate_filter:
            return False
        if surface_filter is not None and surface_filter not in key[1].lower():
            return False
        return True

    @staticmethod
    def _binned_median_points(
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        x = np.asarray(x_values[finite], dtype=float)
        y = np.asarray(y_values[finite], dtype=float)
        if x.size < 6:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        bin_count = min(8, max(3, int(np.sqrt(x.size))))
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bin_count + 1)))
        if edges.size < 3:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        trend_x: list[float] = []
        trend_y: list[float] = []
        for edge_index in range(edges.size - 1):
            left = edges[edge_index]
            right = edges[edge_index + 1]
            if edge_index == edges.size - 2:
                mask = (x >= left) & (x <= right)
            else:
                mask = (x >= left) & (x < right)
            if np.count_nonzero(mask) < 2:
                continue
            trend_x.append(float(np.median(x[mask])))
            trend_y.append(float(np.median(y[mask])))
        return np.asarray(trend_x, dtype=float), np.asarray(trend_y, dtype=float)

    def _draw_individual_thickness_adjustment_rows(
        self,
        rows: list[dict[str, object]],
        filter_label: str,
    ) -> None:
        self.plots_figure.clear()
        grid = self.plots_figure.add_gridspec(
            2,
            2,
            width_ratios=[3.0, 1.05],
            height_ratios=[1.0, 0.78],
            hspace=0.32,
            wspace=0.30,
        )
        adjustment_ax = self.plots_figure.add_subplot(grid[0, 0])
        percent_ax = self.plots_figure.add_subplot(grid[1, 0], sharex=adjustment_ax)
        info_ax = self.plots_figure.add_subplot(grid[:, 1])
        info_ax.axis("off")
        info_ax.set_title("How to read it", fontweight="semibold", fontsize=9)

        material_colors = {
            "TiO2": "#147d77",
            "SiO2": "#5b8db8",
            "Ag": "#6b7280",
            "Au": "#c08c1a",
        }
        material_order = {"TiO2": 0, "SiO2": 1, "Ag": 2, "Au": 3}
        materials = sorted(
            {str(row["material"]) for row in rows},
            key=lambda material: (material_order.get(material, 99), material),
        )

        for material in materials:
            material_rows = [row for row in rows if str(row["material"]) == material]
            x = np.asarray([float(row["base_thickness_nm"]) for row in material_rows], dtype=float)
            adjustment = np.asarray([float(row["adjustment_nm"]) for row in material_rows], dtype=float)
            percent = np.asarray([float(row["percent_change"]) for row in material_rows], dtype=float)
            color = material_colors.get(material, "#334155")
            adjustment_ax.scatter(
                x,
                adjustment,
                s=36,
                color=color,
                alpha=0.58,
                edgecolors="#111827",
                linewidths=0.35,
                label=f"{material} n={len(material_rows)}",
            )
            percent_ax.scatter(
                x,
                percent,
                s=30,
                color=color,
                alpha=0.48,
                edgecolors="#111827",
                linewidths=0.25,
            )
            for ax, y_values in ((adjustment_ax, adjustment), (percent_ax, percent)):
                trend_x, trend_y = self._binned_median_points(x, y_values)
                if trend_x.size:
                    ax.plot(
                        trend_x,
                        trend_y,
                        color=color,
                        linewidth=2.2,
                        marker="o",
                        markersize=4,
                        alpha=0.95,
                    )

        for ax in (adjustment_ax, percent_ax):
            ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.55)
            ax.grid(True, alpha=0.24)
            ax.tick_params(axis="both", labelsize=7.5)
        adjustment_ax.set_ylabel("Adjustment nm", fontsize=8.5)
        percent_ax.set_ylabel("Adjustment %", fontsize=8.5)
        percent_ax.set_xlabel("Starting thickness estimate (nm)", fontsize=8.5)
        adjustment_ax.legend(fontsize=7, loc="best")

        measurement_count = len({(row["sample_name"], row["measurement"]) for row in rows})
        title = (
            "Individual thickness-fit adjustments vs starting thickness\n"
            f"{filter_label}; {len(rows)} layers from {measurement_count} measurements"
        )
        adjustment_ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)

        lines = [
            "Each point is one layer from the best saved individual thickness fit for one measurement.",
            "",
            "Positive adjustment means the fit made the layer thicker than the sputter-rate estimate.",
            "The solid line is the binned median trend for each material.",
            "",
            f"Constants: {self._summarize_row_values(rows, 'profile_name', 'unknown')}",
            f"Model: {self._summarize_row_values(rows, 'model_label', 'unknown')}",
            "",
            "Median by material:",
        ]
        for material in materials:
            material_rows = [row for row in rows if str(row["material"]) == material]
            adjustment_values = np.asarray([float(row["adjustment_nm"]) for row in material_rows], dtype=float)
            percent_values = np.asarray([float(row["percent_change"]) for row in material_rows], dtype=float)
            lines.append(
                f"{material}: n={len(material_rows)}, "
                f"{np.nanmedian(adjustment_values):+.2f} nm, "
                f"{np.nanmedian(percent_values):+.2f}%"
            )
        info_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=7.3,
            family="Consolas",
            wrap=True,
        )
        self.plots_info_var.set(
            f"Individual thickness adjustments: {len(rows)} layer corrections from "
            f"{measurement_count} measurements."
        )
        self.plots_figure.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.88)
        self.plots_canvas.draw_idle()

    def draw_fit_impact_workflow_plot(self) -> None:
        if not hasattr(self, "plots_figure"):
            return
        try:
            try:
                if self.experiment_store is None:
                    self.load_experiment_samples(show_errors=False)
                if self.experiment_cache is None:
                    self.load_experiment_cache()
            except Exception:
                pass
            rows = self._fit_impact_rows()
            if not rows:
                raise ValueError(
                    "No saved fit-impact results were found yet. Build an experiment cache, "
                    "run thickness fits, benchmark models/constants, fit RI candidates, "
                    "or run the empirical fit first."
                )
            summary_path = self._save_fit_impact_summary(rows)
            self._draw_fit_impact_workflow_plot(rows, summary_path)
        except Exception as exc:
            messagebox.showerror("Fit impact", str(exc))

    @staticmethod
    def _fit_impact_float(value: object, default: float = float("nan")) -> float:
        try:
            number = float(value)
        except Exception:
            return default
        return number if np.isfinite(number) else default

    def _append_fit_impact_row(
        self,
        rows: list[dict[str, object]],
        *,
        family: str,
        method: str,
        before: object,
        after: object,
        count: int,
        source: str,
        detail: str = "",
        point_improvements: list[float] | None = None,
        output_path: Path | str | None = None,
    ) -> None:
        before_value = self._fit_impact_float(before)
        after_value = self._fit_impact_float(after)
        if not np.isfinite(before_value) and not np.isfinite(after_value):
            return
        improvement = before_value - after_value if np.isfinite(before_value) and np.isfinite(after_value) else float("nan")
        rows.append(
            {
                "family": family,
                "method": method,
                "before": before_value,
                "after": after_value,
                "improvement": improvement,
                "count": int(count),
                "source": source,
                "detail": detail,
                "point_improvements": [
                    float(value)
                    for value in (point_improvements or [])
                    if np.isfinite(self._fit_impact_float(value))
                ],
                "output_path": "" if output_path is None else str(output_path),
            }
        )

    def _fit_impact_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        rows.extend(self._fit_impact_rows_from_cached_thickness())
        rows.extend(self._fit_impact_rows_from_thickness_summary())
        rows.extend(self._fit_impact_rows_from_experiment_cache_profiles())
        rows.extend(self._fit_impact_rows_from_material_candidates())
        rows.extend(self._fit_impact_rows_from_model_benchmark())
        rows.extend(self._fit_impact_rows_from_configuration_fits())
        rows.extend(self._fit_impact_rows_from_roughness_fits())
        rows.extend(self._fit_impact_rows_from_empirical_fits())
        return sorted(
            rows,
            key=lambda row: (
                -self._fit_impact_float(row.get("improvement")),
                self._fit_impact_float(row.get("after"), float("inf")),
                str(row.get("family", "")),
                str(row.get("method", "")),
            ),
        )

    def _fit_impact_rows_from_cached_thickness(self) -> list[dict[str, object]]:
        if self.experiment_cache is None:
            return []
        source_rows = self._colour_distance_rows()
        if not source_rows:
            return []
        output_rows: list[dict[str, object]] = []
        modes: tuple[tuple[str | None, str], ...] = (
            ("layer", "Individual thickness"),
            ("material_rate", "Grouped sputter-rate thickness"),
            (None, "Best cached thickness"),
        )
        for mode, label in modes:
            before_values: list[float] = []
            after_values: list[float] = []
            improvements: list[float] = []
            records_used: list[dict[str, object]] = []
            for row in source_rows:
                record = self._best_cached_thickness_fit_for_measurement(
                    str(row["sample_name"]),
                    str(row["measurement"]),
                    optimization_mode=mode,
                )
                if record is None:
                    continue
                before = self._fit_impact_float(row.get("model_delta"))
                after = self._fit_impact_float(record.get("delta_e"))
                if not np.isfinite(after):
                    continue
                after_values.append(after)
                records_used.append(record)
                if np.isfinite(before):
                    before_values.append(before)
                    improvements.append(before - after)
            if not after_values:
                continue
            method = label
            range_label = self._thickness_cache_range_label(records_used)
            if range_label:
                method = f"{method} ({range_label})"
            if any(bool(record.get("fit_reflectance_scale", False)) for record in records_used):
                method = f"{method} + reflectance scale"
            detail = self._fit_impact_cached_record_detail(records_used)
            self._append_fit_impact_row(
                output_rows,
                family="Thickness",
                method=method,
                before=np.mean(before_values) if before_values else float("nan"),
                after=np.mean(after_values),
                count=len(after_values),
                source="thickness cache",
                detail=detail,
                point_improvements=improvements,
            )
        return output_rows

    @staticmethod
    def _thickness_cache_range_label(records: list[dict[str, object]]) -> str:
        ranges = sorted(
            {
                round(float(record["range_percent"]), 6)
                for record in records
                if np.isfinite(ThinFilmDesignerApp._fit_impact_float(record.get("range_percent")))
            }
        )
        steps = sorted(
            {
                round(float(record["step_percent"]), 6)
                for record in records
                if np.isfinite(ThinFilmDesignerApp._fit_impact_float(record.get("step_percent")))
            }
        )
        if not ranges:
            return ""
        range_text = f"+/-{ranges[0]:g}%" if len(ranges) == 1 else "mixed ranges"
        step_text = f", {steps[0]:g}% step" if len(steps) == 1 else ""
        return f"{range_text}{step_text}"

    @staticmethod
    def _fit_impact_cached_record_detail(records: list[dict[str, object]]) -> str:
        if not records:
            return ""
        profiles = sorted({str(record.get("profile_name", "") or "unknown") for record in records})
        models = sorted({str(record.get("model_label", "") or "unknown") for record in records})
        profile_text = ", ".join(profiles[:3]) + ("..." if len(profiles) > 3 else "")
        model_text = ", ".join(models[:2]) + ("..." if len(models) > 2 else "")
        return f"{len(records)} cached measurements; constants {profile_text}; model {model_text}"

    def _fit_impact_rows_from_thickness_summary(self) -> list[dict[str, object]]:
        path = Path(__file__).resolve().parent / "outputs" / "thickness_optimization_summary" / "thickness_optimization_summary.csv"
        if not path.exists():
            return []
        try:
            data = pd.read_csv(path)
        except Exception:
            return []
        required = {"sample_name", "measurement_description", "base_delta_e", "optimized_delta_e"}
        if not required.issubset(set(data.columns)):
            return []
        if "colour_metric" in data.columns:
            metric = self._current_colour_metric()
            matching = data[data["colour_metric"].astype(str).str.lower().eq(metric)]
            if not matching.empty:
                data = matching
        unique = data.drop_duplicates(["sample_name", "measurement_description"]).copy()
        if unique.empty:
            return []
        before = pd.to_numeric(unique["base_delta_e"], errors="coerce")
        after = pd.to_numeric(unique["optimized_delta_e"], errors="coerce")
        improvements = list((before - after).dropna().astype(float))
        rows: list[dict[str, object]] = []
        self._append_fit_impact_row(
            rows,
            family="Thickness",
            method="Saved overnight thickness summary",
            before=before.mean(),
            after=after.mean(),
            count=int(after.notna().sum()),
            source="thickness summary CSV",
            detail="One row per sample/measurement; duplicate layer rows are collapsed before averaging.",
            point_improvements=improvements,
            output_path=path,
        )
        return rows

    def _fit_impact_rows_from_experiment_cache_profiles(self) -> list[dict[str, object]]:
        cache_dir = Path(__file__).resolve().parent / "outputs" / "experiment_cache"
        if not cache_dir.exists():
            return []
        metric = self._current_colour_metric()
        base_name = "experiment_results_ciede2000.csv" if metric == COLOUR_METRIC_CIEDE2000 else "experiment_results.csv"
        base_path = cache_dir / base_name
        if not base_path.exists():
            fallback = cache_dir / "experiment_results.csv"
            base_path = fallback if fallback.exists() else base_path
        if not base_path.exists():
            return []
        try:
            base = pd.read_csv(base_path)
        except Exception:
            return []
        if "delta_e" not in base.columns:
            return []
        key_columns = [
            column
            for column in ("sample_name", "measurement_description", "source_system")
            if column in base.columns
        ]
        if len(key_columns) < 2:
            return []
        rows: list[dict[str, object]] = []
        files = sorted(cache_dir.glob("experiment_results*.csv"))
        for path in files:
            if path == base_path:
                continue
            try:
                candidate = pd.read_csv(path)
            except Exception:
                continue
            if "delta_e" not in candidate.columns or not set(key_columns).issubset(candidate.columns):
                continue
            if "colour_metric" in candidate.columns:
                metric_rows = candidate[candidate["colour_metric"].astype(str).str.lower().eq(metric)]
                if not metric_rows.empty:
                    candidate = metric_rows
            merged = base[key_columns + ["delta_e"]].merge(
                candidate[key_columns + ["delta_e"]],
                on=key_columns,
                suffixes=("_before", "_after"),
            )
            if merged.empty:
                continue
            before = pd.to_numeric(merged["delta_e_before"], errors="coerce")
            after = pd.to_numeric(merged["delta_e_after"], errors="coerce")
            label = self._experiment_cache_profile_label(path)
            family = "Refractive index" if "fitted" in path.stem or "constant" in path.stem else "Model cache"
            self._append_fit_impact_row(
                rows,
                family=family,
                method=label,
                before=before.mean(),
                after=after.mean(),
                count=int(after.notna().sum()),
                source="experiment cache profile",
                detail=f"Compared against {base_path.name}",
                point_improvements=list((before - after).dropna().astype(float)),
                output_path=path,
            )
        return rows

    @staticmethod
    def _experiment_cache_profile_label(path: Path) -> str:
        label = path.stem.removeprefix("experiment_results").strip("_")
        label = label.removesuffix("_ciede2000").replace("_", " ").strip()
        if not label:
            label = "experiment model cache"
        if label == "fitted single films":
            return "Fitted single-film refractive index profile"
        return label[:1].upper() + label[1:]

    def _fit_impact_rows_from_material_candidates(self) -> list[dict[str, object]]:
        root = Path(__file__).resolve().parent / "outputs" / "material_candidate_fits"
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(root.glob("candidate_fit_summary*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidate_rows = payload.get("rows", [])
            if not isinstance(candidate_rows, list):
                continue
            suffix = path.stem.removeprefix("candidate_fit_summary").strip("_") or "all groups"
            materials = sorted({str(row.get("material_name", "")) for row in candidate_rows if isinstance(row, dict)})
            for material_name in materials:
                material_rows = [
                    row
                    for row in candidate_rows
                    if isinstance(row, dict) and str(row.get("material_name", "")) == material_name
                ]
                if not material_rows:
                    continue
                current_rows = [
                    row for row in material_rows if str(row.get("candidate_label", "")).lower() == "current"
                ]
                best_row = min(
                    material_rows,
                    key=lambda row: self._fit_impact_float(row.get("mean_delta_e"), float("inf")),
                )
                if not current_rows:
                    continue
                current_row = current_rows[0]
                before = self._fit_impact_float(current_row.get("mean_delta_e"))
                after = self._fit_impact_float(best_row.get("mean_delta_e"))
                count = int(self._fit_impact_float(best_row.get("spectrum_count"), 0.0))
                best_label = str(best_row.get("candidate_label", "unknown"))
                self._append_fit_impact_row(
                    rows,
                    family="Refractive index",
                    method=f"{material_name}: best RI candidate ({suffix})",
                    before=before,
                    after=after,
                    count=count,
                    source="refractiveindex.info candidate fit",
                    detail=f"best candidate {best_label}; current candidate Delta E {before:.2f}",
                    output_path=path,
                )
        return rows

    def _fit_impact_rows_from_model_benchmark(self) -> list[dict[str, object]]:
        root = Path(__file__).resolve().parent / "outputs" / "model_constant_benchmark"
        if not root.exists():
            return []
        summary_files = sorted(root.glob("model_constant_delta_e_summary_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not summary_files:
            return []
        path = summary_files[0]
        try:
            summary = pd.read_csv(path)
        except Exception:
            return []
        required = {"experiment_group", "constants_profile", "optical_model", "mean_delta_e", "count"}
        if not required.issubset(summary.columns):
            return []
        all_rows = summary[summary["experiment_group"].astype(str).eq("All samples")].copy()
        if all_rows.empty:
            all_rows = summary.copy()
        best = all_rows.sort_values("mean_delta_e").iloc[0]
        current = all_rows[
            all_rows["constants_profile"].astype(str).eq(self.material_profile_var.get())
            & all_rows["optical_model"].astype(str).eq(self.model_mode_var.get())
        ]
        before = float(current["mean_delta_e"].min()) if not current.empty else float("nan")
        rows: list[dict[str, object]] = []
        self._append_fit_impact_row(
            rows,
            family="Optical model",
            method="Best constants + optical model benchmark",
            before=before,
            after=float(best["mean_delta_e"]),
            count=int(best["count"]),
            source="model/constants benchmark",
            detail=(
                f"best {best['constants_profile']} / {best['optical_model']}; "
                "before is current GUI selection when present in the benchmark"
            ),
            output_path=path,
        )
        return rows

    def _fit_impact_rows_from_configuration_fits(self) -> list[dict[str, object]]:
        root = Path(__file__).resolve().parent / "outputs" / "configuration_fit"
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for summary_path in sorted(root.glob("**/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            configuration = str(payload.get("configuration", summary_path.parent.name))
            candidate_path = Path(str(payload.get("candidate_csv", "")))
            stage_path = Path(str(payload.get("stage_csv", "")))
            if candidate_path.exists():
                try:
                    candidates = pd.read_csv(candidate_path)
                except Exception:
                    candidates = pd.DataFrame()
                if not candidates.empty and "mean_delta_e" in candidates.columns:
                    best = candidates.sort_values("mean_delta_e").iloc[0]
                    current = candidates[
                        candidates["constants_profile"].astype(str).eq(self.material_profile_var.get())
                        & candidates["optical_model"].astype(str).eq(self.model_mode_var.get())
                    ]
                    before = float(current["mean_delta_e"].min()) if not current.empty else float("nan")
                    self._append_fit_impact_row(
                        rows,
                        family="Optical model",
                        method=f"{configuration}: constants/model choice",
                        before=before,
                        after=float(best["mean_delta_e"]),
                        count=int(best.get("measurement_count", 0)),
                        source="configuration fit",
                        detail=f"best {best.get('constants_profile')} / {best.get('optical_model')} / {best.get('model_parameters')}",
                        output_path=candidate_path,
                    )
            if stage_path.exists():
                try:
                    stages = pd.read_csv(stage_path)
                except Exception:
                    stages = pd.DataFrame()
                if {"stage", "mean_delta_e"}.issubset(stages.columns):
                    base_stage = stages[stages["stage"].astype(str).str.contains("constants", case=False, na=False)]
                    thickness_stage = stages[stages["stage"].astype(str).str.contains("thickness", case=False, na=False)]
                    if not base_stage.empty and not thickness_stage.empty:
                        base = base_stage.iloc[0]
                        after = thickness_stage.iloc[-1]
                        self._append_fit_impact_row(
                            rows,
                            family="Thickness",
                            method=f"{configuration}: thickness after model choice",
                            before=float(base["mean_delta_e"]),
                            after=float(after["mean_delta_e"]),
                            count=int(after.get("measurement_count", base.get("measurement_count", 0))),
                            source="configuration fit",
                            detail="Shows the extra improvement from thickness fitting after the best constants/model were selected.",
                            output_path=stage_path,
                        )
        return rows

    def _fit_impact_rows_from_roughness_fits(self) -> list[dict[str, object]]:
        root = Path(__file__).resolve().parent / "outputs" / "roughness_fits"
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(root.glob("roughness_fit_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            group_label = str(payload.get("group_label", path.stem.removeprefix("roughness_fit_")))
            before = self._baseline_delta_for_roughness_group(group_label)
            after = self._fit_impact_float(payload.get("mean_delta_e"))
            detail = (
                f"RMS {self._fit_impact_float(payload.get('rms_roughness_nm')):.2f} nm, "
                f"scale {self._fit_impact_float(payload.get('scatter_scale')):.2f}, "
                f"exponent {self._fit_impact_float(payload.get('scatter_exponent')):.2f}"
            )
            self._append_fit_impact_row(
                rows,
                family="Roughness",
                method=f"{group_label}: diffuse redistribution fit",
                before=before,
                after=after,
                count=int(self._fit_impact_float(payload.get("spectrum_count"), 0.0)),
                source="roughness fit",
                detail=detail,
                output_path=path,
            )
        return rows

    def _baseline_delta_for_roughness_group(self, group_label: str) -> float:
        if self.experiment_cache is None or self.experiment_cache.count == 0:
            return float("nan")
        parts = group_label.rsplit("_", 1)
        substrate_text = parts[0].replace("_", " ").lower() if parts else ""
        surface_text = parts[1].lower() if len(parts) == 2 else ""
        substrates = self.experiment_cache.substrate_classes.astype(str)
        surfaces = self.experiment_cache.surface_classes.astype(str)
        mask = []
        for substrate, surface in zip(substrates, surfaces):
            substrate_ok = not substrate_text or substrate_text in substrate.lower()
            surface_ok = not surface_text or surface_text in surface.lower()
            mask.append(substrate_ok and surface_ok)
        mask_array = np.asarray(mask, dtype=bool)
        if not np.any(mask_array):
            return float("nan")
        values = np.asarray(self.experiment_cache.delta_e[mask_array], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def _fit_impact_rows_from_empirical_fits(self) -> list[dict[str, object]]:
        root = Path(__file__).resolve().parent / "outputs" / "empirical_fit"
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for predictions_path in sorted(root.glob("*/empirical_predictions.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                predictions = pd.read_csv(predictions_path)
            except Exception:
                continue
            required = {"split", "baseline_delta_e", "fitted_delta_e"}
            if not required.issubset(predictions.columns):
                continue
            for split in ("train", "validation"):
                subset = predictions[predictions["split"].astype(str).eq(split)].copy()
                if subset.empty:
                    continue
                before = pd.to_numeric(subset["baseline_delta_e"], errors="coerce")
                after = pd.to_numeric(subset["fitted_delta_e"], errors="coerce")
                self._append_fit_impact_row(
                    rows,
                    family="Empirical",
                    method=f"Loose empirical n/k fit ({split})",
                    before=before.mean(),
                    after=after.mean(),
                    count=int(after.notna().sum()),
                    source="empirical fit",
                    detail=f"Folder {predictions_path.parent.name}; validation is the honest check of prediction strength.",
                    point_improvements=list((before - after).dropna().astype(float)),
                    output_path=predictions_path,
                )
        return rows

    def _save_fit_impact_summary(self, rows: list[dict[str, object]]) -> Path:
        output_dir = Path(__file__).resolve().parent / "outputs" / "fit_impact"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "fit_impact_summary.csv"
        serializable = [
            {
                key: value
                for key, value in row.items()
                if key != "point_improvements"
            }
            for row in rows
        ]
        pd.DataFrame(serializable).to_csv(path, index=False)
        return path

    @staticmethod
    def _fit_impact_short_label(row: dict[str, object], width: int = 34) -> str:
        family = str(row.get("family", "")).strip()
        method = str(row.get("method", "")).strip()
        replacements = (
            ("best RI candidate", "RI candidate"),
            ("Grouped sputter-rate thickness", "Grouped thickness"),
            ("Saved overnight thickness summary", "Overnight thickness"),
            ("Fitted single-film refractive index profile", "Single-film RI profile"),
            ("diffuse redistribution fit", "roughness fit"),
            ("refractiveindex.info", "RI.info"),
            (" + reflectance scale", " + scale"),
        )
        for old, new in replacements:
            method = method.replace(old, new)
        label = f"{family}: {method}" if family and family not in method else method
        return textwrap.shorten(label, width=width, placeholder="...")

    def _draw_fit_impact_workflow_plot(self, rows: list[dict[str, object]], summary_path: Path) -> None:
        delta_label = self._delta_e_label()
        finite_improvement_rows = [
            row for row in rows if np.isfinite(self._fit_impact_float(row.get("improvement")))
        ]
        finite_after_rows = [
            row for row in rows if np.isfinite(self._fit_impact_float(row.get("after")))
        ]
        ranked_rows = sorted(
            finite_improvement_rows,
            key=lambda row: self._fit_impact_float(row.get("improvement")),
            reverse=True,
        )
        if not ranked_rows:
            ranked_rows = sorted(
                finite_after_rows,
                key=lambda row: self._fit_impact_float(row.get("after"), float("inf")),
            )

        self.plots_figure.clear()
        grid = self.plots_figure.add_gridspec(
            2,
            2,
            width_ratios=[1.30, 1.0],
            height_ratios=[1.05, 1.0],
            hspace=0.48,
            wspace=0.28,
        )
        rank_ax = self.plots_figure.add_subplot(grid[0, 0])
        bars_ax = self.plots_figure.add_subplot(grid[0, 1])
        spread_ax = self.plots_figure.add_subplot(grid[1, 0])
        notes_ax = self.plots_figure.add_subplot(grid[1, 1])

        family_colours = {
            "Thickness": "#147d77",
            "Optical model": "#4f6fad",
            "Refractive index": "#c27803",
            "Roughness": "#7c3aed",
            "Empirical": "#ef6c00",
            "Model cache": "#64748b",
        }

        plot_rows = list(reversed(ranked_rows[:10]))
        if plot_rows:
            values = [self._fit_impact_float(row.get("improvement")) for row in plot_rows]
            if not any(np.isfinite(value) for value in values):
                values = [-self._fit_impact_float(row.get("after")) for row in plot_rows]
                rank_ax.set_xlabel(f"Lower {delta_label} is better")
            else:
                rank_ax.set_xlabel(f"Mean {delta_label} reduction")
            labels = [
                self._fit_impact_short_label(row, width=35)
                for row in plot_rows
            ]
            colours = [family_colours.get(str(row.get("family", "")), "#8aa6b8") for row in plot_rows]
            rank_ax.barh(np.arange(len(plot_rows)), values, color=colours, alpha=0.9)
            rank_ax.set_yticks([])
            finite_values = [value for value in values if np.isfinite(value)]
            value_label_x = 0.0
            if finite_values:
                max_value = max(max(finite_values), 0.1)
                min_value = min(min(finite_values), 0.0)
                rank_ax.set_xlim(min_value - 0.04 * max_value, max_value * 1.22)
                value_label_x = max_value * 1.04
            for index, label in enumerate(labels):
                rank_ax.text(
                    0.01,
                    index,
                    label,
                    transform=rank_ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=7.2,
                    color="#111827",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                )
            rank_ax.axvline(0.0, color="#111827", linewidth=0.8, alpha=0.55)
            for index, (row, value) in enumerate(zip(plot_rows, values)):
                if np.isfinite(value):
                    rank_ax.text(
                        value_label_x if value_label_x else value,
                        index,
                        f"{value:.2f}  n={int(row.get('count', 0))}",
                        va="center",
                        ha="left",
                        fontsize=7,
                    )
        else:
            rank_ax.text(0.5, 0.5, "No finite fit-impact rows yet.", ha="center", va="center")
            rank_ax.set_axis_off()
        rank_ax.set_title("Largest mean change", fontsize=9, fontweight="semibold", pad=6)
        rank_ax.grid(True, axis="x", alpha=0.25)

        before_after_rows = [
            row for row in ranked_rows if np.isfinite(self._fit_impact_float(row.get("before")))
            and np.isfinite(self._fit_impact_float(row.get("after")))
        ][:7]
        if not before_after_rows:
            before_after_rows = finite_after_rows[:7]
        if before_after_rows:
            labels = [
                self._fit_impact_short_label(row, width=30)
                for row in before_after_rows
            ]
            y_values = np.arange(len(before_after_rows))
            before_values = np.asarray(
                [self._fit_impact_float(row.get("before")) for row in before_after_rows],
                dtype=float,
            )
            after_values = np.asarray(
                [self._fit_impact_float(row.get("after")) for row in before_after_rows],
                dtype=float,
            )
            height = 0.36
            if np.any(np.isfinite(before_values)):
                bars_ax.barh(
                    y_values + height / 2.0,
                    np.nan_to_num(before_values, nan=0.0),
                    height=height,
                    color="#9fb3c4",
                    label="Before",
                )
                bars_ax.barh(
                    y_values - height / 2.0,
                    np.nan_to_num(after_values, nan=0.0),
                    height=height,
                    color="#ef6c00",
                    label="After",
                )
            else:
                bars_ax.barh(y_values, after_values, color="#ef6c00", label="After")
            bars_ax.set_yticks([])
            for index, label in enumerate(labels):
                bars_ax.text(
                    0.01,
                    index,
                    label,
                    transform=bars_ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=6.6,
                    color="#111827",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
                )
            bars_ax.set_xlabel(delta_label)
            bars_ax.legend(fontsize=7)
            bars_ax.grid(True, axis="x", alpha=0.25)
        else:
            bars_ax.text(0.5, 0.5, "No before/after rows to plot.", ha="center", va="center")
            bars_ax.set_axis_off()
        bars_ax.set_title("Before and after means", fontsize=9, fontweight="semibold", pad=6)

        spread_rows = [
            row
            for row in ranked_rows
            if len(row.get("point_improvements", [])) >= 2
        ][:7]
        if spread_rows:
            for index, row in enumerate(spread_rows):
                values = np.asarray(row.get("point_improvements", []), dtype=float)
                values = values[np.isfinite(values)]
                if values.size > 120:
                    keep = np.linspace(0, values.size - 1, 120).astype(int)
                    values = np.sort(values)[keep]
                if not values.size:
                    continue
                jitter = np.linspace(-0.22, 0.22, values.size)
                spread_ax.scatter(
                    np.full(values.size, index, dtype=float) + jitter,
                    values,
                    s=18,
                    color=family_colours.get(str(row.get("family", "")), "#147d77"),
                    alpha=0.58,
                    edgecolors="none",
                )
                spread_ax.plot(
                    [index - 0.25, index + 0.25],
                    [float(np.mean(values)), float(np.mean(values))],
                    color="#111827",
                    linewidth=1.2,
                )
            spread_ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.55)
            spread_ax.set_xticks(
                np.arange(len(spread_rows)),
                [str(index + 1) for index in range(len(spread_rows))],
                fontsize=7,
            )
            spread_ax.set_xlabel("Fit number; see guide", fontsize=8)
            spread_ax.set_ylabel(f"{delta_label} reduction", fontsize=8, labelpad=2)
            spread_ax.grid(True, axis="y", alpha=0.25)
        else:
            spread_ax.text(
                0.5,
                0.5,
                "Run per-measurement thickness or empirical fits to see spread.",
                ha="center",
                va="center",
                wrap=True,
            )
            spread_ax.set_axis_off()
        spread_ax.set_title("Measurement-by-measurement effect", fontsize=9, fontweight="semibold", pad=6)

        notes_ax.set_axis_off()
        notes = [
            "Positive change = lower Delta E.",
            "No before value = after-fit level only.",
            "",
        ]
        present_families = {str(row.get("family", "")) for row in rows}
        missing_sources = []
        expected_sources = (
            ("Thickness", "run an overnight or selected thickness cache"),
            ("Optical model", "run Benchmark constants/models or Configuration Fit"),
            ("Refractive index", "run RI candidate fitting or single-film constants"),
            ("Roughness", "run a roughness group fit"),
            ("Empirical", "run the Empirical Fit tab"),
        )
        for family, action in expected_sources:
            if family not in present_families:
                missing_sources.append(f"{family}: {action}")
        if missing_sources:
            notes.append("Not shown yet:")
            notes.extend(
                f"  {textwrap.shorten(text, width=48, placeholder='...')}"
                for text in missing_sources[:4]
            )
            notes.append("")
        if spread_rows:
            notes.append("Fit numbers in lower-left plot:")
            for index, row in enumerate(spread_rows, start=1):
                notes.append(f"  {index}. {self._fit_impact_short_label(row, width=42)}")
            notes.append("")
        notes.append("Top reductions:")
        for row in rows[:3]:
            before = self._fit_impact_float(row.get("before"))
            after = self._fit_impact_float(row.get("after"))
            improvement = self._fit_impact_float(row.get("improvement"))
            before_text = "n/a" if not np.isfinite(before) else f"{before:.2f}"
            after_text = "n/a" if not np.isfinite(after) else f"{after:.2f}"
            improvement_text = "n/a" if not np.isfinite(improvement) else f"{improvement:+.2f}"
            notes.append(
                f"{self._fit_impact_short_label(row, width=45)}"
            )
            notes.append(
                f"  {before_text} -> {after_text}; change {improvement_text}; n={int(row.get('count', 0))}"
            )
        notes_ax.text(
            0.0,
            1.0,
            "\n".join(notes),
            va="top",
            ha="left",
            fontsize=5.9,
            family="monospace",
            linespacing=1.10,
        )
        notes_ax.set_title("Guide", fontsize=9, fontweight="semibold", pad=6)

        self.plots_figure.suptitle(
            f"Fit impact dashboard ({delta_label})\n"
            "Thickness, optical model/constants, refractive index, roughness, and empirical fits",
            fontsize=10.0,
            fontweight="semibold",
        )
        self.plots_info_var.set(
            f"Fit impact collected {len(rows)} saved fit summaries and wrote {summary_path}."
        )
        self.plots_figure.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.86)
        self.plots_canvas.draw_idle()

    def _cached_fit_label_from_path(self, path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                return path.name
            return f"{metadata.get('profile_name', 'unknown')} / {metadata.get('model_label', 'unknown')}"
        except Exception:
            return path.name

    def _colour_distance_current_display_rows(
        self,
        hydrate_fit_colours: bool = True,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if self.experiment_cache is None:
            try:
                self.load_experiment_cache()
            except Exception:
                pass
        if self.experiment_cache is None:
            return [], []
        if self.experiment_store is None:
            self.load_experiment_samples(show_errors=False)
        rows = self._colour_distance_rows()
        display_rows = self._colour_distance_display_rows(rows)
        if hydrate_fit_colours:
            self._hydrate_colour_distance_fit_colours(display_rows)
        for row in display_rows:
            fit_delta = row.get("fit_delta")
            if fit_delta is not None and float(fit_delta) > float(row["model_delta"]):
                row["fit_delta"] = None
                row["fit_rgb"] = None
                row["improvement"] = None
                row["fit_path"] = None
                row["fit_label"] = "Cached fit not better than active before model"
                row["fit_stage_label"] = ""
                row["fit_profile_name"] = ""
                row["fit_model_label"] = ""
        return rows, display_rows

    @staticmethod
    def _colour_distance_axis_positions(
        display_rows: list[dict[str, object]],
        max_labels: int = 48,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x = np.arange(len(display_rows), dtype=float)
        if not display_rows:
            return x, x, []
        step = max(1, int(np.ceil(len(display_rows) / max(1, max_labels))))
        tick_indices = np.arange(0, len(display_rows), step, dtype=int)
        labels = [str(display_rows[index]["sample_name"]) for index in tick_indices]
        return x, x[tick_indices], labels

    @staticmethod
    def _summarize_row_values(
        rows: list[dict[str, object]],
        key: str,
        fallback: str,
        limit: int = 2,
    ) -> str:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key, "")).strip() or fallback
            counts[value] = counts.get(value, 0) + 1
        if not counts:
            return fallback
        total = sum(counts.values())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        labels: list[str] = []
        for value, count in ordered[:limit]:
            labels.append(value if count == total else f"{value} ({count})")
        if len(ordered) > limit:
            labels.append("...")
        return ", ".join(labels)

    def _colour_distance_fit_basis_summary(
        self,
        display_rows: list[dict[str, object]],
        width: int = 150,
    ) -> str:
        fit_rows = [row for row in display_rows if row.get("fit_delta") is not None]
        if not fit_rows:
            return "Fit shown: no cached after-fit points"
        stage = self._summarize_row_values(fit_rows, "fit_stage_label", "Thickness fit")
        profile = self._summarize_row_values(fit_rows, "fit_profile_name", "unknown constants")
        model = self._summarize_row_values(fit_rows, "fit_model_label", "unknown optical model")
        text = f"Fit shown: {stage}; constants: {profile}; optical model: {model}"
        return textwrap.shorten(text, width=width, placeholder="...")

    def _plot_colour_distance_before_after_axis(
        self,
        ax,
        display_rows: list[dict[str, object]],
        *,
        title: str = "Measurement x-axis, before/after",
    ) -> None:
        delta_label = self._delta_e_label()
        if not display_rows:
            ax.text(0.5, 0.5, "No colour-distance rows", ha="center", va="center")
            ax.set_axis_off()
            return
        x_display, tick_positions, tick_labels = self._colour_distance_axis_positions(display_rows)
        model_display = np.asarray([float(row["model_delta"]) for row in display_rows], dtype=float)
        fit_display = np.asarray(
            [
                float(row["fit_delta"]) if row["fit_delta"] is not None else np.nan
                for row in display_rows
            ],
            dtype=float,
        )
        ax.bar(x_display, model_display, color="#8aa3b8", alpha=0.62, label=f"Before {delta_label}")
        finite_after = np.isfinite(fit_display)
        if np.any(finite_after):
            ax.scatter(
                x_display[finite_after],
                fit_display[finite_after],
                c="#f97316",
                s=25,
                edgecolors="#111827",
                linewidths=0.35,
                label=f"After {delta_label}",
                zorder=3,
            )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=6.7)
        ax.set_xlabel("Sample designation, grouped by series", fontsize=8)
        ax.set_ylabel(delta_label, fontsize=8)
        ax.set_title(
            f"{title}\n{self._colour_distance_fit_basis_summary(display_rows)}",
            fontsize=9.0,
            fontweight="semibold",
            pad=5,
        )
        ax.tick_params(axis="y", labelsize=7.4)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    def _plot_colour_distance_colours_axis(
        self,
        ax,
        display_rows: list[dict[str, object]],
        *,
        title: str = "Measured/after/before colours",
    ) -> None:
        if not display_rows:
            ax.text(0.5, 0.5, "No colour rows", ha="center", va="center")
            ax.set_axis_off()
            return
        _, tick_positions, tick_labels = self._colour_distance_axis_positions(display_rows)
        measured = np.asarray([row["measured_rgb"] for row in display_rows], dtype=float)
        before = np.asarray([row["model_rgb"] for row in display_rows], dtype=float)
        after = np.asarray(
            [
                row["fit_rgb"] if row["fit_rgb"] is not None else row["model_rgb"]
                for row in display_rows
            ],
            dtype=float,
        )
        ax.imshow(np.stack([measured, after, before], axis=0), aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Measured", "After", "Before"], fontsize=7.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=6.7)
        ax.set_xlabel("Sample designation, grouped by series", fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight="semibold", pad=5)
        ax.tick_params(axis="both", length=0)

    def download_colour_distance_before_after_figure(self) -> None:
        _, display_rows = self._colour_distance_current_display_rows(hydrate_fit_colours=True)
        if not display_rows:
            messagebox.showinfo("Download figure", "No colour-distance rows are available to save.")
            return
        figure = Figure(figsize=(7.8, 2.75), dpi=150)
        ax = figure.add_subplot(1, 1, 1)
        self._plot_colour_distance_before_after_axis(ax, display_rows)
        figure.subplots_adjust(left=0.08, right=0.985, bottom=0.34, top=0.86)
        self._download_figure(figure, "colour_distance_before_after")

    def download_colour_distance_colours_figure(self) -> None:
        _, display_rows = self._colour_distance_current_display_rows(hydrate_fit_colours=True)
        if not display_rows:
            messagebox.showinfo("Download figure", "No colour-distance rows are available to save.")
            return
        figure = Figure(figsize=(7.8, 2.25), dpi=150)
        ax = figure.add_subplot(1, 1, 1)
        self._plot_colour_distance_colours_axis(ax, display_rows)
        figure.subplots_adjust(left=0.08, right=0.985, bottom=0.40, top=0.82)
        self._download_figure(figure, "colour_distance_colours")

    def _draw_colour_distance_rows(
        self,
        rows: list[dict[str, object]],
        hydrate_fit_colours: bool = True,
    ) -> None:
        for item in self.colour_distance_tree.get_children():
            self.colour_distance_tree.delete(item)

        for row_number, row in enumerate(rows[:400]):
            fit_delta = row["fit_delta"]
            improvement = row["improvement"]
            self.colour_distance_tree.insert(
                "",
                tk.END,
                iid=str(row_number),
                values=(
                    f"{row['sample_name']} - {row['measurement']}",
                    row["series"],
                    row["substrate"],
                    row["surface"],
                    row["kind"],
                    f"{float(row['model_delta']):.2f}",
                    "" if fit_delta is None else f"{float(fit_delta):.2f}",
                    "" if improvement is None else f"{float(improvement):+.2f}",
                    row["fit_label"],
                ),
            )

        self.colour_distance_figure.clear()
        grid = self.colour_distance_figure.add_gridspec(2, 1, height_ratios=[1.25, 0.92], hspace=0.74)
        delta_ax = self.colour_distance_figure.add_subplot(grid[0, 0])
        colour_ax = self.colour_distance_figure.add_subplot(grid[1, 0])

        if not rows:
            for ax in (delta_ax, colour_ax):
                ax.text(0.5, 0.5, "No filtered experiment rows", ha="center", va="center")
                ax.set_axis_off()
            self.colour_distance_figure.subplots_adjust(
                left=0.08,
                right=0.97,
                bottom=0.10,
                top=0.90,
                hspace=0.50,
            )
            self.colour_distance_canvas.draw_idle()
            return

        delta_label = self._delta_e_label()
        display_rows = self._colour_distance_display_rows(rows)
        if hydrate_fit_colours:
            self._hydrate_colour_distance_fit_colours(display_rows)
        for row in display_rows:
            fit_delta = row.get("fit_delta")
            if fit_delta is not None and float(fit_delta) > float(row["model_delta"]):
                row["fit_delta"] = None
                row["fit_rgb"] = None
                row["improvement"] = None
                row["fit_path"] = None
                row["fit_label"] = "Cached fit not better than active before model"
                row["fit_stage_label"] = ""
                row["fit_profile_name"] = ""
                row["fit_model_label"] = ""
        self._plot_colour_distance_before_after_axis(delta_ax, display_rows)
        self._plot_colour_distance_colours_axis(colour_ax, display_rows)

        fit_rows = [row for row in display_rows if row["fit_delta"] is not None]
        fit_values = np.asarray([float(row["fit_delta"]) for row in fit_rows], dtype=float)
        summary = (
            f"{len(rows)} filtered measurements"
            if not fit_rows
            else f"{len(rows)} filtered measurements; {len(fit_rows)} with cached fits; "
            f"median fit {delta_label} {float(np.median(fit_values)):.2f}"
        )
        self.colour_distance_figure.suptitle(summary, fontsize=8.5, color="#52606d", y=0.985)
        self.colour_distance_figure.subplots_adjust(
            left=0.08,
            right=0.985,
            bottom=0.16,
            top=0.88,
            hspace=0.74,
        )
        self.colour_distance_canvas.draw_idle()
        self.status_var.set(summary)

    def _select_colour_distance_if_requested(self, redraw_only: bool) -> None:
        if not redraw_only and hasattr(self, "colour_distance_tab"):
            self.notebook.select(self.colour_distance_tab)

    def _load_cached_thickness_fit_result(self, path: Path) -> ThicknessOptimizationResult:
        if self.experiment_store is None:
            raise ValueError("Load experiment data first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Cache file has no metadata.")
        evaluations = data.get("evaluations", {})
        best_key = data.get("best_key")
        best = evaluations.get(best_key, {}) if isinstance(evaluations, dict) else {}
        if not isinstance(best, dict):
            raise ValueError("Cache file has no best trial.")
        colour_metric = normalise_colour_metric(metadata.get("colour_metric", COLOUR_METRIC_CIE76))

        sample_name = str(metadata["sample_name"])
        measurement_description = str(metadata["measurement_description"])
        sample = self.experiment_store.load_sample(sample_name)
        measurement_index = next(
            (
                idx
                for idx, measurement in enumerate(sample.measurements)
                if measurement.description == measurement_description
            ),
            None,
        )
        if measurement_index is None:
            raise ValueError("Cached measurement is no longer present in the imported data.")
        measurement = sample.measurements[measurement_index]

        profile_name = str(metadata.get("profile_name", self.material_profile_var.get()))
        model_label = str(metadata.get("model_label", self.model_mode_var.get()))
        model_settings = metadata.get("model_settings", {})
        if not isinstance(model_settings, dict):
            model_settings = {}
        materials = self._materials_for_profile(profile_name)
        model = self._model_from_cache_metadata(model_label, model_settings)
        substrate_name = str(metadata.get("substrate_name") or measurement.substrate_hint or self.substrate_var.get())
        angle_deg = float(metadata.get("angle_deg", self.angle_var.get()))
        wavelengths_nm = wavelength_grid(400.0, 700.0, int(metadata.get("wavelength_count", 151) or 151))
        interface_thickness_nm = float(metadata.get("interface_thickness_nm", self.roughness_thickness_var.get()))
        interface_fraction = float(metadata.get("interface_fraction", self.roughness_fraction_var.get()))
        use_effective_interfaces = bool(
            metadata.get("use_effective_interfaces", "Effective interface" in model_label)
        )
        native_oxide = self._native_oxide_from_cache_metadata(metadata, materials, substrate_name)

        stack = build_stack_from_estimates(
            sample,
            materials=materials,
            substrate_name=substrate_name,
            native_oxide=native_oxide,
            use_effective_interfaces=use_effective_interfaces,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
        )
        prepared = model.prepare_stack(stack, wavelengths_nm)
        base_d_list = np.asarray(prepared.base_d_list, dtype=float)
        base_reflectance = model.reflectance_from_prepared(prepared, base_d_list, angle_deg)

        best_offsets = [float(value) for value in best.get("offsets_percent", [])]
        variable_labels = best.get("variable_labels") or metadata.get("variable_labels") or []
        layer_percents = self._cached_layer_percents(sample.layer_estimates, best_offsets, variable_labels)
        optimized_d_list = base_d_list.copy()
        for tmm_index, percent in zip(prepared.display_layer_indices, layer_percents):
            optimized_d_list[tmm_index] = base_d_list[tmm_index] * (1.0 + percent / 100.0)
        optimized_reflectance = model.reflectance_from_prepared(prepared, optimized_d_list, angle_deg)
        base_scale = float(metadata.get("base_reflectance_scale", 1.0))
        optimized_scale = float(best.get("reflectance_scale", metadata.get("reflectance_scale", 1.0)))
        base_reflectance = np.clip(base_reflectance * base_scale, 0.0, 1.0)
        optimized_reflectance = np.clip(optimized_reflectance * optimized_scale, 0.0, 1.0)

        measured_wavelengths, measured_raw = load_reflectance_csv(measurement.csv_path)
        measured_reflectance = np.interp(wavelengths_nm, measured_wavelengths, measured_raw)
        color_cache = prepare_color_conversion(wavelengths_nm)
        measured_color = self._perceived_color_from_reflectance(measured_reflectance, color_cache)
        base_color = self._perceived_color_from_reflectance(base_reflectance, color_cache)
        optimized_color = self._perceived_color_from_reflectance(optimized_reflectance, color_cache)

        layer_results = []
        for layer, percent in zip(sample.layer_estimates, layer_percents):
            optimized_thickness = float(layer.thickness_nm) * (1.0 + float(percent) / 100.0)
            base_rate = layer.rate_nm_per_min
            optimized_rate = (
                optimized_thickness / layer.time_min
                if layer.time_min not in (None, 0)
                else None
            )
            layer_results.append(
                ThicknessOptimizationLayerResult(
                    material_name=layer.material_name,
                    base_thickness_nm=float(layer.thickness_nm),
                    optimized_thickness_nm=optimized_thickness,
                    percent_change=float(percent),
                    base_rate_nm_per_min=base_rate,
                    optimized_rate_nm_per_min=optimized_rate,
                    deposition_time_min=layer.time_min,
                )
            )

        return ThicknessOptimizationResult(
            sample_name=sample_name,
            measurement_description=measurement_description,
            stack_label=prepared.display_summary,
            wavelengths_nm=wavelengths_nm,
            measured_reflectance=measured_reflectance,
            base_reflectance=base_reflectance,
            optimized_reflectance=optimized_reflectance,
            measured_color=measured_color,
            base_color=base_color,
            optimized_color=optimized_color,
            base_delta_e=delta_e_colour(
                measured_color.xyz,
                base_color.xyz,
                metric=colour_metric,
            ),
            optimized_delta_e=float(
                best.get(
                    "delta_e",
                    delta_e_colour(
                        measured_color.xyz,
                        optimized_color.xyz,
                        metric=colour_metric,
                    ),
                )
            ),
            layer_results=tuple(layer_results),
            cache_path=path,
            evaluated_count=len(evaluations),
            reused_count=len(evaluations),
            new_count=0,
            colour_metric=colour_metric,
            reflectance_scale=optimized_scale,
            scale_fit_enabled=bool(metadata.get("fit_reflectance_scale", optimized_scale != 1.0)),
        )

    def _materials_for_profile(self, profile_name: str) -> dict[str, Material]:
        if profile_name == "fitted_single_films" and self.fitted_constants_path.exists():
            return load_fitted_materials(built_in_materials("current"), self.fitted_constants_path)
        if profile_name == "best_refractiveindex_candidates" and self.best_candidate_profile_path.exists():
            return load_best_candidate_materials(
                built_in_materials("current"),
                self.best_candidate_profile_path,
            )
        if profile_name.startswith("best_candidates_"):
            group_path = self._group_candidate_profile_path_from_name(profile_name)
            if group_path.exists():
                return load_best_candidate_materials(built_in_materials("current"), group_path)
        return built_in_materials(profile_name if profile_name in material_profile_names() else "current")

    def _model_from_cache_metadata(self, model_label: str, settings: dict[str, object]):
        label = model_label.lower()
        if "diffuse redistribution" in label:
            return TMMWithDiffuseRedistributionModel(
                DiffuseRedistributionSettings(
                    rms_roughness_nm=float(settings.get("rms_roughness_nm", self.rms_roughness_var.get())),
                    scatter_scale=float(settings.get("scatter_scale", self.scatter_scale_var.get())),
                    wavelength_exponent=float(settings.get("scatter_exponent", self.scatter_exponent_var.get())),
                    max_scatter_fraction=float(settings.get("max_scatter_fraction", self.scatter_max_var.get())),
                    diffuse_angle_min_deg=0.0,
                    diffuse_angle_max_deg=80.0,
                    diffuse_angle_samples=17,
                )
            )
        if "rms" in label:
            return TMMWithRoughnessModel(
                RoughnessCorrectionSettings(
                    rms_roughness_nm=float(settings.get("rms_roughness_nm", self.rms_roughness_var.get()))
                )
            )
        return TMMModel()

    def _native_oxide_from_cache_metadata(
        self,
        metadata: dict[str, object],
        materials: dict[str, Material],
        substrate_name: str,
    ) -> NativeOxide | None:
        native_data = metadata.get("native_oxide")
        if isinstance(native_data, list) and len(native_data) >= 2:
            material_name = str(native_data[0])
            if material_name in materials:
                try:
                    return NativeOxide(materials[material_name], float(native_data[1]))
                except (TypeError, ValueError):
                    return None
        default_oxide = native_oxide_for_substrate(materials, substrate_name)
        if default_oxide is None or not self.native_oxide_enabled_var.get():
            return None
        return NativeOxide(default_oxide.material, self.native_oxide_thickness_var.get())

    @staticmethod
    def _cached_layer_percents(layer_estimates, offsets: list[float], variable_labels) -> list[float]:
        labels = [str(label) for label in variable_labels] if isinstance(variable_labels, list) else []
        if len(offsets) == len(layer_estimates):
            return [float(value) for value in offsets]
        percents: list[float] = []
        for layer in layer_estimates:
            material = str(layer.material_name)
            index = next(
                (
                    idx
                    for idx, label in enumerate(labels)
                    if label == material or label.startswith(material)
                ),
                None,
            )
            percents.append(float(offsets[index]) if index is not None and index < len(offsets) else 0.0)
        return percents

    @staticmethod
    def _perceived_color_from_reflectance(reflectance, color_cache) -> PerceivedColor:
        xyz = reflectance_to_xyz(reflectance, cache=color_cache)
        srgb = reflectance_to_srgb(reflectance, cache=color_cache)
        srgb_255 = tuple(int(round(channel * 255.0)) for channel in srgb)
        return PerceivedColor(
            srgb=tuple(float(channel) for channel in srgb),
            srgb_255=srgb_255,
            xyz=tuple(float(value) for value in xyz),
        )

    @staticmethod
    def _safe_cache_prefix(text: str) -> str:
        import re

        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
        return cleaned.strip("_") or "thickness_optimization"

    def run_experiment_comparison(self) -> None:
        if self.experiment_store is None:
            messagebox.showerror("Experiments", "Load experiment data first.")
            return
        try:
            sample_name, measurement_index = self._selected_cache_or_control_measurement()
            sample = self.experiment_store.load_sample(sample_name)
            measurement = sample.measurements[measurement_index]
            substrate_name = measurement.substrate_hint or self.substrate_var.get()
            native_oxide = self._native_oxide_from_controls(substrate_name)
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            self._start_busy("comparing selected measurement")
            result = self.experiment_store.compare_sample(
                sample_name=sample_name,
                measurement_index=measurement_index,
                materials=self.materials,
                model=self._model_from_controls(),
                wavelengths_nm=wavelengths_nm,
                angle_deg=self.angle_var.get(),
                substrate_name=substrate_name,
                native_oxide=native_oxide,
                use_effective_interfaces=self._use_effective_interfaces(),
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
            )
            self._draw_experiment_result(result)
            self.notebook.select(self.experiments_tab)
            self._stop_busy(
                f"{sample_name}: simulated {result.simulated_color.hex}, "
                f"measured {result.measured_color.hex}"
            )
        except Exception as exc:
            self._stop_busy("Selected comparison failed.")
            messagebox.showerror("Experiments", str(exc))

    def _schedule_selected_experiment_live_compare(self) -> None:
        if not hasattr(self, "notebook") or self.experiment_store is None:
            return
        try:
            if self.notebook.tab(self.notebook.select(), "text") != "Experiments":
                return
        except tk.TclError:
            return
        if self.background_task_running:
            return
        if self.experiment_live_compare_job is not None:
            self.root.after_cancel(self.experiment_live_compare_job)
        self.experiment_live_compare_job = self.root.after(450, self._live_compare_selected_experiment)

    def _live_compare_selected_experiment(self) -> None:
        self.experiment_live_compare_job = None
        if self.background_task_running or self.experiment_store is None:
            return
        try:
            sample_name, measurement_index = self._selected_cache_or_control_measurement()
            sample = self.experiment_store.load_sample(sample_name)
            measurement = sample.measurements[measurement_index]
            substrate_name = measurement.substrate_hint or self.substrate_var.get()
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            result = self.experiment_store.compare_sample(
                sample_name=sample_name,
                measurement_index=measurement_index,
                materials=self.materials,
                model=self._model_from_controls(),
                wavelengths_nm=wavelengths_nm,
                angle_deg=self.angle_var.get(),
                substrate_name=substrate_name,
                native_oxide=self._native_oxide_from_controls(substrate_name),
                use_effective_interfaces=self._use_effective_interfaces(),
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
            )
            self._draw_experiment_result(result)
            self.experiment_info_var.set(
                f"Live comparison updated for {sample_name}: "
                f"measured {result.measured_color.hex}, simulated {result.simulated_color.hex}."
            )
        except Exception as exc:
            self.experiment_info_var.set(f"Could not update selected experiment live: {exc}")

    def optimize_selected_experiment_thicknesses(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Thickness optimization", "Load experiment data first.")
            return

        try:
            sample_name, measurement_index = self._selected_cache_or_control_measurement()
            sample = self.experiment_store.load_sample(sample_name)
            measurement = sample.measurements[measurement_index]
            substrate_name = measurement.substrate_hint or self.substrate_var.get()
            native_oxide = self._native_oxide_from_controls(substrate_name)
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)

            self.experiment_info_var.set("Optimizing thicknesses with cached TMM trials...")

            store = self.experiment_store
            materials = self.materials.copy()
            model = self._model_from_controls()
            angle_deg = self.angle_var.get()
            use_effective_interfaces = self._use_effective_interfaces()
            interface_thickness_nm = self.roughness_thickness_var.get()
            interface_fraction = self.roughness_fraction_var.get()
            model_settings = self._model_settings_signature()
            range_percent = self.thickness_opt_range_percent_var.get()
            step_percent = self.thickness_opt_step_percent_var.get()
            profile_name = self.material_profile_var.get()
            model_label = self.model_mode_var.get()
            colour_metric = self._current_colour_metric()
            group_by_material = self._thickness_fit_group_by_material()
            fit_reflectance_scale = bool(self.thickness_fit_scale_enabled_var.get())
            reflectance_scale_min = float(self.thickness_fit_scale_min_var.get())
            reflectance_scale_max = float(self.thickness_fit_scale_max_var.get())
            total_trials = self._thickness_optimization_trial_count(
                sample,
                range_percent,
                step_percent,
                group_by_material=group_by_material,
            )

            def task(progress):
                def trial_progress(done: int, total: int) -> None:
                    self._wait_if_paused(progress)
                    progress(done, f"{sample_name}: trial {done:,}/{total:,}")
                self._wait_if_paused(progress)
                return optimize_experiment_thicknesses(
                    store=store,
                    sample_name=sample_name,
                    measurement_index=measurement_index,
                    materials=materials,
                    model=model,
                    wavelengths_nm=wavelengths_nm,
                    angle_deg=angle_deg,
                    substrate_name=substrate_name,
                    native_oxide=native_oxide,
                    use_effective_interfaces=use_effective_interfaces,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                    range_percent=range_percent,
                    step_percent=step_percent,
                    cache_dir=cache_dir,
                    profile_name=profile_name,
                    model_label=model_label,
                    model_settings=model_settings,
                    group_by_material=group_by_material,
                    fixed_metal_threshold_nm=50.0,
                    colour_metric=colour_metric,
                    fit_reflectance_scale=fit_reflectance_scale,
                    reflectance_scale_min=reflectance_scale_min,
                    reflectance_scale_max=reflectance_scale_max,
                    progress_callback=trial_progress,
                )

            def on_success(result: ThicknessOptimizationResult) -> str:
                self.last_thickness_optimization_result = result
                self.plots_after_points_cache = None
                self._draw_thickness_optimization_result(result)
                layer_text = "; ".join(
                    f"{layer.material_name}: {layer.base_thickness_nm:.1f} -> "
                    f"{layer.optimized_thickness_nm:.1f} nm ({layer.percent_change:+.1f}%)"
                    for layer in result.layer_results
                )
                delta_label = "Delta E00" if normalise_colour_metric(result.colour_metric) == COLOUR_METRIC_CIEDE2000 else "Delta E*"
                self.experiment_info_var.set(
                    f"Optimized {delta_label} {result.base_delta_e:.2f} -> "
                    f"{result.optimized_delta_e:.2f}; "
                    f"reflectance scale {result.reflectance_scale:.3f}; "
                    f"new trials {result.new_count}, reused {result.reused_count}."
                )
                return layer_text

            self._run_background(
                task,
                on_success,
                title="Thickness optimization",
                busy_message="optimizing selected thicknesses",
                progress_max=total_trials,
            )
        except Exception as exc:
            self._stop_busy("Thickness optimization failed.")
            messagebox.showerror("Thickness optimization", str(exc))

    def resimulate_optimized_stack_with_current_roughness(self) -> None:
        result = self.last_thickness_optimization_result
        if result is None:
            messagebox.showinfo(
                "Roughness re-simulation",
                "Run a thickness optimization first. Then adjust interface/RMS settings and press this button.",
            )
            return

        if self.experiment_store is None:
            messagebox.showerror("Roughness re-simulation", "Load experiment data first.")
            return

        try:
            sample = self.experiment_store.load_sample(result.sample_name)
            measurement_index = next(
                (
                    index
                    for index, measurement in enumerate(sample.measurements)
                    if measurement.description == result.measurement_description
                ),
                None,
            )
            if measurement_index is None:
                raise ValueError("Could not find the optimized measurement in the raw experiment data.")
            measurement = sample.measurements[measurement_index]
            substrate_name = measurement.substrate_hint or self.substrate_var.get()
            substrate = self.materials[substrate_name]
            native_oxide = self._native_oxide_from_controls(substrate_name)
            optimized_layers = [
                Layer(self.materials[layer.material_name], layer.optimized_thickness_nm)
                for layer in result.layer_results
            ]
            if self._use_effective_interfaces():
                stack = make_stack_with_interfaces(
                    incident_medium=self.materials["air"],
                    deposited_layers=optimized_layers,
                    substrate=substrate,
                    native_oxide=native_oxide,
                    interface_thickness_nm=self.roughness_thickness_var.get(),
                    interface_fraction=self.roughness_fraction_var.get(),
                    name=f"{result.sample_name} optimized stack retuned roughness",
                )
            else:
                optical_layers = list(optimized_layers)
                if native_oxide is not None:
                    optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
                stack = make_stack(
                    incident_medium=self.materials["air"],
                    substrate=substrate,
                    layers=optical_layers,
                    name=f"{result.sample_name} optimized stack retuned roughness",
                    display_layers=optimized_layers,
                )

            model = self._model_from_controls()
            simulated = model.simulate(stack, result.wavelengths_nm, self.angle_var.get())
            retuned_color = perceived_color_from_result(simulated)
            retuned_delta = delta_e_colour(result.measured_color.xyz, retuned_color.xyz, metric=result.colour_metric)
            retuned = ThicknessOptimizationResult(
                sample_name=result.sample_name,
                measurement_description=result.measurement_description,
                stack_label=stack.display_summary(),
                wavelengths_nm=result.wavelengths_nm,
                measured_reflectance=result.measured_reflectance,
                base_reflectance=result.optimized_reflectance,
                optimized_reflectance=simulated.reflectance,
                measured_color=result.measured_color,
                base_color=result.optimized_color,
                optimized_color=retuned_color,
                base_delta_e=result.optimized_delta_e,
                optimized_delta_e=float(retuned_delta),
                layer_results=result.layer_results,
                cache_path=result.cache_path,
                evaluated_count=1,
                reused_count=1,
                new_count=0,
                colour_metric=result.colour_metric,
            )
            self._draw_roughness_retune_result(result, retuned)
            self.experiment_info_var.set(
                f"Re-simulated optimized thickness stack: {colour_metric_label(result.colour_metric)} {result.optimized_delta_e:.2f} -> "
                f"{retuned_delta:.2f} with current interface/RMS settings."
            )
        except Exception as exc:
            messagebox.showerror("Roughness re-simulation", str(exc))

    def _build_retuned_optimized_result(
        self,
        result: ThicknessOptimizationResult,
    ) -> ThicknessOptimizationResult:
        if self.experiment_store is None:
            raise ValueError("Load experiment data first.")
        sample = self.experiment_store.load_sample(result.sample_name)
        measurement_index = next(
            (
                index
                for index, measurement in enumerate(sample.measurements)
                if measurement.description == result.measurement_description
            ),
            None,
        )
        if measurement_index is None:
            raise ValueError("Could not find the optimized measurement in the raw experiment data.")
        measurement = sample.measurements[measurement_index]
        substrate_name = measurement.substrate_hint or self.substrate_var.get()
        substrate = self.materials[substrate_name]
        native_oxide = self._native_oxide_from_controls(substrate_name)
        optimized_layers = [
            Layer(self.materials[layer.material_name], layer.optimized_thickness_nm)
            for layer in result.layer_results
        ]
        if self._use_effective_interfaces():
            stack = make_stack_with_interfaces(
                incident_medium=self.materials["air"],
                deposited_layers=optimized_layers,
                substrate=substrate,
                native_oxide=native_oxide,
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
                name=f"{result.sample_name} optimized stack retuned roughness",
            )
        else:
            optical_layers = list(optimized_layers)
            if native_oxide is not None:
                optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
            stack = make_stack(
                incident_medium=self.materials["air"],
                substrate=substrate,
                layers=optical_layers,
                name=f"{result.sample_name} optimized stack retuned roughness",
                display_layers=optimized_layers,
            )

        simulated = self._model_from_controls().simulate(stack, result.wavelengths_nm, self.angle_var.get())
        retuned_color = perceived_color_from_result(simulated)
        retuned_delta = delta_e_colour(result.measured_color.xyz, retuned_color.xyz, metric=result.colour_metric)
        return ThicknessOptimizationResult(
            sample_name=result.sample_name,
            measurement_description=result.measurement_description,
            stack_label=stack.display_summary(),
            wavelengths_nm=result.wavelengths_nm,
            measured_reflectance=result.measured_reflectance,
            base_reflectance=result.optimized_reflectance,
            optimized_reflectance=simulated.reflectance,
            measured_color=result.measured_color,
            base_color=result.optimized_color,
            optimized_color=retuned_color,
            base_delta_e=result.optimized_delta_e,
            optimized_delta_e=float(retuned_delta),
            layer_results=result.layer_results,
            cache_path=result.cache_path,
            evaluated_count=1,
            reused_count=1,
            new_count=0,
            colour_metric=result.colour_metric,
        )

    def _schedule_experiment_retune_update(self) -> None:
        if self.last_thickness_optimization_result is None:
            return
        if not hasattr(self, "notebook") or self.notebook.select() != str(self.experiments_tab):
            return
        if self.experiment_retune_job is not None:
            self.root.after_cancel(self.experiment_retune_job)
        self.experiment_retune_job = self.root.after(300, self._auto_resimulate_optimized_stack)

    def _auto_resimulate_optimized_stack(self) -> None:
        self.experiment_retune_job = None
        result = self.last_thickness_optimization_result
        if result is None or self.experiment_store is None:
            return
        try:
            retuned = self._build_retuned_optimized_result(result)
            self._draw_roughness_retune_result(result, retuned)
            delta_label = "Delta E00" if normalise_colour_metric(result.colour_metric) == COLOUR_METRIC_CIEDE2000 else "Delta E*"
            self.experiment_info_var.set(
                f"Auto-updated current roughness: {delta_label} {result.optimized_delta_e:.2f} -> "
                f"{retuned.optimized_delta_e:.2f}."
            )
        except Exception as exc:
            self.experiment_info_var.set(f"Could not auto-update current roughness: {exc}")

    def fit_selected_roughness_group(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Roughness fit", "Load experiment data first.")
            return

        substrate_choice = self.experiment_substrate_filter_var.get()
        surface_choice = self.experiment_surface_filter_var.get()
        kind_choice = self.experiment_kind_filter_var.get()
        substrate_filter = None if substrate_choice == "All" else substrate_choice
        surface_filter = None if surface_choice == "All" else surface_choice
        kind_filter = None if kind_choice == "All" else kind_choice
        group_label = self._candidate_fit_group_label(substrate_choice, surface_choice, kind_choice)
        if surface_filter != "rough":
            answer = messagebox.askyesno(
                "Roughness fit",
                "The selected surface filter is not 'rough'. Continue fitting roughness parameters anyway?",
            )
            if not answer:
                return

        project_root = Path(__file__).resolve().parent
        store = self.experiment_store
        materials = self.materials.copy()
        wavelengths_nm = wavelength_grid(400.0, 700.0, 61)
        angle_deg = self.angle_var.get()
        substrate_default = self.substrate_var.get()
        use_effective_interfaces = self._use_effective_interfaces()
        interface_thickness_nm = self.roughness_thickness_var.get()
        interface_fraction = self.roughness_fraction_var.get()
        max_scatter_fraction = self.scatter_max_var.get()
        initial_rms = self.rms_roughness_var.get()
        colour_metric = self._current_colour_metric()

        def native_oxide_for_name(name: str) -> NativeOxide | None:
            return self._native_oxide_from_controls(name)

        def task(progress):
            self._wait_if_paused(progress)
            return fit_roughness_redistribution_parameters(
                project_root=project_root,
                store=store,
                materials=materials,
                wavelengths_nm=wavelengths_nm,
                angle_deg=angle_deg,
                substrate_default=substrate_default,
                native_oxide_factory=native_oxide_for_name,
                use_effective_interfaces=use_effective_interfaces,
                interface_thickness_nm=interface_thickness_nm,
                interface_fraction=interface_fraction,
                group_label=group_label,
                substrate_filter=substrate_filter,
                surface_filter=surface_filter,
                measurement_kind_filter=kind_filter,
                max_scatter_fraction=max_scatter_fraction,
                initial_rms_roughness_nm=initial_rms,
                fit_rms=True,
                colour_metric=colour_metric,
                progress_callback=lambda done, message: (self._wait_if_paused(progress), progress(done, message)),
            )

        def on_success(result) -> str:
            self.rms_roughness_var.set(result.rms_roughness_nm)
            self.scatter_scale_var.set(result.scatter_scale)
            self.scatter_exponent_var.set(result.scatter_exponent)
            self.scatter_max_var.set(result.max_scatter_fraction)
            if "diffuse redistribution" not in self.model_mode_var.get().lower():
                self.model_mode_var.set(
                    "Effective interface + diffuse redistribution"
                    if self._use_effective_interfaces()
                    else "Diffuse redistribution TMM"
                )
                self._on_model_mode_changed()
            self.schedule_reflectance_update()
            messagebox.showinfo(
                "Roughness fit",
                (
                    f"Saved roughness fit for group {result.group_label}:\n"
                    f"{result.output_path}\n\n"
                    f"RMS = {result.rms_roughness_nm:.3g} nm\n"
                    f"Scatter scale = {result.scatter_scale:.3g}\n"
                    f"Scatter exponent = {result.scatter_exponent:.3g}\n"
                    f"Mean {colour_metric_label(result.colour_metric)} = {result.mean_delta_e:.2f}\n"
                    f"Mean RMSE = {result.mean_rmse:.4f}"
                ),
            )
            return (
                f"Roughness fit {result.group_label}: RMS {result.rms_roughness_nm:.3g} nm, "
                f"scale {result.scatter_scale:.3g}, exponent {result.scatter_exponent:.3g}, "
                f"mean {colour_metric_label(result.colour_metric)} {result.mean_delta_e:.2f}."
            )

        self._run_background(
            task,
            on_success,
            title="Roughness fit",
            busy_message=f"fitting roughness group {group_label}",
        )

    def precalculate_all_thickness_optimizations(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Thickness optimization", "Load experiment data first.")
            return

        try:
            measurement_pairs = self._selected_fit_measurement_pairs()
            sample_names = list(dict.fromkeys(sample_name for sample_name, _index in measurement_pairs))
            measurement_counts: dict[str, int] = {}
            for sample_name, _measurement_index in measurement_pairs:
                measurement_counts[sample_name] = measurement_counts.get(sample_name, 0) + 1
            total_measurements = len(measurement_pairs)
            if total_measurements == 0:
                raise ValueError("No filtered experiment measurements with thickness estimates were found.")
            settings = self._precalculate_settings_dialog(
                sample_names,
                total_measurements,
                measurement_counts=measurement_counts,
            )
            if settings is None:
                return
            range_percent, step_percent, total_trials = settings
            group_by_material = self._thickness_fit_group_by_material()
            total_trials = sum(
                self._thickness_optimization_trial_count(
                    self.experiment_store.load_sample(sample_name),
                    range_percent,
                    step_percent,
                    group_by_material=group_by_material,
                )
                for sample_name, _measurement_index in measurement_pairs
            )

            cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            store = self.experiment_store
            materials = self.materials.copy()
            model = self._model_from_controls()
            angle_deg = self.angle_var.get()
            substrate_default = self.substrate_var.get()
            native_oxide_enabled = self.native_oxide_enabled_var.get()
            native_oxide_thickness_nm = self.native_oxide_thickness_var.get()
            use_effective_interfaces = self._use_effective_interfaces()
            interface_thickness_nm = self.roughness_thickness_var.get()
            interface_fraction = self.roughness_fraction_var.get()
            model_settings = self._model_settings_signature()
            profile_name = self.material_profile_var.get()
            model_label = self.model_mode_var.get()
            colour_metric = self._current_colour_metric()
            fit_reflectance_scale = bool(self.thickness_fit_scale_enabled_var.get())
            reflectance_scale_min = float(self.thickness_fit_scale_min_var.get())
            reflectance_scale_max = float(self.thickness_fit_scale_max_var.get())

            def native_oxide_for_name(name: str) -> NativeOxide | None:
                if not native_oxide_enabled:
                    return None
                default_oxide = native_oxide_for_substrate(materials, name)
                if default_oxide is None:
                    return None
                return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

            def task(progress):
                completed = 0
                completed_trials = 0
                skipped = 0
                total_new = 0
                total_reused = 0
                results: list[ThicknessOptimizationResult] = []
                for sample_name, measurement_index in measurement_pairs:
                    self._wait_if_paused(progress)
                    sample = store.load_sample(sample_name)
                    measurement = sample.measurements[measurement_index]
                    substrate_name = measurement.substrate_hint or substrate_default
                    progress(
                        completed_trials,
                        f"{sample_name} ({completed + 1}/{total_measurements})",
                    )
                    sample_trial_count = self._thickness_optimization_trial_count(
                        sample,
                        range_percent,
                        step_percent,
                        group_by_material=group_by_material,
                    )

                    def trial_progress(done: int, _total: int) -> None:
                        self._wait_if_paused(progress)
                        progress(
                            completed_trials + done,
                            f"{sample_name}: trial {done:,}/{sample_trial_count:,}; "
                            f"measurement {completed + 1}/{total_measurements}",
                        )

                    result = optimize_experiment_thicknesses(
                        store=store,
                        sample_name=sample_name,
                        measurement_index=measurement_index,
                        materials=materials,
                        model=model,
                        wavelengths_nm=wavelengths_nm,
                        angle_deg=angle_deg,
                        substrate_name=substrate_name,
                        native_oxide=native_oxide_for_name(substrate_name),
                        use_effective_interfaces=use_effective_interfaces,
                        interface_thickness_nm=interface_thickness_nm,
                        interface_fraction=interface_fraction,
                        range_percent=range_percent,
                        step_percent=step_percent,
                        cache_dir=cache_dir,
                        profile_name=profile_name,
                        model_label=model_label,
                        model_settings=model_settings,
                        group_by_material=group_by_material,
                        fixed_metal_threshold_nm=50.0,
                        colour_metric=colour_metric,
                        fit_reflectance_scale=fit_reflectance_scale,
                        reflectance_scale_min=reflectance_scale_min,
                        reflectance_scale_max=reflectance_scale_max,
                        progress_callback=trial_progress,
                    )
                    results.append(result)
                    total_new += result.new_count
                    total_reused += result.reused_count
                    completed += 1
                    completed_trials += sample_trial_count
                    progress(
                        completed_trials,
                        f"cached {completed}/{total_measurements}; "
                        f"new {total_new}, reused {total_reused}",
                    )
                summary_paths = save_optimization_summary_outputs(
                    results,
                    Path(__file__).resolve().parent / "outputs" / "thickness_optimization_summary",
                )
                return {
                    "completed": completed,
                    "skipped": skipped,
                    "total_new": total_new,
                    "total_reused": total_reused,
                    "cache_dir": cache_dir,
                    "summary_paths": summary_paths,
                }

            def on_success(summary: dict) -> str:
                self.plots_after_points_cache = None
                self.experiment_info_var.set(
                    f"Finished feasible {range_percent:g}% / {step_percent:g}% thickness caches "
                    f"in {summary['cache_dir']}. "
                    f"Summary plots saved in {summary['summary_paths']['summary_csv'].parent}."
                )
                return (
                    f"Precalculated {summary['completed']} thickness optimizations. "
                    f"New trials {summary['total_new']}, reused {summary['total_reused']}, "
                    f"summary saved."
                )

            self._run_background(
                task,
                on_success,
                title="Thickness optimization",
                busy_message="precalculating all thickness fits",
                progress_max=total_trials,
            )
        except Exception as exc:
            self._stop_busy("Batch thickness optimization stopped.")
            messagebox.showerror("Thickness optimization", str(exc))

    def fit_selected_sputter_rate_groups_from_colour(self) -> None:
        selected_keys = self._selected_rate_group_keys()
        if not selected_keys:
            messagebox.showerror("Sputter-rate fit", "Select one or more rate groups first.")
            return
        self.fit_sputter_rates_from_colour(selected_group_keys=selected_keys)

    def fit_all_sputter_rate_groups_from_colour(self) -> None:
        self.fit_sputter_rates_from_colour(selected_group_keys=None)

    def fit_sputter_rates_from_colour(
        self,
        selected_group_keys: set[tuple[str, str, float | None, float | None]] | None = None,
    ) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Sputter-rate fit", "Load experiment data first.")
            return

        store = self.experiment_store
        materials = self.materials.copy()
        model = self._model_from_controls()
        wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
        angle_deg = self.angle_var.get()
        substrate_default = self.substrate_var.get()
        native_oxide_enabled = self.native_oxide_enabled_var.get()
        native_oxide_thickness_nm = self.native_oxide_thickness_var.get()
        use_effective_interfaces = self._use_effective_interfaces()
        interface_thickness_nm = self.roughness_thickness_var.get()
        interface_fraction = self.roughness_fraction_var.get()
        output_dir = Path(__file__).resolve().parent / "outputs" / "sputter_rate_colour_fit"
        range_percent = float(self.fit_rate_range_percent_var.get())
        num_points = int(self.fit_rate_points_var.get())
        if num_points % 2 == 0:
            num_points += 1
        colour_metric = self._current_colour_metric()

        def sample_filter(sample) -> bool:
            return self._sample_matches_fit_filters(sample.sample_name, sample)

        def native_oxide_for_name(name: str) -> NativeOxide | None:
            if not native_oxide_enabled:
                return None
            default_oxide = native_oxide_for_substrate(materials, name)
            if default_oxide is None:
                return None
            return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

        def task(progress):
            return fit_sputter_rates_from_colour(
                store=store,
                materials=materials,
                model=model,
                wavelengths_nm=wavelengths_nm,
                angle_deg=angle_deg,
                substrate_default=substrate_default,
                native_oxide_factory=native_oxide_for_name,
                use_effective_interfaces=use_effective_interfaces,
                interface_thickness_nm=interface_thickness_nm,
                interface_fraction=interface_fraction,
                range_percent=range_percent,
                num_points=num_points,
                selected_group_keys=selected_group_keys,
                sample_filter=sample_filter,
                colour_metric=colour_metric,
                progress_callback=lambda done, total, label: progress(
                    int(100 * done / max(total, 1)),
                    f"{label} ({done}/{total})",
                ),
            )

        def on_success(results) -> str:
            if not results:
                raise ValueError("No grouped sputter-rate fits could be calculated.")
            paths = save_sputter_rate_fit_outputs(results, output_dir)
            self._draw_sputter_rate_fit_summary(results, paths)
            self._select_fit_optimize_section("Rate groups")
            best_improvement = max(
                result.mean_delta_e_before - result.mean_delta_e_after for result in results
            )
            self.experiment_info_var.set(
                f"Saved colour-fitted sputter rates to {paths['csv']}. "
                f"Best mean {self._delta_e_label()} improvement: {best_improvement:.2f}."
            )
            scope = "selected" if selected_group_keys else "all"
            return f"Fitted {len(results)} {scope} grouped sputter rates from colour."

        self._run_background(
            task,
            on_success,
            title="Sputter-rate fit",
            busy_message="fitting grouped sputter rates from colour",
            progress_max=100,
        )

    def run_physical_calibration(self) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("Calibration", "Load experiment data first.")
            return
        if self.background_task_running:
            messagebox.showinfo("Calibration", "A calculation is already running.")
            return

        try:
            group = self.calibration_group_var.get()
            rate_range = float(self.calibration_rate_range_var.get())
            rate_points = int(self.calibration_rate_points_var.get())
            if rate_points < 3:
                raise ValueError("Use at least 3 rate points.")
            if rate_points % 2 == 0:
                rate_points += 1

            profiles: list[tuple[str, dict[str, Material]]] = []
            for profile in self._material_profile_choices():
                try:
                    profiles.append((profile, self._materials_for_profile(profile)))
                except Exception:
                    continue
            if not profiles:
                raise ValueError("No constants profiles could be loaded.")

            model_labels = self._optical_model_labels()
            total_candidates = max(len(profiles) * len(model_labels), 1)
            total_steps = total_candidates * 100
            store = self.experiment_store
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            angle_deg = float(self.angle_var.get())
            substrate_default = self.substrate_var.get()
            native_oxide_enabled = bool(self.native_oxide_enabled_var.get())
            native_oxide_thickness_nm = float(self.native_oxide_thickness_var.get())
            interface_thickness_nm = float(self.roughness_thickness_var.get())
            interface_fraction = float(self.roughness_fraction_var.get())
            roughness_settings = {
                "rms_roughness_nm": float(self.rms_roughness_var.get()),
                "scatter_scale": float(self.scatter_scale_var.get()),
                "scatter_exponent": float(self.scatter_exponent_var.get()),
                "max_scatter_fraction": float(self.scatter_max_var.get()),
            }
            colour_metric = self._current_colour_metric()
            sample_filter = self._calibration_sample_filter(group)
            output_dir = (
                Path(__file__).resolve().parent
                / "outputs"
                / "physical_calibration"
                / self._safe_cache_prefix(group)
            )

            def task(progress):
                rows: list[dict[str, object]] = []
                candidate_index = 0
                for profile_name, materials in profiles:
                    for model_label in model_labels:
                        self._wait_if_paused(progress)
                        candidate_index += 1
                        candidate_base = (candidate_index - 1) * 100
                        progress(
                            candidate_base,
                            f"{group}: starting {candidate_index}/{total_candidates}: "
                            f"{profile_name} / {model_label}",
                        )
                        model = self._model_for_label(model_label, roughness_settings)
                        use_effective = self._use_effective_interfaces_for_label(model_label)

                        def native_oxide_for_name(name: str) -> NativeOxide | None:
                            if not native_oxide_enabled:
                                return None
                            default_oxide = native_oxide_for_substrate(materials, name)
                            if default_oxide is None:
                                return None
                            return NativeOxide(default_oxide.material, native_oxide_thickness_nm)

                        rate_results = fit_sputter_rates_from_colour(
                            store=store,
                            materials=materials,
                            model=model,
                            wavelengths_nm=wavelengths_nm,
                            angle_deg=angle_deg,
                            substrate_default=substrate_default,
                            native_oxide_factory=native_oxide_for_name,
                            use_effective_interfaces=use_effective,
                            interface_thickness_nm=interface_thickness_nm,
                            interface_fraction=interface_fraction,
                            range_percent=rate_range,
                            num_points=rate_points,
                            sample_filter=sample_filter,
                            colour_metric=colour_metric,
                            progress_callback=lambda done, total, label, base=candidate_base, profile=profile_name, model_name=model_label: (
                                self._wait_if_paused(progress),
                                progress(
                                    min(base + int(100 * done / max(total, 1)), total_steps),
                                    f"{group}: {profile} / {model_name}: {label} ({done}/{total})",
                                ),
                            ),
                        )
                        if not rate_results:
                            continue
                        weights = np.asarray([result.measurement_count for result in rate_results], dtype=float)
                        if not np.any(weights > 0):
                            continue
                        before = float(np.average(
                            [result.mean_delta_e_before for result in rate_results],
                            weights=weights,
                        ))
                        after = float(np.average(
                            [result.mean_delta_e_after for result in rate_results],
                            weights=weights,
                        ))
                        max_rate_change = float(max(abs(result.percent_change) for result in rate_results))
                        rows.append(
                            {
                                "group": group,
                                "constants_profile": profile_name,
                                "optical_model": model_label,
                                "mean_delta_e_before": before,
                                "mean_delta_e_after": after,
                                "improvement": before - after,
                                "max_abs_rate_change_percent": max_rate_change,
                                "rate_group_count": len(rate_results),
                                "measurement_count": int(np.sum(weights)),
                                "rate_results": rate_results,
                            }
                        )
                if not rows:
                    raise ValueError(f"No calibration results were found for {group}.")
                rows.sort(key=lambda row: float(row["mean_delta_e_after"]))
                output_dir.mkdir(parents=True, exist_ok=True)
                table_rows = [
                    {key: value for key, value in row.items() if key != "rate_results"}
                    for row in rows
                ]
                summary_path = output_dir / "physical_calibration_summary.csv"
                pd.DataFrame(table_rows).to_csv(summary_path, index=False)
                return {"rows": rows, "summary_path": summary_path}

            def on_success(payload) -> str:
                rows = payload["rows"]
                self._draw_physical_calibration_result(rows, payload["summary_path"])
                self._select_fit_optimize_section("Model calibration")
                best = rows[0]
                return (
                    "Calibration complete: "
                    f"{best['constants_profile']} / {best['optical_model']} "
                    f"{self._delta_e_label()} {float(best['mean_delta_e_after']):.2f}."
                )

            self._run_background(
                task,
                on_success,
                title="Calibration",
                busy_message="calibrating constants, model, and sputter rates",
                progress_max=total_steps,
            )
        except Exception as exc:
            messagebox.showerror("Calibration", str(exc))

    def _calibration_sample_filter(self, group: str):
        group_normalized = str(group or "All").strip().lower()

        def sample_matches(sample) -> bool:
            if group_normalized == "all":
                return True
            for measurement in sample.measurements:
                substrate = measurement.substrate_group or measurement.substrate_hint or ""
                surface = measurement.surface_class or ""
                if group_normalized == "smooth si" and substrate == "Si" and surface == "smooth":
                    return True
                if (
                    group_normalized == "smooth si double polished"
                    and substrate == "Si double polished"
                    and surface == "smooth"
                ):
                    return True
                if group_normalized == "rough si" and substrate == "Si" and surface == "rough":
                    return True
                if (
                    group_normalized == "rough si double polished"
                    and substrate == "Si double polished"
                    and surface == "rough"
                ):
                    return True
                if group_normalized == "rough ti" and substrate == "Ti" and surface == "rough":
                    return True
                if group_normalized == "smooth" and surface == "smooth":
                    return True
                if group_normalized == "rough" and surface == "rough":
                    return True
            return False

        return sample_matches

    def _draw_physical_calibration_result(self, rows: list[dict[str, object]], summary_path: Path) -> None:
        for item in self.calibration_tree.get_children():
            self.calibration_tree.delete(item)
        for index, row in enumerate(rows[:30], start=1):
            self.calibration_tree.insert(
                "",
                tk.END,
                values=(
                    index,
                    row["group"],
                    row["constants_profile"],
                    row["optical_model"],
                    f"{float(row['mean_delta_e_before']):.2f}",
                    f"{float(row['mean_delta_e_after']):.2f}",
                    f"{float(row['improvement']):+.2f}",
                    f"{float(row['max_abs_rate_change_percent']):.2f}%",
                    row["rate_group_count"],
                    row["measurement_count"],
                ),
            )

        self.calibration_figure.clear()
        grid = self.calibration_figure.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
        score_ax = self.calibration_figure.add_subplot(grid[0, 0])
        rate_ax = self.calibration_figure.add_subplot(grid[0, 1])

        top = rows[:12]
        labels = [
            f"{row['constants_profile']}\n{row['optical_model']}"
            for row in top
        ]
        x = np.arange(len(top))
        score_ax.plot(
            x,
            [float(row["mean_delta_e_before"]) for row in top],
            marker="o",
            label="Before rate fit",
            color="#8aa3b8",
        )
        score_ax.plot(
            x,
            [float(row["mean_delta_e_after"]) for row in top],
            marker="o",
            label="After rate fit",
            color="#2f6f9f",
        )
        score_ax.set_xticks(x)
        score_ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=6.5)
        score_ax.set_ylabel(r"Weighted mean $\Delta E^*_{Lab}$")
        score_ax.set_title("Best constants/models", fontsize=9, fontweight="semibold", pad=4)
        score_ax.tick_params(axis="y", labelsize=7)
        score_ax.grid(True, alpha=0.25)
        score_ax.legend(fontsize=7)

        best_rate_results = rows[0]["rate_results"]
        rate_labels = [result.group_label for result in best_rate_results]
        y = np.arange(len(best_rate_results))
        rate_ax.barh(
            y,
            [result.percent_change for result in best_rate_results],
            color="#2f6f9f",
        )
        rate_ax.axvline(0.0, color="#111827", linewidth=0.8)
        rate_ax.set_yticks(y)
        rate_ax.set_yticklabels(rate_labels, fontsize=7)
        rate_ax.invert_yaxis()
        rate_ax.set_xlabel("Fitted sputter-rate change (%)")
        rate_ax.set_title("Shared rate corrections", fontsize=9, fontweight="semibold", pad=4)
        rate_ax.tick_params(axis="x", labelsize=7)
        rate_ax.grid(True, axis="x", alpha=0.25)

        self.calibration_figure.suptitle(
            f"Physical calibration saved: {summary_path}",
            fontsize=8,
            color="#52606d",
        )
        self.calibration_figure.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.88, wspace=0.36)
        self.calibration_canvas.draw_idle()

    def apply_selected_calibration_model(self) -> None:
        if not hasattr(self, "calibration_tree"):
            return
        selection = self.calibration_tree.selection()
        if not selection:
            children = self.calibration_tree.get_children()
            if not children:
                messagebox.showinfo("Model calibration", "Run calibration first, then select a row.")
                return
            selection = (children[0],)
        values = self.calibration_tree.item(selection[0], "values")
        if len(values) < 4:
            return
        profile = str(values[2])
        model_label = str(values[3])
        if profile:
            self.material_profile_var.set(profile)
            self._on_material_profile_changed()
        if model_label:
            self.model_mode_var.set(model_label)
            self._update_roughness_control_states()
        self.schedule_reflectance_update()
        self.status_var.set(
            f"Using calibrated model for search and fitting: {profile} / {model_label}"
        )

    def _precalculate_settings_dialog(
        self,
        sample_names: list[str],
        total_measurements: int,
        measurement_counts: dict[str, int] | None = None,
    ) -> tuple[float, float, int] | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Precalculate thickness fits")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        range_var = tk.DoubleVar(value=float(self.thickness_opt_range_percent_var.get()))
        step_var = tk.DoubleVar(value=float(self.thickness_opt_step_percent_var.get()))
        group_by_material = self._thickness_fit_group_by_material()
        mode_label = self.thickness_fit_mode_var.get()
        estimate_var = tk.StringVar()
        result: dict[str, tuple[float, float, int] | None] = {"value": None}

        wrapper = ttk.Frame(dialog, padding=14)
        wrapper.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            wrapper,
            text="Precalculate cached thickness optimizations",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=tk.W)
        ttk.Label(
            wrapper,
            text=(
                "Choose the sputter-rate error range and step. Smaller steps grow very fast "
                "for multilayer samples, but completed trials are reused from cache."
            ),
            foreground="#52606d",
            wraplength=520,
        ).pack(anchor=tk.W, pady=(4, 12))

        controls = ttk.Frame(wrapper)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Rate error +/- (%)").grid(row=0, column=0, sticky=tk.W, pady=3)
        range_spin = self._spinbox(controls, range_var, 0.0, 25.0, 0.5, None)
        range_spin.grid(row=0, column=1, sticky=tk.EW, padx=(12, 0), pady=3)
        ttk.Label(controls, text="Step (%)").grid(row=1, column=0, sticky=tk.W, pady=3)
        step_spin = self._spinbox(controls, step_var, 0.1, 10.0, 0.1, None)
        step_spin.grid(row=1, column=1, sticky=tk.EW, padx=(12, 0), pady=3)
        controls.columnconfigure(1, weight=1)

        estimate_label = ttk.Label(
            wrapper,
            textvariable=estimate_var,
            foreground="#1f2933",
            justify=tk.LEFT,
            wraplength=520,
        )
        estimate_label.pack(anchor=tk.W, pady=(12, 8))

        def estimate() -> None:
            try:
                range_percent = max(float(range_var.get()), 0.0)
                step_percent = max(float(step_var.get()), 0.1)
                total_trials = sum(
                    self._thickness_optimization_trial_count(
                        self.experiment_store.load_sample(name),  # type: ignore[union-attr]
                        range_percent,
                        step_percent,
                        group_by_material=group_by_material,
                    )
                    * (
                        measurement_counts.get(name, 0)
                        if measurement_counts is not None
                        else len(self.experiment_store.load_sample(name).measurements)  # type: ignore[union-attr]
                    )
                    for name in sample_names
                )
                new_time = self._format_duration(total_trials * 0.035)
                cached_time = self._format_duration(total_trials * 0.003)
                estimate_var.set(
                    f"{len(sample_names)} samples, {total_measurements} measurements\n"
                    f"Thickness mode: {mode_label}\n"
                    f"Estimated trial evaluations: {total_trials:,}\n"
                    f"Very rough time estimate: {cached_time} if mostly cached, up to {new_time} if mostly new."
                )
            except Exception as exc:
                estimate_var.set(f"Could not estimate trial count: {exc}")

        for variable in (range_var, step_var):
            variable.trace_add("write", lambda *_args: estimate())
        estimate()

        button_row = ttk.Frame(wrapper)
        button_row.pack(fill=tk.X, pady=(8, 0))

        def start() -> None:
            try:
                range_percent = max(float(range_var.get()), 0.0)
                step_percent = max(float(step_var.get()), 0.1)
                total_trials = sum(
                    self._thickness_optimization_trial_count(
                        self.experiment_store.load_sample(name),  # type: ignore[union-attr]
                        range_percent,
                        step_percent,
                        group_by_material=group_by_material,
                    )
                    * (
                        measurement_counts.get(name, 0)
                        if measurement_counts is not None
                        else len(self.experiment_store.load_sample(name).measurements)  # type: ignore[union-attr]
                    )
                    for name in sample_names
                )
                self.thickness_opt_range_percent_var.set(range_percent)
                self.thickness_opt_step_percent_var.set(step_percent)
                self._save_gui_settings()
                result["value"] = (range_percent, step_percent, total_trials)
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("Precalculate all", str(exc), parent=dialog)

        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Start caching", command=start).pack(side=tk.RIGHT, padx=(0, 8))

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_width()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_height()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        self.root.wait_window(dialog)
        return result["value"]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return f"{seconds:.0f} s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f} min"
        hours = minutes / 60.0
        return f"{hours:.1f} h"

    def _selected_cache_or_control_measurement(self) -> tuple[str, int]:
        selection = self.experiment_results_tree.selection()
        if self.experiment_cache is not None and selection:
            index = int(selection[0])
            sample_name = str(self.experiment_cache.sample_names[index])
            description = str(self.experiment_cache.measurement_descriptions[index])
            sample = self.experiment_store.load_sample(sample_name)  # type: ignore[union-attr]
            for measurement_index, measurement in enumerate(sample.measurements):
                if measurement.description == description:
                    return sample_name, measurement_index
            raise ValueError("Could not match the selected cached measurement to the raw data.")
        return self.experiment_sample_var.get(), self._selected_experiment_measurement_index()

    def _thickness_fit_group_by_material(self) -> bool:
        return self.thickness_fit_mode_var.get() == "Same material together"

    @staticmethod
    def _thickness_optimization_trial_count(
        sample,
        range_percent: float,
        step_percent: float,
        group_by_material: bool = False,
    ) -> int:
        optimizable_layers = [
            layer
            for layer in sample.layer_estimates
            if not (layer.material_name in {"Ag", "Au"} and layer.thickness_nm >= 50.0)
        ]
        if group_by_material:
            variable_count = len({layer.material_name for layer in optimizable_layers})
        else:
            variable_count = len(optimizable_layers)
        if variable_count == 0:
            return 1
        point_count = int(np.floor((2.0 * range_percent) / step_percent + 0.5)) + 1
        return int(point_count ** variable_count)

    def _selected_experiment_measurement_index(self) -> int:
        label = self.experiment_measurement_var.get()
        if not label:
            raise ValueError("Select a measurement.")
        return int(label.split(":", maxsplit=1)[0]) - 1

    def _experiment_plot_text_sizes(self) -> dict[str, float]:
        try:
            scale = float(self.experiment_plot_text_scale_var.get())
        except (tk.TclError, ValueError):
            scale = 0.72
        scale = max(0.45, min(1.05, scale))
        return {
            "title": 9.2 * scale,
            "label": 8.4 * scale,
            "tick": 7.8 * scale,
            "legend": 7.4 * scale,
            "swatch_title": 9.0 * scale,
            "swatch_text": 8.6 * scale,
        }

    def _style_experiment_cie_legend(self, ax) -> None:
        sizes = self._experiment_plot_text_sizes()
        ax.legend(
            loc="upper right",
            fontsize=sizes["legend"],
            markerscale=0.45,
            handlelength=1.15,
            handletextpad=0.35,
            borderpad=0.25,
            labelspacing=0.22,
            borderaxespad=0.25,
            framealpha=0.82,
        )

    def _download_figure(self, figure: Figure, label: str) -> None:
        try:
            output_dir = Path(__file__).resolve().parent / "outputs" / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower()).strip("_")
            if not safe_label:
                safe_label = "figure"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"{safe_label}_{timestamp}.png"
            figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            self.status_var.set(f"Saved figure to {path}")
        except Exception as exc:
            messagebox.showerror("Download figure", str(exc))

    def _download_primary_axis_figure(self, figure: Figure, label: str) -> None:
        try:
            if not figure.axes:
                raise ValueError("No plot is available to save.")
            axis = next((ax for ax in figure.axes if ax.get_visible()), figure.axes[0])
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            bbox = axis.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted())
            bbox = bbox.padded(0.08)

            output_dir = Path(__file__).resolve().parent / "outputs" / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower()).strip("_")
            if not safe_label:
                safe_label = "figure"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"{safe_label}_{timestamp}.png"
            figure.savefig(
                path,
                dpi=300,
                bbox_inches=bbox,
                facecolor="white",
                edgecolor="none",
            )
            self.status_var.set(f"Saved clean figure to {path}")
        except Exception as exc:
            messagebox.showerror("Download figure", str(exc))

    def _pack_download_figure_button(
        self,
        parent: ttk.Frame,
        figure_getter,
        label: str,
        *,
        side: str = tk.RIGHT,
    ) -> None:
        ttk.Button(
            parent,
            text="Download figure",
            command=lambda: self._download_figure(figure_getter(), label),
        ).pack(side=side, padx=(6, 0), pady=4)

    def _wrapped_experiment_plot_title(
        self,
        *lines: object,
        width: int = 92,
        max_lines: int = 3,
    ) -> str:
        sizes = self._experiment_plot_text_sizes()
        wrap_width = max(42, int(width / max(0.65, sizes["title"] / 11.0)))
        wrapped: list[str] = []
        for line in lines:
            text = str(line).strip()
            if not text:
                continue
            wrapped.extend(
                textwrap.wrap(text, width=wrap_width, break_long_words=False)
                or [text]
            )
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            wrapped[-1] = textwrap.shorten(
                wrapped[-1],
                width=wrap_width,
                placeholder="...",
            )
        return "\n".join(wrapped)

    def _style_experiment_spectrum_axis(
        self,
        ax,
        title: str,
        legend_columns: int = 1,
    ) -> None:
        sizes = self._experiment_plot_text_sizes()
        ax.set_title(title, fontsize=sizes["title"], fontweight="semibold", pad=3)
        ax.xaxis.label.set_size(sizes["label"])
        ax.yaxis.label.set_size(sizes["label"])
        ax.tick_params(axis="both", labelsize=sizes["tick"])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                ncol=legend_columns,
                fontsize=sizes["legend"],
                framealpha=0.92,
            )

    def _draw_experiment_result(self, result) -> None:
        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(
            2,
            3,
            height_ratios=[0.9, 2.4],
            hspace=0.38,
            wspace=0.28,
        )
        measured_ax = self.experiment_figure.add_subplot(grid[0, 0])
        simulated_ax = self.experiment_figure.add_subplot(grid[0, 1])
        cie_ax = self.experiment_figure.add_subplot(grid[:, 2])
        spectrum_ax = self.experiment_figure.add_subplot(grid[1, :2])

        self._draw_color_swatch(
            measured_ax,
            "Measured colour",
            result.measured_color.srgb,
            result.measured_color.hex,
            result.measured_color.srgb_255,
        )
        self._draw_color_swatch(
            simulated_ax,
            "Estimated-stack TMM colour",
            result.simulated_color.srgb,
            result.simulated_color.hex,
            result.simulated_color.srgb_255,
        )

        mask = (
            (result.measured_wavelengths_nm >= 400.0)
            & (result.measured_wavelengths_nm <= 700.0)
        )
        spectrum_ax.plot(
            result.measured_wavelengths_nm[mask],
            result.measured_reflectance[mask],
            color="#0f766e",
            linewidth=2.0,
            label="Measured reflectance",
        )
        spectrum_ax.plot(
            result.simulated_result.wavelengths_nm,
            result.simulated_result.reflectance,
            color="#111827",
            linewidth=2.0,
            linestyle="--",
            label="Simulated from thickness estimate",
        )
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.set_xlim(400.0, 700.0)
        spectrum_ax.set_ylim(0.0, 1.0)
        spectrum_ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
        self._style_experiment_spectrum_axis(
            spectrum_ax,
            self._wrapped_experiment_plot_title(
                f"{result.sample_name}: {result.stack.display_summary()}",
                result.measurement.description,
            ),
        )
        self._draw_cie_panel(
            cie_ax,
            measured_xy=xyz_to_xy(result.measured_color.xyz),
            simulated_xy=xyz_to_xy(result.simulated_color.xyz),
            delta_e=self._colour_delta_e(result.measured_color.xyz, result.simulated_color.xyz),
        )
        self.experiment_figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.10,
            top=0.90,
            hspace=0.42,
            wspace=0.20,
        )
        self.experiment_canvas.draw_idle()

    def _draw_cached_experiment_result(self, index: int) -> None:
        if self.experiment_cache is None:
            return
        cache = self.experiment_cache
        sample_name = str(cache.sample_names[index])
        selected_substrate = str(cache.substrate_classes[index])
        selected_surface = str(cache.surface_classes[index])
        selected_kind = str(cache.measurement_kinds[index])
        sample_indices = np.flatnonzero(
            (cache.sample_names == sample_name)
            & (cache.substrate_classes == selected_substrate)
            & (cache.surface_classes == selected_surface)
            & (cache.measurement_kinds == selected_kind)
        )
        if sample_indices.size == 0:
            sample_indices = np.asarray([index], dtype=int)
        primary = int(sample_indices[0])
        selected_position = int(np.where(sample_indices == index)[0][0])
        selected_measured_rgb = cache.measured_rgb[index]
        selected_simulated_rgb = cache.simulated_rgb[index]
        selected_delta_e = float(cache.delta_e[index])

        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(
            2,
            3,
            height_ratios=[0.9, 2.4],
            hspace=0.38,
            wspace=0.28,
        )
        measured_ax = self.experiment_figure.add_subplot(grid[0, 0])
        simulated_ax = self.experiment_figure.add_subplot(grid[0, 1])
        cie_ax = self.experiment_figure.add_subplot(grid[:, 2])
        spectrum_ax = self.experiment_figure.add_subplot(grid[1, :2])

        measured_hex = self._rgb_tuple_to_hex(selected_measured_rgb)
        simulated_hex = self._rgb_tuple_to_hex(selected_simulated_rgb)
        self._draw_color_swatch(
            measured_ax,
            "Selected measured colour",
            selected_measured_rgb,
            measured_hex,
            self._rgb_tuple_to_255(selected_measured_rgb),
        )
        self._draw_color_swatch(
            simulated_ax,
            "Selected TMM colour",
            selected_simulated_rgb,
            simulated_hex,
            self._rgb_tuple_to_255(selected_simulated_rgb),
        )
        measured_color = "#0f766e"
        simulated_color = "#111827"
        for plot_number, row_index in enumerate(sample_indices, start=1):
            is_selected = row_index == index
            alpha = 1.0 if is_selected else (0.28 if sample_indices.size > 1 else 1.0)
            linewidth = 2.9 if is_selected else 1.2
            label = "Selected measured reflectance" if is_selected else (
                "Other measured reflectance" if plot_number == 1 else None
            )
            spectrum_ax.plot(
                cache.wavelengths_nm,
                cache.measured_reflectance[row_index],
                color=measured_color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )
        spectrum_ax.plot(
            cache.wavelengths_nm,
            cache.simulated_reflectance[index],
            color=simulated_color,
            linewidth=2.1,
            linestyle="--",
            label="Simulated from thickness estimate",
        )
        if sample_indices.size > 1:
            spectrum_ax.plot(
                cache.wavelengths_nm,
                np.mean(cache.measured_reflectance[sample_indices], axis=0),
                color=measured_color,
                linewidth=1.8,
                alpha=0.75,
                label="Measured mean",
            )
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.set_xlim(float(cache.wavelengths_nm.min()), float(cache.wavelengths_nm.max()))
        spectrum_ax.set_ylim(0.0, 1.0)
        spectrum_ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
        self._style_experiment_spectrum_axis(
            spectrum_ax,
            self._wrapped_experiment_plot_title(
                f"{sample_name}: {cache.stack_labels[primary]}",
                (
                    f"Selected: {cache.measurement_descriptions[index]} "
                    f"[{selected_substrate}, {selected_surface}, {selected_kind}] "
                    f"({sample_indices.size} spectrum/spectra shown)"
                ),
            ),
        )
        self._draw_cie_panel(
            cie_ax,
            measured_xy=cache.measured_xy[sample_indices],
            simulated_xy=cache.simulated_xy[sample_indices],
            delta_e=selected_delta_e,
            selected_index=selected_position,
        )
        self.experiment_figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.10,
            top=0.90,
            hspace=0.42,
            wspace=0.28,
        )
        self.experiment_canvas.draw_idle()
        self.status_var.set(
            f"{sample_name}: selected {self._delta_e_label()} {selected_delta_e:.2f}, "
            f"{selected_substrate}, {selected_surface}, {selected_kind}; {sample_indices.size} spectra shown"
        )

    def _draw_thickness_optimization_result(
        self,
        result: ThicknessOptimizationResult,
    ) -> None:
        delta_label = "Delta E00" if normalise_colour_metric(result.colour_metric) == COLOUR_METRIC_CIEDE2000 else "Delta E*"
        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(
            2,
            4,
            height_ratios=[0.9, 2.4],
            hspace=0.40,
            wspace=0.28,
        )
        measured_ax = self.experiment_figure.add_subplot(grid[0, 0])
        base_ax = self.experiment_figure.add_subplot(grid[0, 1])
        optimized_ax = self.experiment_figure.add_subplot(grid[0, 2])
        cie_ax = self.experiment_figure.add_subplot(grid[:, 3])
        spectrum_ax = self.experiment_figure.add_subplot(grid[1, :3])

        self._draw_color_swatch(
            measured_ax,
            "Measured colour",
            result.measured_color.srgb,
            result.measured_color.hex,
            result.measured_color.srgb_255,
        )
        self._draw_color_swatch(
            base_ax,
            "Before fit",
            result.base_color.srgb,
            result.base_color.hex,
            result.base_color.srgb_255,
        )
        self._draw_color_swatch(
            optimized_ax,
            "After thickness fit",
            result.optimized_color.srgb,
            result.optimized_color.hex,
            result.optimized_color.srgb_255,
        )

        spectrum_ax.plot(
            result.wavelengths_nm,
            result.measured_reflectance,
            color="#0f766e",
            linewidth=2.6,
            label="Measured reflectance",
        )
        spectrum_ax.plot(
            result.wavelengths_nm,
            result.base_reflectance,
            color="#111827",
            linewidth=1.9,
            linestyle="--",
            label=f"Before fit ({delta_label} {result.base_delta_e:.2f})",
        )
        spectrum_ax.plot(
            result.wavelengths_nm,
            result.optimized_reflectance,
            color="#f97316",
            linewidth=2.2,
            label=(
                f"After fit ({delta_label} {result.optimized_delta_e:.2f}, "
                f"scale {result.reflectance_scale:.3f})"
            )
            if result.scale_fit_enabled
            else f"After fit ({delta_label} {result.optimized_delta_e:.2f})",
        )
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.set_xlim(float(result.wavelengths_nm.min()), float(result.wavelengths_nm.max()))
        spectrum_ax.set_ylim(0.0, 1.0)
        spectrum_ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
        layer_lines = [
            f"{layer.material_name}: {layer.optimized_thickness_nm:.1f} nm "
            f"({layer.percent_change:+.1f}%)"
            for layer in result.layer_results
        ]
        if result.scale_fit_enabled:
            layer_lines.append(f"reflectance scale {result.reflectance_scale:.3f}")
        self._style_experiment_spectrum_axis(
            spectrum_ax,
            self._wrapped_experiment_plot_title(
                f"{result.sample_name}: {result.measurement_description}",
                " | ".join(layer_lines),
            ),
        )

        self._draw_cie_panel(
            cie_ax,
            measured_xy=np.asarray([result.measured_xy, result.measured_xy], dtype=float),
            simulated_xy=np.asarray([result.base_xy, result.optimized_xy], dtype=float),
            delta_e=result.optimized_delta_e,
            selected_index=1,
        )
        cie_ax.scatter(
            [result.optimized_xy[0]],
            [result.optimized_xy[1]],
            s=105,
            c="#f97316",
            edgecolors="white",
            linewidths=1.2,
            marker="D",
            label="Optimized",
            zorder=5,
        )
        self._style_experiment_cie_legend(cie_ax)

        self.experiment_figure.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.10,
            top=0.90,
            hspace=0.42,
            wspace=0.28,
        )
        self.experiment_canvas.draw_idle()

    def _draw_roughness_retune_result(
        self,
        previous: ThicknessOptimizationResult,
        retuned: ThicknessOptimizationResult,
    ) -> None:
        delta_label = "Delta E00" if normalise_colour_metric(retuned.colour_metric) == COLOUR_METRIC_CIEDE2000 else "Delta E*"
        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(
            2,
            4,
            height_ratios=[0.9, 2.4],
            hspace=0.40,
            wspace=0.28,
        )
        measured_ax = self.experiment_figure.add_subplot(grid[0, 0])
        thickness_ax = self.experiment_figure.add_subplot(grid[0, 1])
        retuned_ax = self.experiment_figure.add_subplot(grid[0, 2])
        cie_ax = self.experiment_figure.add_subplot(grid[:, 3])
        spectrum_ax = self.experiment_figure.add_subplot(grid[1, :3])

        self._draw_color_swatch(
            measured_ax,
            "Measured colour",
            retuned.measured_color.srgb,
            retuned.measured_color.hex,
            retuned.measured_color.srgb_255,
        )
        self._draw_color_swatch(
            thickness_ax,
            "Best thickness stack",
            previous.optimized_color.srgb,
            previous.optimized_color.hex,
            previous.optimized_color.srgb_255,
        )
        self._draw_color_swatch(
            retuned_ax,
            "Current roughness",
            retuned.optimized_color.srgb,
            retuned.optimized_color.hex,
            retuned.optimized_color.srgb_255,
        )

        spectrum_ax.plot(
            retuned.wavelengths_nm,
            retuned.measured_reflectance,
            color="#0f766e",
            linewidth=2.6,
            label="Measured reflectance",
        )
        spectrum_ax.plot(
            previous.wavelengths_nm,
            previous.optimized_reflectance,
            color="#f97316",
            linewidth=2.0,
            label=f"Best thickness ({delta_label} {previous.optimized_delta_e:.2f})",
        )
        spectrum_ax.plot(
            retuned.wavelengths_nm,
            retuned.optimized_reflectance,
            color="#7c3aed",
            linewidth=2.3,
            label=f"Current roughness ({delta_label} {retuned.optimized_delta_e:.2f})",
        )
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.set_xlim(float(retuned.wavelengths_nm.min()), float(retuned.wavelengths_nm.max()))
        spectrum_ax.set_ylim(0.0, 1.0)
        spectrum_ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
        layer_lines = [
            f"{layer.material_name}: {layer.optimized_thickness_nm:.1f} nm "
            f"({layer.percent_change:+.1f}%)"
            for layer in retuned.layer_results
        ]
        self._style_experiment_spectrum_axis(
            spectrum_ax,
            self._wrapped_experiment_plot_title(
                f"{retuned.sample_name}: thickness fixed, roughness/interface re-simulated",
                " | ".join(layer_lines),
            ),
        )

        self._draw_cie_panel(
            cie_ax,
            measured_xy=np.asarray([retuned.measured_xy, retuned.measured_xy], dtype=float),
            simulated_xy=np.asarray([previous.optimized_xy, retuned.optimized_xy], dtype=float),
            delta_e=retuned.optimized_delta_e,
            selected_index=1,
        )
        cie_ax.scatter(
            [retuned.optimized_xy[0]],
            [retuned.optimized_xy[1]],
            s=105,
            c="#7c3aed",
            edgecolors="white",
            linewidths=1.2,
            marker="D",
            label="Current roughness",
            zorder=5,
        )
        self._style_experiment_cie_legend(cie_ax)

        self.experiment_figure.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.10,
            top=0.90,
            hspace=0.42,
            wspace=0.28,
        )
        self.experiment_canvas.draw_idle()

    def plot_tio2_sio2_experiment_colour_map(self, use_optimized: bool = False) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror("TiO2/SiO2 map", "Load experiment data first.")
            return
        try:
            points = self._tio2_sio2_measurement_points(use_optimized=use_optimized)
            if not points:
                mode = "optimized cached" if use_optimized else "estimated"
                raise ValueError(f"No {mode} TiO2/SiO2/Ag measurement points were found.")
            tio2_min, tio2_max = self._map_bounds([float(point["tio2_nm"]) for point in points])
            sio2_min, sio2_max = self._map_bounds([float(point["sio2_nm"]) for point in points])
            stack = self._build_stack_from_controls()
            model = self._model_from_controls()
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_2d_var.get())
            quality = self.sweep_quality_var.get()

            def task(_progress):
                return run_thickness_sweep_2d(
                    stack=stack,
                    model=model,
                    layer_1="TiO2",
                    layer_2="SiO2",
                    thickness_1_min_nm=tio2_min,
                    thickness_1_max_nm=tio2_max,
                    thickness_2_min_nm=sio2_min,
                    thickness_2_max_nm=sio2_max,
                    angle_deg=angle_deg,
                    num_points_1=num_points,
                    num_points_2=num_points,
                    quality=quality,
                )

            def on_success(result) -> str:
                self._draw_tio2_sio2_experiment_colour_map(result, points, use_optimized=use_optimized)
                self.experiment_info_var.set(
                    f"Plotted {len(points)} measurement colours on TiO2/SiO2/Ag map "
                    f"using {'cached optimized' if use_optimized else 'estimated'} thicknesses."
                )
                return "TiO2/SiO2 colour map complete."

            self._run_background(
                task,
                on_success,
                title="TiO2/SiO2 map",
                busy_message="calculating TiO2/SiO2 colour map",
            )
        except Exception as exc:
            messagebox.showerror("TiO2/SiO2 map", str(exc))

    def plot_single_material_experiment_colour_map(
        self,
        material_name: str,
        use_optimized: bool = False,
    ) -> None:
        if self.experiment_store is None:
            self.load_experiment_samples()
        if self.experiment_store is None:
            messagebox.showerror(f"{material_name} map", "Load experiment data first.")
            return
        try:
            points = self._single_material_measurement_points(material_name, use_optimized=use_optimized)
            if not points:
                raise ValueError(f"No {material_name}-only measurements were found.")
            thickness_min, thickness_max = self._map_bounds(
                [float(point["thickness_nm"]) for point in points]
            )
            stack = self._single_material_stack(material_name)
            model = self._model_from_controls()
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_1d_var.get())
            quality = self.sweep_quality_var.get()

            def task(_progress):
                return run_thickness_sweep_1d(
                    stack=stack,
                    model=model,
                    layer=material_name,
                    thickness_min_nm=thickness_min,
                    thickness_max_nm=thickness_max,
                    angle_deg=angle_deg,
                    num_points=num_points,
                    quality=quality,
                )

            def on_success(result) -> str:
                self._draw_single_material_experiment_colour_map(result, points, material_name, use_optimized)
                self.experiment_info_var.set(
                    f"Plotted {len(points)} {material_name} single-film measurement colours."
                )
                return f"{material_name} colour map complete."

            self._run_background(
                task,
                on_success,
                title=f"{material_name} map",
                busy_message=f"calculating {material_name} colour map",
            )
        except Exception as exc:
            messagebox.showerror(f"{material_name} map", str(exc))

    def _draw_all_measured_cie_with_teeth(
        self,
        cache: CachedExperimentResults,
        row_indices: np.ndarray | None = None,
    ) -> None:
        self.experiment_figure.clear()
        self.experiment_cie_ax = None
        if row_indices is None:
            row_indices = np.arange(cache.count)
        row_indices = np.asarray(row_indices, dtype=int)
        grid = self.experiment_figure.add_gridspec(
            2,
            2,
            height_ratios=[2.2, 1.0],
            width_ratios=[1.05, 1.0],
            hspace=0.34,
            wspace=0.28,
        )
        ab_ax = self.experiment_figure.add_subplot(grid[0, 0])
        lb_ax = self.experiment_figure.add_subplot(grid[0, 1])
        table_ax = self.experiment_figure.add_subplot(grid[1, :])
        table_ax.axis("off")

        measured_lab = np.asarray([xyz_to_lab(xyz) for xyz in cache.measured_xyz[row_indices]], dtype=float)
        measured_rgb = np.clip(np.asarray(cache.measured_rgb[row_indices], dtype=float), 0.0, 1.0)
        all_groups = self._measured_tooth_plot_groups(cache)
        groups = [all_groups[index] for index in row_indices]
        marker_by_group = {
            "smooth Si": "o",
            "smooth Si double polished": "D",
            "rough Si": "^",
            "rough Si double polished": "v",
            "rough Ti": "s",
            "other": "x",
        }
        white_target_lab = (100.0, 0.0, 0.0)
        threshold = 2.7
        theta = np.linspace(0.0, 2.0 * np.pi, 240)
        ab_circle = np.column_stack(
            [
                white_target_lab[1] + threshold * np.cos(theta),
                white_target_lab[2] + threshold * np.sin(theta),
            ]
        )
        lb_circle = np.column_stack(
            [
                white_target_lab[2] + threshold * np.cos(theta),
                white_target_lab[0] + threshold * np.sin(theta),
            ]
        )

        for group_name in (
            "smooth Si",
            "smooth Si double polished",
            "rough Si",
            "rough Si double polished",
            "rough Ti",
            "other",
        ):
            indices = [index for index, value in enumerate(groups) if value == group_name]
            if not indices:
                continue
            marker = marker_by_group[group_name]
            scatter_kwargs = {
                "s": 68,
                "c": measured_rgb[indices],
                "marker": marker,
                "linewidths": 0.9,
                "alpha": 0.92,
                "label": f"Measured {group_name}",
                "zorder": 3,
            }
            if marker != "x":
                scatter_kwargs["edgecolors"] = "#111827"
            ab_ax.scatter(measured_lab[indices, 1], measured_lab[indices, 2], **scatter_kwargs)
            lb_ax.scatter(measured_lab[indices, 2], measured_lab[indices, 0], **scatter_kwargs)

        ab_ax.plot(
            ab_circle[:, 0],
            ab_circle[:, 1],
            color="#f97316",
            linewidth=2.0,
            label=r"Tooth AT$_{ab}$=2.7",
            zorder=4,
        )
        lb_ax.plot(
            lb_circle[:, 0],
            lb_circle[:, 1],
            color="#f97316",
            linewidth=2.0,
            label=r"Tooth AT$_{ab}$=2.7",
            zorder=4,
        )
        for axis, x_value, y_value, xlabel, ylabel in (
            (ab_ax, white_target_lab[1], white_target_lab[2], "a* green-red", "b* blue-yellow"),
            (lb_ax, white_target_lab[2], white_target_lab[0], "b* blue-yellow", "L* lightness"),
        ):
            axis.scatter(
                [x_value],
                [y_value],
                s=185,
                marker="*",
                c="#ffffff",
                edgecolors="#111827",
                linewidths=1.4,
                label="D65 neutral white target",
                zorder=5,
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)

        ab_ax.annotate(
            "D65 neutral white target\nL*=100, a*=0, b*=0",
            xy=(white_target_lab[1], white_target_lab[2]),
            xytext=(white_target_lab[1] + 2.5, white_target_lab[2] + 4.0),
            arrowprops={"arrowstyle": "->", "color": "#f97316", "lw": 1.2},
            fontsize=9,
            color="#111827",
        )
        lb_ax.annotate(
            "White/grey difference\nis mainly L*",
            xy=(white_target_lab[2], white_target_lab[0]),
            xytext=(white_target_lab[2] + 4.0, white_target_lab[0] - 8.0),
            arrowprops={"arrowstyle": "->", "color": "#f97316", "lw": 1.1},
            fontsize=9,
            color="#111827",
        )

        ab_margin = 4.0
        a_values = measured_lab[:, 1]
        b_values = measured_lab[:, 2]
        l_values = measured_lab[:, 0]
        ab_ax.set_xlim(
            min(float(np.nanmin(a_values)), white_target_lab[1] - threshold) - ab_margin,
            max(float(np.nanmax(a_values)), white_target_lab[1] + threshold) + ab_margin,
        )
        ab_ax.set_ylim(
            min(float(np.nanmin(b_values)), white_target_lab[2] - threshold) - ab_margin,
            max(float(np.nanmax(b_values)), white_target_lab[2] + threshold) + ab_margin,
        )
        lb_ax.set_xlim(
            min(float(np.nanmin(b_values)), white_target_lab[2] - threshold) - ab_margin,
            max(float(np.nanmax(b_values)), white_target_lab[2] + threshold) + ab_margin,
        )
        lb_ax.set_ylim(
            min(float(np.nanmin(l_values)), 55.0) - 4.0,
            max(float(np.nanmax(l_values)), white_target_lab[0] + threshold) + 4.0,
        )
        ab_ax.axhline(0.0, color="#52606d", linewidth=0.8, alpha=0.6)
        ab_ax.axvline(0.0, color="#52606d", linewidth=0.8, alpha=0.6)
        lb_ax.axhline(white_target_lab[0], color="#111827", linewidth=0.9, alpha=0.45)
        ab_ax.set_title(
            "Measured colours in CIELAB a*b*\n"
            "Dental acceptability is a real Lab circle here",
            fontsize=13,
            fontweight="semibold",
        )
        lb_ax.set_title(
            "Lightness included: L* vs b*\n"
            "Grey and white separate vertically",
            fontsize=13,
            fontweight="semibold",
        )
        ab_ax.legend(loc="best", fontsize=8)
        lb_ax.legend(loc="best", fontsize=8)
        self.experiment_figure.suptitle(
            "Measured sample colours vs D65 neutral white target "
            r"($AT_{ab}=2.7$ from the article)",
            fontsize=14,
            fontweight="semibold",
        )
        closest_path = self._save_and_draw_white_target_table(
            table_ax,
            cache,
            measured_lab,
            groups,
            white_target_lab,
            row_indices=row_indices,
        )
        self.experiment_figure.text(
            0.5,
            0.015,
            f"Closest-colour table saved: {closest_path}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#52606d",
        )
        self.experiment_figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.86, wspace=0.28)
        self.experiment_canvas.draw_idle()

    def _save_and_draw_white_target_table(
        self,
        ax,
        cache: CachedExperimentResults,
        measured_lab: np.ndarray,
        groups: list[str],
        white_target_lab: tuple[float, float, float],
        row_indices: np.ndarray,
        display_count: int = 12,
    ) -> Path:
        distances = np.linalg.norm(measured_lab - np.asarray(white_target_lab, dtype=float), axis=1)
        measured_rgb = np.clip(np.asarray(cache.measured_rgb[row_indices], dtype=float), 0.0, 1.0)
        rows = []
        for local_index, source_index in enumerate(row_indices):
            rows.append(
                {
                    "rank": "",
                    "sample_name": str(cache.sample_names[source_index]),
                    "measurement": str(cache.measurement_descriptions[source_index]),
                    "stack": str(cache.stack_labels[source_index]),
                    "group": groups[local_index],
                    "L*": float(measured_lab[local_index, 0]),
                    "a*": float(measured_lab[local_index, 1]),
                    "b*": float(measured_lab[local_index, 2]),
                    "delta_e_to_D65_white": float(distances[local_index]),
                    "measured_hex": self._rgb_to_hex(measured_rgb[local_index]),
                }
            )
        rows.sort(key=lambda row: row["delta_e_to_D65_white"])
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        output_dir = Path(__file__).resolve().parent / "outputs" / "colour_targets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "closest_TiO2_ending_measured_colours_to_D65_white.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)

        display_rows = [
            {
                "rank": "D65",
                "sample_name": "neutral white target",
                "group": "reference",
                "L*": white_target_lab[0],
                "a*": white_target_lab[1],
                "b*": white_target_lab[2],
                "delta_e_to_D65_white": 0.0,
                "measured_hex": "#ffffff",
            }
        ]
        display_rows.extend(rows[:display_count])
        cell_text = [
            [
                row["rank"],
                row["sample_name"],
                row["group"],
                f"{float(row['L*']):.1f}",
                f"{float(row['a*']):.1f}",
                f"{float(row['b*']):.1f}",
                f"{float(row['delta_e_to_D65_white']):.2f}",
                "",
            ]
            for row in display_rows
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=("Rank", "Sample", "Group", "L*", "a*", "b*", "Delta E* to D65", "Colour"),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.22)
        for row_index, row in enumerate(display_rows, start=1):
            color = str(row["measured_hex"])
            table[(row_index, 7)].set_facecolor(color)
            table[(row_index, 7)].get_text().set_text("")
        ax.set_title(
            "Closest measured colours to D65 white target, ranked by CIELAB Delta E*",
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
        return output_path

    @staticmethod
    def _rgb_to_hex(rgb: np.ndarray) -> str:
        values = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
        channels = [int(round(channel * 255.0)) for channel in values]
        return "#{:02x}{:02x}{:02x}".format(*channels)

    @staticmethod
    def _stack_ending_material_indices(
        cache: CachedExperimentResults,
        material_name: str,
    ) -> np.ndarray:
        indices: list[int] = []
        for index, stack_label in enumerate(cache.stack_labels.astype(str)):
            if ThinFilmDesignerApp._last_deposited_material(stack_label) == material_name:
                indices.append(index)
        return np.asarray(indices, dtype=int)

    @staticmethod
    def _last_deposited_material(stack_label: str) -> str:
        parts = [part.strip() for part in str(stack_label).split("/") if part.strip()]
        if len(parts) <= 2:
            return ""
        deposited_parts = parts[1:-1]
        if not deposited_parts:
            return ""
        last_layer = deposited_parts[-1]
        tokens = last_layer.split()
        return tokens[-1] if tokens else ""

    @staticmethod
    def _measured_tooth_plot_groups(cache: CachedExperimentResults) -> list[str]:
        groups: list[str] = []
        for substrate, surface in zip(cache.substrate_classes.astype(str), cache.surface_classes.astype(str)):
            if substrate == "Si double polished" and surface == "smooth":
                groups.append("smooth Si double polished")
            elif substrate == "Si double polished" and surface == "rough":
                groups.append("rough Si double polished")
            elif substrate == "Si" and surface == "smooth":
                groups.append("smooth Si")
            elif substrate == "Si" and surface == "rough":
                groups.append("rough Si")
            elif substrate == "Ti" and surface == "rough":
                groups.append("rough Ti")
            else:
                groups.append("other")
        return groups

    def _tio2_sio2_measurement_points(self, use_optimized: bool) -> list[dict[str, object]]:
        points: list[dict[str, object]] = []
        for sample_name in self.experiment_store.sample_names(require_spectra=True):  # type: ignore[union-attr]
            sample = self.experiment_store.load_sample(sample_name)  # type: ignore[union-attr]
            base_thicknesses = self._material_thickness_lookup(sample.layer_estimates)
            if not {"TiO2", "SiO2", "Ag"}.issubset(base_thicknesses):
                continue
            for measurement in sample.measurements:
                if not self._measurement_matches_experiment_sorting(sample_name, measurement):
                    continue
                thicknesses = dict(base_thicknesses)
                optimized_used = False
                if use_optimized:
                    cached = self._best_cached_thickness_fit_for_measurement(
                        sample_name,
                        measurement.description,
                    )
                    if cached is None:
                        continue
                    fit_result = self._load_cached_thickness_fit_result(Path(cached["path"]))
                    thicknesses = self._material_thickness_lookup(fit_result.layer_results, optimized=True)
                    optimized_used = True
                    color = fit_result.measured_color
                    delta_e = fit_result.optimized_delta_e
                else:
                    measured_wavelengths, measured_reflectance = load_reflectance_csv(measurement.csv_path)
                    wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
                    measured_on_grid = np.interp(wavelengths_nm, measured_wavelengths, measured_reflectance)
                    color_cache = prepare_color_conversion(wavelengths_nm)
                    color = self._perceived_color_from_reflectance(measured_on_grid, color_cache)
                    delta_e = None
                if "TiO2" not in thicknesses or "SiO2" not in thicknesses:
                    continue
                points.append(
                    {
                        "sample_name": sample_name,
                        "description": measurement.description,
                        "surface": measurement.surface_class,
                        "substrate": measurement.substrate_group or measurement.substrate_hint or self.substrate_var.get(),
                        "tio2_nm": float(thicknesses["TiO2"]),
                        "sio2_nm": float(thicknesses["SiO2"]),
                        "rgb": color.srgb,
                        "hex": color.hex,
                        "delta_e": delta_e,
                        "optimized": optimized_used,
                    }
                )
        return points

    def _single_material_measurement_points(
        self,
        material_name: str,
        use_optimized: bool,
    ) -> list[dict[str, object]]:
        points: list[dict[str, object]] = []
        for sample_name in self.experiment_store.sample_names(require_spectra=True):  # type: ignore[union-attr]
            sample = self.experiment_store.load_sample(sample_name)  # type: ignore[union-attr]
            base_thicknesses = self._material_thickness_lookup(sample.layer_estimates)
            nonzero_materials = {name for name, thickness in base_thicknesses.items() if thickness > 0}
            if nonzero_materials != {material_name}:
                continue
            for measurement in sample.measurements:
                if not self._measurement_matches_experiment_sorting(sample_name, measurement):
                    continue
                thickness = float(base_thicknesses[material_name])
                optimized_used = False
                if use_optimized:
                    cached = self._best_cached_thickness_fit_for_measurement(
                        sample_name,
                        measurement.description,
                    )
                    if cached is None:
                        continue
                    fit_result = self._load_cached_thickness_fit_result(Path(cached["path"]))
                    thicknesses = self._material_thickness_lookup(fit_result.layer_results, optimized=True)
                    if material_name not in thicknesses:
                        continue
                    thickness = float(thicknesses[material_name])
                    color = fit_result.measured_color
                    delta_e = fit_result.optimized_delta_e
                    optimized_used = True
                else:
                    color = self._measured_color_for_measurement(measurement)
                    delta_e = None
                points.append(
                    {
                        "sample_name": sample_name,
                        "description": measurement.description,
                        "surface": measurement.surface_class,
                        "substrate": measurement.substrate_group or measurement.substrate_hint or self.substrate_var.get(),
                        "thickness_nm": thickness,
                        "rgb": color.srgb,
                        "hex": color.hex,
                        "delta_e": delta_e,
                        "optimized": optimized_used,
                    }
                )
        return points

    def _measurement_matches_experiment_sorting(self, sample_name: str, measurement) -> bool:
        series_filter = self.experiment_series_filter_var.get()
        substrate_filter = self.experiment_substrate_filter_var.get()
        surface_filter = self.experiment_surface_filter_var.get()
        kind_filter = self.experiment_kind_filter_var.get()
        substrate = measurement.substrate_group or measurement.substrate_hint or self.substrate_var.get()
        if series_filter != "All" and sample_series_from_name(sample_name) != series_filter:
            return False
        if substrate_filter != "All" and substrate != substrate_filter:
            return False
        if surface_filter != "All" and measurement.surface_class != surface_filter:
            return False
        if kind_filter != "All" and measurement.measurement_kind != kind_filter:
            return False
        return True

    def _measured_color_for_measurement(self, measurement) -> PerceivedColor:
        measured_wavelengths, measured_reflectance = load_reflectance_csv(measurement.csv_path)
        wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
        measured_on_grid = np.interp(wavelengths_nm, measured_wavelengths, measured_reflectance)
        color_cache = prepare_color_conversion(wavelengths_nm)
        return self._perceived_color_from_reflectance(measured_on_grid, color_cache)

    def _single_material_stack(self, material_name: str):
        substrate_name = self.substrate_var.get()
        substrate = self.materials[substrate_name]
        native_oxide = self._native_oxide_from_controls(substrate_name)
        layer = Layer(self.materials[material_name], max(self.sweep_min_var.get(), 1.0))
        if self._use_effective_interfaces():
            return make_stack_with_interfaces(
                incident_medium=self.materials["air"],
                deposited_layers=[layer],
                substrate=substrate,
                native_oxide=native_oxide,
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
                name=f"{material_name} single-film sweep",
            )
        layers = [layer]
        if native_oxide is not None:
            layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
        return make_stack(
            incident_medium=self.materials["air"],
            substrate=substrate,
            layers=layers,
            name=f"{material_name} single-film sweep",
            display_layers=[layer],
        )

    def _map_bounds(self, values: list[float]) -> tuple[float, float]:
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            return float(self.sweep_min_var.get()), float(self.sweep_max_var.get())
        lower = min(float(self.sweep_min_var.get()), float(np.min(finite)))
        upper = max(float(self.sweep_max_var.get()), float(np.max(finite)))
        span = max(upper - lower, 20.0)
        margin = 0.06 * span
        return max(0.0, lower - margin), upper + margin

    @staticmethod
    def _material_thickness_lookup(layers, optimized: bool = False) -> dict[str, float]:
        values: dict[str, float] = {}
        for layer in layers:
            material_name = str(layer.material_name)
            thickness = (
                float(layer.optimized_thickness_nm)
                if optimized and hasattr(layer, "optimized_thickness_nm")
                else float(layer.thickness_nm)
                if hasattr(layer, "thickness_nm")
                else float(layer.base_thickness_nm)
            )
            values[material_name] = values.get(material_name, 0.0) + thickness
        return values

    def _best_cached_thickness_fit_for_measurement(
        self,
        sample_name: str,
        measurement_description: str,
        optimization_mode: str | None = None,
    ) -> dict[str, object] | None:
        records = self._cached_thickness_fit_records_for_measurement(
            sample_name,
            measurement_description,
            optimization_mode=optimization_mode,
        )
        if not records:
            return None
        return min(records, key=lambda record: (float(record["delta_e"]), -float(record["mtime"])))

    def _cached_thickness_fit_records_for_measurement(
        self,
        sample_name: str,
        measurement_description: str,
        optimization_mode: str | None = None,
    ) -> list[dict[str, object]]:
        cache_dir = default_thickness_optimization_cache_dir(Path(__file__).resolve().parent)
        if not cache_dir.exists():
            return []
        safe_sample = self._safe_cache_prefix(sample_name)
        active_metric = self._current_colour_metric()
        records: list[dict[str, object]] = []
        for path in cache_dir.glob(f"{safe_sample}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                if str(metadata.get("sample_name", "")) != sample_name:
                    continue
                if str(metadata.get("measurement_description", "")) != measurement_description:
                    continue
                mode = str(metadata.get("optimization_mode", "layer"))
                if optimization_mode is not None and mode != optimization_mode:
                    continue
                colour_metric = normalise_colour_metric(metadata.get("colour_metric", COLOUR_METRIC_CIE76))
                if colour_metric != active_metric:
                    continue
                best = data.get("evaluations", {}).get(data.get("best_key"), {})
                if not isinstance(best, dict):
                    continue
                delta = float(best.get("delta_e", float("inf")))
                record = {
                    "path": path,
                    "delta_e": delta,
                    "mtime": path.stat().st_mtime,
                    "optimization_mode": mode,
                    "colour_metric": colour_metric,
                    "stage_label": self._fit_stage_label_from_metadata(metadata),
                    "profile_name": str(metadata.get("profile_name", "")),
                    "model_label": str(metadata.get("model_label", "")),
                    "fit_reflectance_scale": bool(metadata.get("fit_reflectance_scale", False)),
                    "range_percent": float(metadata.get("range_percent_last_run", float("nan"))),
                    "step_percent": float(metadata.get("step_percent", float("nan"))),
                }
                records.append(record)
            except Exception:
                continue
        return records

    @staticmethod
    def _fit_stage_label_from_metadata(metadata: dict[str, object]) -> str:
        mode = str(metadata.get("optimization_mode", "layer"))
        if mode == "material_rate":
            label = "Same-material thickness fit"
        elif mode == "layer":
            label = "Individual thickness fit"
        else:
            label = mode.replace("_", " ").strip().title() or "Thickness fit"
        if bool(metadata.get("fit_reflectance_scale", False)):
            label += " + reflectance scale"
        return label

    def _draw_tio2_sio2_experiment_colour_map(
        self,
        result,
        points: list[dict[str, object]],
        use_optimized: bool,
    ) -> None:
        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(1, 2, width_ratios=[2.8, 1.0], wspace=0.28)
        ax = self.experiment_figure.add_subplot(grid[0, 0])
        legend_ax = self.experiment_figure.add_subplot(grid[0, 1])

        ax.imshow(
            result.rgb_grid,
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_1_nm[0]),
                float(result.thickness_values_1_nm[-1]),
                float(result.thickness_values_2_nm[0]),
                float(result.thickness_values_2_nm[-1]),
            ],
        )
        ax.set_xlim(float(result.thickness_values_1_nm[0]), float(result.thickness_values_1_nm[-1]))
        ax.set_ylim(float(result.thickness_values_2_nm[0]), float(result.thickness_values_2_nm[-1]))
        marker_by_surface = {"smooth": "o", "rough": "^"}
        plotted_labels: set[str] = set()
        for point in points:
            surface = str(point["surface"])
            substrate = str(point["substrate"])
            label = f"{surface}, {substrate}"
            ax.scatter(
                [float(point["tio2_nm"])],
                [float(point["sio2_nm"])],
                s=62,
                marker=marker_by_surface.get(surface, "s"),
                facecolors=[point["rgb"]],
                edgecolors="#111827",
                linewidths=0.9,
                alpha=0.96,
                label=None if label in plotted_labels else label,
                zorder=4,
            )
            plotted_labels.add(label)
        ax.set_xlabel("TiO2 thickness (nm)")
        ax.set_ylabel("SiO2 thickness (nm)")
        ax.set_title(
            "Measured colours on TiO2 / SiO2 / Ag simulated colour sweep\n"
            f"{'Cached optimized thicknesses' if use_optimized else 'Estimated thicknesses'}; "
            f"background stack: {result.stack_label}",
            fontsize=11,
            fontweight="semibold",
        )
        ax.grid(True, color="white", alpha=0.25, linewidth=0.7)
        if plotted_labels:
            ax.legend(loc="upper right", fontsize=8)

        legend_ax.axis("off")
        legend_ax.set_title("Measurement colours", fontweight="semibold")
        lines = []
        for point in points[:30]:
            delta = point["delta_e"]
            delta_text = "" if delta is None else f", {self._delta_e_label()} {float(delta):.1f}"
            lines.append(
                f"{point['sample_name']}: TiO2 {float(point['tio2_nm']):.1f}, "
                f"SiO2 {float(point['sio2_nm']):.1f}, {point['hex']}{delta_text}"
            )
        if len(points) > 30:
            lines.append(f"... and {len(points) - 30} more")
        legend_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=7.5,
            family="Consolas",
        )
        self.experiment_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.90)
        self.experiment_canvas.draw_idle()

    def _draw_single_material_experiment_colour_map(
        self,
        result,
        points: list[dict[str, object]],
        material_name: str,
        use_optimized: bool,
    ) -> None:
        self.experiment_figure.clear()
        grid = self.experiment_figure.add_gridspec(1, 2, width_ratios=[2.8, 1.0], wspace=0.28)
        ax = self.experiment_figure.add_subplot(grid[0, 0])
        legend_ax = self.experiment_figure.add_subplot(grid[0, 1])

        strip = np.repeat(result.rgb_values[np.newaxis, :, :], 24, axis=0)
        ax.imshow(
            strip,
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_nm[0]),
                float(result.thickness_values_nm[-1]),
                0.0,
                1.0,
            ],
        )
        ax.set_xlim(float(result.thickness_values_nm[0]), float(result.thickness_values_nm[-1]))
        ax.set_ylim(0.0, 1.0)
        marker_by_surface = {"smooth": "o", "rough": "^"}
        plotted_labels: set[str] = set()
        for index, point in enumerate(points):
            surface = str(point["surface"])
            substrate = str(point["substrate"])
            label = f"{surface}, {substrate}"
            y_value = 0.38 + 0.24 * ((index % 5) / 4.0)
            ax.scatter(
                [float(point["thickness_nm"])],
                [y_value],
                s=68,
                marker=marker_by_surface.get(surface, "s"),
                facecolors=[point["rgb"]],
                edgecolors="#111827",
                linewidths=0.9,
                alpha=0.98,
                label=None if label in plotted_labels else label,
                zorder=4,
            )
            plotted_labels.add(label)
        ax.set_yticks([])
        ax.set_xlabel(f"{material_name} thickness (nm)")
        ax.set_title(
            f"Measured colours on {material_name} single-film simulated colour sweep\n"
            f"{'Cached optimized thicknesses' if use_optimized else 'Estimated thicknesses'}; "
            f"background stack: {result.stack_label}",
            fontsize=11,
            fontweight="semibold",
        )
        ax.grid(True, axis="x", color="white", alpha=0.35, linewidth=0.7)
        if plotted_labels:
            ax.legend(loc="upper right", fontsize=8)

        legend_ax.axis("off")
        legend_ax.set_title("Measurement colours", fontweight="semibold")
        lines = []
        for point in points[:34]:
            delta = point["delta_e"]
            delta_text = "" if delta is None else f", {self._delta_e_label()} {float(delta):.1f}"
            lines.append(
                f"{point['sample_name']}: {float(point['thickness_nm']):.1f} nm, "
                f"{point['hex']}{delta_text}"
            )
        if len(points) > 34:
            lines.append(f"... and {len(points) - 34} more")
        legend_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=7.5,
            family="Consolas",
        )
        self.experiment_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.86)
        self.experiment_canvas.draw_idle()

    def _draw_sputter_rate_fit_summary(self, results, paths: dict[str, Path]) -> None:
        figure = getattr(self, "rate_groups_figure", self.experiment_figure)
        canvas = getattr(self, "rate_groups_canvas", self.experiment_canvas)
        figure.clear()
        grid = figure.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
        rate_ax = figure.add_subplot(grid[0, 0])
        delta_ax = figure.add_subplot(grid[0, 1])

        labels = [result.group_label for result in results]
        x = np.arange(len(results))
        rate_ax.bar(x, [result.percent_change for result in results], color="#2f6f9f")
        rate_ax.axhline(0.0, color="#111827", linewidth=0.8)
        rate_ax.set_ylabel("Rate change (%)")
        rate_ax.set_title("Rate correction", fontsize=9, fontweight="semibold", pad=4)
        rate_ax.set_xticks(x)
        rate_ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        rate_ax.tick_params(axis="y", labelsize=7)
        rate_ax.grid(True, axis="y", alpha=0.28)

        delta_ax.plot(x, [result.mean_delta_e_before for result in results], marker="o", label="Before")
        delta_ax.plot(x, [result.mean_delta_e_after for result in results], marker="o", label="After")
        delta_ax.set_ylabel(r"Mean $\Delta E^*_{Lab}$")
        delta_ax.set_title("Distance before/after", fontsize=9, fontweight="semibold", pad=4)
        delta_ax.set_xticks(x)
        delta_ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        delta_ax.tick_params(axis="y", labelsize=7)
        delta_ax.grid(True, alpha=0.28)
        delta_ax.legend(fontsize=7)

        figure.suptitle(
            f"Grouped sputter-rate fit from measured colours\nSaved: {paths['csv']}",
            fontsize=8.5,
            fontweight="semibold",
        )
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.36, top=0.84, wspace=0.34)
        canvas.draw_idle()

    def _draw_benchmark_summary(self, summary: pd.DataFrame, plot_path: Path | str) -> None:
        self.experiment_figure.clear()
        if "experiment_group" not in summary.columns:
            summary = summary.copy()
            summary["experiment_group"] = "All samples"

        groups = self._ordered_experiment_groups(summary["experiment_group"].dropna().astype(str).unique())
        heatmap_groups = groups[:6]
        rows = int(np.ceil(len(heatmap_groups) / 2))
        grid = self.experiment_figure.add_gridspec(
            rows,
            3,
            width_ratios=[1.0, 1.0, 0.85],
            wspace=0.38,
            hspace=0.60,
        )
        values = summary["mean_delta_e"].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        vmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        heatmap_axes = []
        last_image = None
        for index, group in enumerate(heatmap_groups):
            ax = self.experiment_figure.add_subplot(grid[index // 2, index % 2])
            heatmap_axes.append(ax)
            group_summary = summary[summary["experiment_group"].astype(str) == group]
            pivot = group_summary.pivot(
                index="constants_profile",
                columns="optical_model",
                values="mean_delta_e",
            )
            last_image = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_title(group, fontweight="semibold", fontsize=10)
            if index // 2 == rows - 1:
                ax.set_xlabel("Optical model", fontsize=8)
            if index % 2 == 0:
                ax.set_ylabel("Constants profile", fontsize=8)
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    value = pivot.values[y, x]
                    if np.isfinite(value):
                        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white")
        if last_image is not None and heatmap_axes:
            cbar = self.experiment_figure.colorbar(last_image, ax=heatmap_axes, fraction=0.025, pad=0.02)
            cbar.set_label(f"Mean {self._delta_e_label()}")

        table_ax = self.experiment_figure.add_subplot(grid[:, 2])
        table_ax.axis("off")
        top = summary.head(12).copy()
        table_ax.set_title("Best combinations", fontsize=9, fontweight="semibold", pad=4)
        lines = [
            f"{row.mean_delta_e:5.2f}  {row.experiment_group}: {row.constants_profile} / {row.optical_model}"
            for row in top.itertuples(index=False)
        ]
        table_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            family="Consolas",
            fontsize=8,
        )
        table_ax.text(
            0.0,
            0.02,
            f"Saved plot:\n{plot_path}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#52606d",
        )
        self.experiment_figure.suptitle(
            "Mean colour distance by experiment group, constants, and optical model",
            fontsize=8.5,
            fontweight="semibold",
        )
        self.experiment_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.84, hspace=0.72, wspace=0.42)
        self.experiment_canvas.draw_idle()

    @staticmethod
    def _save_benchmark_summary_plot(summary: pd.DataFrame, path: Path) -> None:
        if "experiment_group" not in summary.columns:
            summary = summary.copy()
            summary["experiment_group"] = "All samples"
        groups = ThinFilmDesignerApp._ordered_experiment_groups(
            summary["experiment_group"].dropna().astype(str).unique()
        )[:6]
        rows = int(np.ceil(len(groups) / 2))
        fig = Figure(figsize=(15, max(6, 3.8 * rows)), dpi=180)
        axes = []
        grid = fig.add_gridspec(rows, 2, wspace=0.36, hspace=0.62)
        values = summary["mean_delta_e"].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        vmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        image = None
        for index, group in enumerate(groups):
            ax = fig.add_subplot(grid[index // 2, index % 2])
            axes.append(ax)
            pivot = summary[summary["experiment_group"].astype(str) == group].pivot(
                index="constants_profile",
                columns="optical_model",
                values="mean_delta_e",
            )
            image = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title(group, fontweight="semibold")
            ax.set_xlabel("Optical model")
            ax.set_ylabel("Constants profile")
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    value = pivot.values[y, x]
                    if np.isfinite(value):
                        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white")
        if image is not None and axes:
            fig.colorbar(image, ax=axes, label=f"Mean {self._delta_e_label()}")
        fig.suptitle(f"Mean {self._delta_e_label()} by experiment group, constants profile, and optical model", fontweight="semibold")
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.90)
        fig.savefig(path)

    @staticmethod
    def _experiment_group_label(surface_class: str | None, substrate_name: str | None) -> str:
        surface = (surface_class or "unknown").strip().lower()
        substrate = (substrate_name or "unknown").strip().lower()
        if "double polished" in substrate and ("si" in substrate or "silicon" in substrate):
            substrate_label = "Si double polished"
        else:
            substrate_label = "Ti" if substrate.startswith("ti") else "Si" if substrate.startswith("si") else "other"
        if surface == "rough":
            return f"Rough {substrate_label}"
        if surface == "smooth":
            return f"Smooth {substrate_label}"
        return f"Unknown {substrate_label}"

    @staticmethod
    def _ordered_experiment_groups(groups) -> list[str]:
        preferred = ["All samples", "Smooth Si", "Rough Si", "Rough Ti", "Smooth Ti", "Unknown Si", "Unknown Ti"]
        group_set = {str(group) for group in groups}
        ordered = [group for group in preferred if group in group_set]
        ordered.extend(sorted(group_set.difference(ordered)))
        return ordered

    def _draw_cie_panel(
        self,
        ax,
        measured_xy,
        simulated_xy,
        delta_e: float,
        selected_index: int | None = None,
    ) -> None:
        measured_points = np.atleast_2d(np.asarray(measured_xy, dtype=float))
        simulated_points = np.atleast_2d(np.asarray(simulated_xy, dtype=float))
        if selected_index is None:
            selected_index = 0
        selected_index = max(0, min(selected_index, measured_points.shape[0] - 1))
        x_values, y_values, rgb, locus = cie_xy_background()
        self.experiment_cie_ax = ax
        image = ax.imshow(
            rgb,
            origin="lower",
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            aspect="auto",
            interpolation="bilinear",
            rasterized=True,
        )
        locus_polygon = np.vstack([locus, locus[0]])
        clip_patch = PathPatch(
            MplPath(locus_polygon),
            transform=ax.transData,
            facecolor="none",
            edgecolor="none",
        )
        ax.add_patch(clip_patch)
        image.set_clip_path(clip_patch)
        ax.plot(
            locus[:, 0],
            locus[:, 1],
            color="#111827",
            linewidth=1.4,
            label="CIE 1931 locus",
            zorder=2,
        )
        ax.plot(
            [locus[-1, 0], locus[0, 0]],
            [locus[-1, 1], locus[0, 1]],
            color="#6b21a8",
            linewidth=1.4,
            zorder=2,
        )
        ax.scatter(
            measured_points[:, 0],
            measured_points[:, 1],
            s=70,
            c="#0f766e",
            edgecolors="white",
            linewidths=1.4,
            label="Measured",
            zorder=3,
        )
        ax.scatter(
            simulated_points[:, 0],
            simulated_points[:, 1],
            s=70,
            c="#111827",
            edgecolors="white",
            linewidths=1.4,
            marker="s",
            label="Simulated",
            zorder=3,
        )
        ax.scatter(
            [measured_points[selected_index, 0]],
            [measured_points[selected_index, 1]],
            s=145,
            facecolors="none",
            edgecolors="#f97316",
            linewidths=2.0,
            label="Selected",
            zorder=4,
        )
        ax.scatter(
            [simulated_points[selected_index, 0]],
            [simulated_points[selected_index, 1]],
            s=145,
            facecolors="none",
            edgecolors="#f97316",
            linewidths=2.0,
            marker="s",
            zorder=4,
        )
        for measured_point, simulated_point in zip(measured_points, simulated_points):
            ax.plot(
                [measured_point[0], simulated_point[0]],
                [measured_point[1], simulated_point[1]],
                color="#334e68",
                linewidth=1.0,
                alpha=0.75,
            )
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(0.0, 0.9)
        ax.set_xlabel("CIE x")
        ax.set_ylabel("CIE y")
        sizes = self._experiment_plot_text_sizes()
        ax.set_title(
            f"CIE 1931 xy\n{self._delta_e_label()} {delta_e:.2f}",
            fontsize=sizes["title"],
            fontweight="semibold",
        )
        ax.xaxis.label.set_size(sizes["label"])
        ax.yaxis.label.set_size(sizes["label"])
        ax.tick_params(axis="both", labelsize=sizes["tick"])
        ax.grid(True, color="#ffffff", alpha=0.35, linewidth=0.6)
        self._style_experiment_cie_legend(ax)

    @staticmethod
    def _lab_to_xyz(lab: tuple[float, float, float]) -> tuple[float, float, float]:
        lightness, a_value, b_value = lab
        white = np.array([95.047, 100.0, 108.883], dtype=float)
        fy = (float(lightness) + 16.0) / 116.0
        fx = fy + float(a_value) / 500.0
        fz = fy - float(b_value) / 200.0

        def inverse_f(value: float) -> float:
            delta = 6.0 / 29.0
            return value**3 if value > delta else 3.0 * delta**2 * (value - 4.0 / 29.0)

        xyz = white * np.array([inverse_f(fx), inverse_f(fy), inverse_f(fz)], dtype=float)
        return tuple(float(value) for value in xyz)

    def _lab_threshold_xy_curve(
        self,
        center_lab: tuple[float, float, float],
        delta_e: float,
        samples: int = 181,
    ) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, int(samples))
        curve = []
        lightness, a_value, b_value = center_lab
        for angle in angles:
            lab = (
                lightness,
                a_value + float(delta_e) * float(np.cos(angle)),
                b_value + float(delta_e) * float(np.sin(angle)),
            )
            curve.append(xyz_to_xy(self._lab_to_xyz(lab)))
        return np.asarray(curve, dtype=float)

    @staticmethod
    def _xyz_to_display_rgb(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.clip(xyz_to_srgb(xyz), 0.0, 1.0))

    def _on_experiment_scroll(self, event) -> None:
        if self.experiment_cie_ax is None or event.inaxes is not self.experiment_cie_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        ax = self.experiment_cie_ax
        scale = 0.78 if event.button == "up" else 1.28
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = (x_max - x_min) * scale
        y_range = (y_max - y_min) * scale
        x_center = float(event.xdata)
        y_center = float(event.ydata)

        new_x_min = x_center - (x_center - x_min) * scale
        new_x_max = new_x_min + x_range
        new_y_min = y_center - (y_center - y_min) * scale
        new_y_max = new_y_min + y_range

        new_x_min, new_x_max = self._clamp_axis_limits(new_x_min, new_x_max, 0.0, 0.8)
        new_y_min, new_y_max = self._clamp_axis_limits(new_y_min, new_y_max, 0.0, 0.9)

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        self.experiment_canvas.draw_idle()

    @staticmethod
    def _clamp_axis_limits(
        lower: float,
        upper: float,
        outer_lower: float,
        outer_upper: float,
    ) -> tuple[float, float]:
        width = upper - lower
        outer_width = outer_upper - outer_lower
        if width >= outer_width:
            return outer_lower, outer_upper
        if lower < outer_lower:
            upper += outer_lower - lower
            lower = outer_lower
        if upper > outer_upper:
            lower -= upper - outer_upper
            upper = outer_upper
        return lower, upper

    def _draw_color_swatch(self, ax, title: str, rgb, hex_value: str, rgb_255) -> None:
        sizes = self._experiment_plot_text_sizes()
        ax.set_facecolor(rgb)
        ax.set_title(title, fontsize=sizes["swatch_title"], fontweight="semibold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#111827")
        ax.text(
            0.5,
            0.5,
            f"{hex_value}  RGB{rgb_255}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=sizes["swatch_text"],
            fontweight="semibold",
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
        )

    def add_layer(self, material_name: str = "TiO2", thickness_nm: float = 80.0) -> None:
        row_frame = ttk.Frame(self.layers_frame)
        row_frame.pack(fill=tk.X, pady=2)

        material_var = tk.StringVar(value=material_name)
        thickness_var = tk.DoubleVar(value=thickness_nm)
        material_var.trace_add("write", self._schedule_save_gui_settings)
        thickness_var.trace_add("write", self._schedule_save_gui_settings)
        material_var.trace_add("write", self._update_sputter_time_estimate)
        thickness_var.trace_add("write", self._update_sputter_time_estimate)
        row = LayerRow(row_frame, material_var, thickness_var)
        self.layer_rows.append(row)

        grip = ttk.Label(row_frame, text="\u2630", width=2, cursor="fleur")
        grip.pack(side=tk.LEFT)
        for widget in (row_frame, grip):
            widget.bind("<ButtonPress-1>", lambda event, row=row: self._start_layer_drag(event, row))
            widget.bind("<ButtonRelease-1>", lambda event, row=row: self._finish_layer_drag(event, row))

        combo = ttk.Combobox(
            row_frame,
            textvariable=material_var,
            values=self._material_names(),
            width=10,
            state="readonly",
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self._on_stack_changed)
        self._spinbox(row_frame, thickness_var, 0.0, 1000.0, 1.0, self._on_stack_changed).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(row_frame, text="nm").pack(side=tk.LEFT)
        ttk.Button(row_frame, text="-", width=3, command=lambda: self.remove_layer(row)).pack(
            side=tk.RIGHT
        )
        ttk.Button(row_frame, text="\u2193", width=3, command=lambda: self.move_layer(row, 1)).pack(
            side=tk.RIGHT, padx=(2, 0)
        )
        ttk.Button(row_frame, text="\u2191", width=3, command=lambda: self.move_layer(row, -1)).pack(
            side=tk.RIGHT, padx=(2, 0)
        )
        self._refresh_all_layer_choices()
        self.schedule_reflectance_update()
        self._update_sputter_time_estimate()
        self._schedule_save_gui_settings()

    def remove_layer(self, row: LayerRow) -> None:
        if len(self.layer_rows) <= 1:
            messagebox.showinfo("Stack", "Keep at least one layer.")
            return
        self.layer_rows.remove(row)
        row.frame.destroy()
        self._refresh_all_layer_choices()
        self.schedule_reflectance_update()
        self._update_sputter_time_estimate()
        self._schedule_save_gui_settings()

    def move_layer(self, row: LayerRow, delta: int) -> None:
        old_index = self.layer_rows.index(row)
        new_index = max(0, min(len(self.layer_rows) - 1, old_index + delta))
        if old_index == new_index:
            return
        self.layer_rows.pop(old_index)
        self.layer_rows.insert(new_index, row)
        self._repack_layer_rows()
        self._refresh_all_layer_choices()
        self.schedule_reflectance_update()
        self._update_sputter_time_estimate()
        self._schedule_save_gui_settings()

    def _refresh_latest_sputter_rates(self) -> None:
        self.latest_sputter_rates_cache = None
        rates = self._latest_sputter_rates()
        self._update_sputter_time_estimate()
        if rates:
            self.status_var.set(f"Loaded latest sputter rates for {len(rates)} materials.")

    def _latest_sputter_rates(self, allow_load: bool = True) -> dict[str, LatestSputterRate]:
        if self.latest_sputter_rates_cache is not None:
            return self.latest_sputter_rates_cache
        try:
            if self.experiment_store is None:
                if not allow_load:
                    return {}
                self.experiment_store = ExperimentDataStore(self._resolve_experiment_data_path())
            self.latest_sputter_rates_cache = self.experiment_store.latest_sputter_rates()
        except Exception:
            self.latest_sputter_rates_cache = {}
        return self.latest_sputter_rates_cache

    def _update_sputter_time_estimate(self, *_args) -> None:
        if not hasattr(self, "sputter_time_tree"):
            return

        for item in self.sputter_time_tree.get_children():
            self.sputter_time_tree.delete(item)

        if not self.layer_rows:
            self.sputter_time_total_var.set("No deposited layers to estimate.")
            return

        rates = self._latest_sputter_rates(allow_load=self.experiment_store is not None)
        if not rates:
            self.sputter_time_total_var.set("No sputter-rate data found in Reflectivity.")

        total_minutes = 0.0
        known_count = 0
        missing_materials: list[str] = []
        used_sources: set[str] = set()
        for layer_index, row in enumerate(self.layer_rows, start=1):
            material = row.material_var.get().strip()
            try:
                thickness_nm = max(float(row.thickness_var.get()), 0.0)
            except (tk.TclError, TypeError, ValueError):
                thickness_nm = 0.0
            rate = rates.get(material)
            layer_label = f"{layer_index}: {material}"
            if rate is None or rate.rate_nm_per_min <= 0:
                self.sputter_time_tree.insert(
                    "",
                    tk.END,
                    values=(layer_label, "missing", "-", "No rate for material"),
                )
                if material:
                    missing_materials.append(material)
                continue

            minutes = thickness_nm / rate.rate_nm_per_min
            total_minutes += minutes
            known_count += 1
            if rate.source:
                used_sources.add(Path(rate.source).name)
            self.sputter_time_tree.insert(
                "",
                tk.END,
                values=(
                    layer_label,
                    self._format_sputter_rate(rate),
                    self._format_sputter_minutes(minutes),
                    rate.settings_label or rate.period or rate.source,
                ),
            )

        if known_count == 0:
            missing = ", ".join(sorted(set(missing_materials))) or "all layers"
            self.sputter_time_total_var.set(f"No known sputter rates for {missing}.")
            return

        total_text = self._format_sputter_minutes(total_minutes)
        if used_sources:
            source_text = (
                next(iter(used_sources))
                if len(used_sources) == 1
                else f"{len(used_sources)} rate sources"
            )
        else:
            source_text = "Reflectivity rates"
        if missing_materials:
            missing = ", ".join(sorted(set(missing_materials)))
            self.sputter_time_total_var.set(
                f"Known sputter time: {total_text}; missing rates for {missing}. "
                f"Using {source_text}."
            )
        else:
            self.sputter_time_total_var.set(
                f"Total sputter time: {total_text} ({total_minutes:.1f} min). "
                f"Using {source_text}."
            )

    @staticmethod
    def _format_sputter_rate(rate: LatestSputterRate) -> str:
        if rate.error_nm_per_min is not None and rate.error_nm_per_min > 0:
            return f"{rate.rate_nm_per_min:.3g} +/- {rate.error_nm_per_min:.2g}"
        return f"{rate.rate_nm_per_min:.3g}"

    @staticmethod
    def _format_sputter_minutes(minutes: float) -> str:
        minutes = max(float(minutes), 0.0)
        if minutes < 0.05:
            return "<0.1 min"
        if minutes < 60:
            return f"{minutes:.1f} min"
        hours = int(minutes // 60)
        remainder = minutes - (hours * 60)
        return f"{hours} h {remainder:.0f} min"

    def load_constants_editor(self, material_name: str | None = None) -> None:
        name = material_name or self.constants_material_var.get()
        if name not in self.materials:
            return
        self.constants_material_var.set(name)
        wavelengths, n_values, k_values = visible_material_table(self.materials[name])
        lines = ["# wavelength_nm, n, k"]
        lines.extend(
            f"{wl:g}, {n:g}, {k:g}" for wl, n, k in zip(wavelengths, n_values, k_values)
        )
        self.constants_text.delete("1.0", tk.END)
        self.constants_text.insert("1.0", "\n".join(lines))

    def apply_constants_table(self) -> None:
        try:
            wavelengths, n_values, k_values = self._parse_constants_table(self.constants_text.get("1.0", tk.END))
            name = self.constants_material_var.get().strip()
            self.materials[name] = make_tabulated_material(name, wavelengths, n_values, k_values)
            self._refresh_all_layer_choices()
            self.schedule_reflectance_update()
            self.status_var.set(f"Updated optical constants for {name}.")
        except Exception as exc:
            messagebox.showerror("Constants", str(exc))

    def _on_constants_material_selected(self, *_args) -> None:
        self.load_constants_editor()
        self._refresh_material_source_choices()

    def _refresh_material_source_choices(self) -> None:
        if not hasattr(self, "constants_source_combo"):
            return
        material = self.constants_material_var.get().strip()
        choices = list(self._material_source_choices(material))
        self.constants_source_combo.configure(values=choices)
        if choices and self.constants_source_var.get() not in choices:
            self.constants_source_var.set(choices[0])

    def _material_source_choices(self, material_name: str) -> tuple[str, ...]:
        choices: list[str] = []
        for profile in ("current", "legacy_ideal", "legacy_wip"):
            if material_name in built_in_materials(profile):
                choices.append(profile)
        if self.fitted_constants_path.exists():
            try:
                if material_name in load_fitted_materials(built_in_materials("current"), self.fitted_constants_path):
                    choices.append("fitted_single_films")
            except Exception:
                pass
        if self.best_candidate_profile_path.exists():
            try:
                if material_name in load_best_candidate_materials(built_in_materials("current"), self.best_candidate_profile_path):
                    choices.append("best_refractiveindex_candidates")
            except Exception:
                pass
        try:
            candidates = load_candidate_config(default_candidate_config_path(Path(__file__).resolve().parent))
            for candidate in candidates:
                if candidate.material_name == material_name:
                    choices.append(f"refractiveindex.info:{candidate.source_name}")
        except Exception:
            pass
        for profile_name in self._group_candidate_profile_names():
            try:
                path = self._group_candidate_profile_path_from_name(profile_name)
                if material_name in load_best_candidate_materials(built_in_materials("current"), path):
                    choices.append(profile_name)
            except Exception:
                pass
        return tuple(dict.fromkeys(choices))

    def apply_material_source(self) -> None:
        try:
            material_name = self.constants_material_var.get().strip()
            source = self.constants_source_var.get().strip()
            if not source:
                raise ValueError("Choose a source first.")
            if source in {"current", "legacy_ideal", "legacy_wip"}:
                self.materials[material_name] = built_in_materials(source)[material_name]
            elif source == "fitted_single_films":
                self.materials[material_name] = load_fitted_materials(
                    built_in_materials("current"),
                    self.fitted_constants_path,
                )[material_name]
            elif source == "best_refractiveindex_candidates":
                self.materials[material_name] = load_best_candidate_materials(
                    built_in_materials("current"),
                    self.best_candidate_profile_path,
                )[material_name]
            elif source.startswith("best_candidates_"):
                self.materials[material_name] = load_best_candidate_materials(
                    built_in_materials("current"),
                    self._group_candidate_profile_path_from_name(source),
                )[material_name]
            elif source.startswith("refractiveindex.info:"):
                source_name = source.split(":", maxsplit=1)[1]
                root = Path(__file__).resolve().parent
                local_path = (
                    default_candidate_data_dir(root)
                    / safe_candidate_name(material_name)
                    / f"{safe_candidate_name(source_name)}.yml"
                )
                if not local_path.exists():
                    candidates = tuple(
                        candidate
                        for candidate in load_candidate_config(default_candidate_config_path(root))
                        if candidate.material_name == material_name and candidate.source_name == source_name
                    )
                    if not candidates:
                        raise ValueError(f"No configured candidate found for {source}.")
                    download_candidate_records(candidates, default_candidate_data_dir(root))
                self.materials[material_name] = material_from_refractiveindex_yaml(material_name, local_path)
            else:
                raise ValueError(f"Unknown material source: {source}")
            self.load_constants_editor(material_name)
            self.schedule_reflectance_update()
            self.status_var.set(f"{material_name} constants set to {source}.")
        except Exception as exc:
            messagebox.showerror("Material source", str(exc))

    def fit_constants_from_single_films(self) -> None:
        try:
            base_materials = built_in_materials("current")
            sample_data_root = self.experiment_data_path_var.get()
            angle_deg = float(self.angle_var.get())
            fitted_path = self.fitted_constants_path

            def task(_progress):
                result = fit_single_film_constants(
                    sample_data_root=sample_data_root,
                    base_materials=base_materials,
                    angle_deg=angle_deg,
                )
                if not result.materials:
                    raise ValueError("No suitable single-film calibration samples were found.")
                save_fitted_constants(result, fitted_path)
                return result

            def on_success(result) -> str:
                self.material_profile_var.set("fitted_single_films")
                self._on_material_profile_changed()
                fitted_text = ", ".join(
                    f"{material.material_name}: n={material.n:.4g}, k={material.k:.4g}, rms={material.rms_error:.4g}"
                    for material in result.materials
                )
                messagebox.showinfo(
                    "Fitted constants",
                    f"Saved fitted constants to:\n{fitted_path}\n\n{fitted_text}",
                )
                return f"Fitted constants saved: {fitted_text}"

            self._run_background(
                task,
                on_success,
                title="Fitted constants",
                busy_message="fitting constants from single-film samples",
            )
        except Exception as exc:
            messagebox.showerror("Fitted constants", str(exc))

    @staticmethod
    def _candidate_fit_group_options() -> tuple[dict[str, object], ...]:
        return (
            {"key": "si_smooth", "label": "Smooth Si", "substrate": "Si", "surface": "smooth", "kind": "All", "default": True},
            {"key": "si_double_smooth", "label": "Smooth Si double polished", "substrate": "Si double polished", "surface": "smooth", "kind": "All", "default": False},
            {"key": "si_rough", "label": "Rough Si", "substrate": "Si", "surface": "rough", "kind": "All", "default": True},
            {"key": "si_double_rough", "label": "Rough Si double polished", "substrate": "Si double polished", "surface": "rough", "kind": "All", "default": False},
            {"key": "ti_rough", "label": "Rough Ti", "substrate": "Ti", "surface": "rough", "kind": "All", "default": True},
            {"key": "ti_smooth", "label": "Smooth Ti", "substrate": "Ti", "surface": "smooth", "kind": "All", "default": False},
            {"key": "all_smooth", "label": "All smooth", "substrate": "All", "surface": "smooth", "kind": "All", "default": False},
            {"key": "all_rough", "label": "All rough", "substrate": "All", "surface": "rough", "kind": "All", "default": False},
            {"key": "all_samples", "label": "All samples", "substrate": "All", "surface": "All", "kind": "All", "default": False},
        )

    def _selected_candidate_fit_groups(self) -> list[dict[str, object]]:
        selected = []
        for option in self._candidate_fit_group_options():
            variable = self.candidate_fit_group_vars.get(str(option["key"]))
            if variable is not None and variable.get():
                selected.append(option)
        return selected

    def fit_refractiveindex_candidate_constants(self) -> None:
        project_root = Path(__file__).resolve().parent
        sample_data_root = self.experiment_data_path_var.get()
        angle_deg = self.angle_var.get()
        model = self._model_from_controls()
        colour_metric = self._current_colour_metric()
        groups = self._selected_candidate_fit_groups()
        if not groups:
            messagebox.showerror(
                "Refractive-index candidate fit",
                "Choose at least one candidate-fit group on the left side of the Constants tab.",
            )
            return

        def task(progress):
            results = []
            for index, group in enumerate(groups, start=1):
                self._wait_if_paused(progress)
                substrate_choice = str(group["substrate"])
                surface_choice = str(group["surface"])
                kind_choice = str(group["kind"])
                group_label = self._candidate_fit_group_label(substrate_choice, surface_choice, kind_choice)
                progress(index, f"candidate group {index}/{len(groups)}: {group['label']}")
                results.append(
                    (
                        group,
                        group_label,
                        fit_refractiveindex_candidates(
                            project_root=project_root,
                            sample_data_root=sample_data_root,
                            angle_deg=angle_deg,
                            download_missing=True,
                            surface_class_filter=None if surface_choice == "All" else surface_choice,
                            measurement_kind_filter=None if kind_choice == "All" else kind_choice,
                            substrate_filter=None if substrate_choice == "All" else substrate_choice,
                            model=model,
                            fit_group_label=group_label,
                            colour_metric=colour_metric,
                        ),
                    )
                )
            return results

        def on_success(results) -> str:
            self._refresh_material_profile_choices()
            first_group_label = results[0][1]
            self.material_profile_var.set(f"best_candidates_{first_group_label}")
            self._on_material_profile_changed()
            group_lines = []
            for group, group_label, result in results:
                best_text = ", ".join(
                    f"{name}: {row.candidate_label} "
                    f"(RMSE {row.reflectance_rmse:.4f}, {self._delta_e_label()} {row.mean_delta_e:.2f})"
                    for name, row in sorted(result.best_by_material.items())
                )
                group_lines.append(
                    f"{group['label']} -> best_candidates_{group_label}\n"
                    f"{result.best_profile_path}\n{best_text}"
                )
            messagebox.showinfo(
                "Refractive-index candidate fit",
                "Saved candidate rankings and best profiles:\n\n" + "\n\n".join(group_lines),
            )
            return f"Candidate fit complete for {len(results)} group(s)."

        self._run_background(
            task,
            on_success,
            title="Refractive-index candidate fit",
            busy_message="testing refractiveindex.info candidates",
            progress_max=len(groups),
        )

    @staticmethod
    def _candidate_fit_group_label(substrate: str, surface: str, measurement: str) -> str:
        parts = []
        if substrate != "All":
            parts.append(substrate)
        if surface != "All":
            parts.append(surface)
        if measurement != "All":
            parts.append(measurement)
        return "_".join(parts) if parts else "all"

    def import_constants_from_url(self) -> None:
        url = self.constants_url_var.get().strip()
        if not url:
            messagebox.showerror("Constants", "Enter a YAML URL first.")
            return
        try:
            normalized_url = self._normalize_refractiveindex_url(url)
            with urlopen(normalized_url, timeout=20) as response:
                raw_text = response.read().decode("utf-8")
            wavelengths, n_values, k_values = self._parse_refractiveindex_payload(raw_text)
            name = self.constants_material_var.get().strip()
            self.materials[name] = make_tabulated_material(name, wavelengths, n_values, k_values)
            self._refresh_all_layer_choices()
            self.load_constants_editor(name)
            self.schedule_reflectance_update()
            self.status_var.set(f"Imported tabulated optical constants for {name}.")
        except URLError as exc:
            messagebox.showerror("Constants", f"Could not download URL:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Constants", str(exc))

    def schedule_reflectance_update(self) -> None:
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(250, self.update_reflectance)

    def update_reflectance(self) -> None:
        self.update_job = None
        try:
            stack = self._build_stack_from_controls()
            self.current_stack = stack
            wavelengths_nm = wavelength_grid(400.0, 700.0, 151)
            model = self._model_from_controls()
            result = model.simulate(stack, wavelengths_nm, self.angle_var.get())
            perceived_color = perceived_color_from_result(result)

            self.reflectance_figure.clear()
            grid = self.reflectance_figure.add_gridspec(
                2,
                2,
                height_ratios=[1.05, 2.25],
                width_ratios=[1.1, 1.0],
                hspace=0.42,
                wspace=0.20,
            )
            swatch_ax = self.reflectance_figure.add_subplot(grid[0, 0])
            stack_ax = self.reflectance_figure.add_subplot(grid[0, 1])
            ax = self.reflectance_figure.add_subplot(grid[1, :])
            self.reflectance_figure.suptitle(
                f"{result.stack_summary}\nReflected spectrum under D65 illumination",
                fontsize=14,
                fontweight="semibold",
            )
            swatch_ax.set_facecolor(perceived_color.srgb)
            swatch_ax.set_xticks([])
            swatch_ax.set_yticks([])
            for spine in swatch_ax.spines.values():
                spine.set_linewidth(1.8)
                spine.set_color("#111827")
            swatch_ax.text(
                0.5,
                0.5,
                f"{perceived_color.hex}  RGB{perceived_color.srgb_255}",
                ha="center",
                va="center",
                transform=swatch_ax.transAxes,
                fontsize=12,
                fontweight="semibold",
            )
            self._draw_stack_visualization(stack_ax, stack)
            self._fill_reflectance_with_wavelength_colors(ax, result.wavelengths_nm, result.reflectance)
            ax.plot(result.wavelengths_nm, result.reflectance, color="black", linewidth=2)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(float(result.wavelengths_nm.min()), float(result.wavelengths_nm.max()))
            ax.grid(True, color="#bcccdc", alpha=0.35, linewidth=0.8)
            self.reflectance_figure.subplots_adjust(
                left=0.10,
                right=0.96,
                bottom=0.10,
                top=0.82,
                hspace=0.52,
                wspace=0.22,
            )
            self.reflectance_canvas.draw_idle()
            if not hasattr(self, "notebook") or self.notebook.select() == str(self.reflectance_tab):
                warning = self._metal_mirror_roughness_warning(stack)
                suffix = f" {warning}" if warning else ""
                self.status_var.set(f"Updated reflectance: {perceived_color.hex}{suffix}")
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")

    def _metal_mirror_roughness_warning(self, stack) -> str:
        display_layers = stack.display_layers if stack.display_layers is not None else stack.layers
        has_thick_ag = any(
            layer.material.name == "Ag" and layer.thickness_nm >= 50.0 for layer in display_layers
        )
        if not has_thick_ag:
            return ""
        if self._use_effective_interfaces() and self.roughness_thickness_var.get() > 0:
            return "(mixed interface layers are tinting optically thick Ag; turn off Mixed roughness interfaces for smooth white Ag.)"
        if self.rms_roughness_var.get() <= 0:
            return ""
        model_label = self.model_mode_var.get().lower()
        if "rms" not in model_label and "diffuse redistribution" not in model_label:
            return ""
        for layer in display_layers:
            if layer.material.name == "Ag" and layer.thickness_nm >= 50.0:
                return "(RMS/diffuse roughness is tinting optically thick Ag; use Effective interface TMM for smooth white Ag.)"
        return ""

    def run_1d_sweep(self) -> None:
        try:
            stack = self._build_stack_from_controls()
            model = self._model_from_controls()
            layer, occurrence = self._selected_sweep_layer(self.sweep_layer_1_var)
            thickness_min_nm = float(self.sweep_min_var.get())
            thickness_max_nm = float(self.sweep_max_var.get())
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_1d_var.get())
            quality = self.sweep_quality_var.get()

            def task(_progress):
                return run_thickness_sweep_1d(
                    stack=stack,
                    model=model,
                    layer=layer,
                    layer_occurrence=occurrence,
                    thickness_min_nm=thickness_min_nm,
                    thickness_max_nm=thickness_max_nm,
                    angle_deg=angle_deg,
                    num_points=num_points,
                    quality=quality,
                )

            def on_success(result) -> str:
                tab, figure, canvas = self._new_sweep_plot("1D thickness", plot_kind="1d")
                self._draw_1d_sweep_result(result, figure, canvas)
                self._select_sweep_tab(tab)
                return "1D sweep complete."

            self._run_background(
                task,
                on_success,
                title="1D sweep",
                busy_message="running 1D thickness sweep",
            )
        except Exception as exc:
            messagebox.showerror("1D sweep", str(exc))

    def run_2d_sweep(self) -> None:
        try:
            stack = self._build_stack_from_controls()
            model = self._model_from_controls()
            layer_1, occurrence_1 = self._selected_sweep_layer(self.sweep_layer_1_var)
            layer_2, occurrence_2 = self._selected_sweep_layer(self.sweep_layer_2_var)
            thickness_min_nm = float(self.sweep_min_var.get())
            thickness_max_nm = float(self.sweep_max_var.get())
            angle_deg = float(self.angle_var.get())
            num_points = int(self.sweep_points_2d_var.get())
            quality = self.sweep_quality_var.get()

            def task(_progress):
                return run_thickness_sweep_2d(
                    stack=stack,
                    model=model,
                    layer_1=layer_1,
                    layer_1_occurrence=occurrence_1,
                    layer_2=layer_2,
                    layer_2_occurrence=occurrence_2,
                    thickness_1_min_nm=thickness_min_nm,
                    thickness_1_max_nm=thickness_max_nm,
                    thickness_2_min_nm=thickness_min_nm,
                    thickness_2_max_nm=thickness_max_nm,
                    angle_deg=angle_deg,
                    num_points_1=num_points,
                    num_points_2=num_points,
                    quality=quality,
                )

            def on_success(result) -> str:
                tab, figure, canvas = self._new_sweep_plot("2D thickness", plot_kind="2d")
                self._draw_2d_sweep_result(result, figure, canvas)
                self._select_sweep_tab(tab)
                return "2D sweep complete."

            self._run_background(
                task,
                on_success,
                title="2D sweep",
                busy_message="running 2D thickness sweep",
            )
        except Exception as exc:
            messagebox.showerror("2D sweep", str(exc))

    def run_angle_sweep(self) -> None:
        try:
            stack = self._build_stack_from_controls()
            model = self._model_from_controls()
            angle_min_deg = float(self.angle_sweep_min_var.get())
            angle_max_deg = float(self.angle_sweep_max_var.get())
            num_points = int(self.angle_sweep_points_var.get())
            quality = self.sweep_quality_var.get()

            def task(_progress):
                return run_angle_sweep(
                    stack=stack,
                    model=model,
                    angle_min_deg=angle_min_deg,
                    angle_max_deg=angle_max_deg,
                    num_points=num_points,
                    quality=quality,
                )

            def on_success(result) -> str:
                tab, figure, canvas = self._new_sweep_plot("Angle", plot_kind="angle")
                self._draw_angle_sweep_result(result, figure, canvas)
                self._select_sweep_tab(tab)
                return "Angle sweep complete."

            self._run_background(
                task,
                on_success,
                title="Angle sweep",
                busy_message="running angle sweep",
            )
        except Exception as exc:
            messagebox.showerror("Angle sweep", str(exc))

    def _new_sweep_plot(self, title: str, plot_kind: str = "2d"):
        self.sweep_counter += 1
        tab = ttk.Frame(self.sweep_results_notebook)
        header = ttk.Frame(tab)
        header.pack(fill=tk.X)
        figure_size = (7.2, 2.65) if plot_kind in {"1d", "angle"} else (6.8, 5.8)
        figure = Figure(figsize=figure_size, dpi=170)
        self._pack_download_figure_button(
            header,
            lambda fig=figure: fig,
            f"sweep_{title}_{self.sweep_counter}",
        )
        ttk.Button(
            header,
            text="Close",
            command=lambda frame=tab: self.sweep_results_notebook.forget(frame),
        ).pack(side=tk.RIGHT, padx=4, pady=4)
        canvas = FigureCanvasTkAgg(figure, tab)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.sweep_results_notebook.add(tab, text=f"{title} {self.sweep_counter}")
        return tab, figure, canvas

    def _select_sweep_tab(self, tab) -> None:
        self.notebook.select(self.sweep_tab)
        self.sweep_results_notebook.select(tab)

    @staticmethod
    def _short_report_text(text: object, width: int = 76) -> str:
        cleaned = " ".join(str(text or "").split())
        return textwrap.shorten(cleaned, width=width, placeholder="...")

    @staticmethod
    def _sample_sweep_short_name(sample_label: str) -> str:
        short = str(sample_label).split("(", maxsplit=1)[0].strip().rstrip(";")
        return short or ThinFilmDesignerApp._short_report_text(sample_label, width=20)

    def _sweep_report_title(self, headline: str, stack_label: object, *, context: object = "") -> str:
        lines = [self._short_report_text(headline, width=62)]
        context_text = self._short_report_text(context, width=74)
        if context_text:
            lines.append(context_text)
        stack_text = self._short_report_text(stack_label, width=76)
        if stack_text:
            lines.append(stack_text)
        return "\n".join(lines[:3])

    @staticmethod
    def _style_report_sweep_axis(ax, *, title_size: float = 15.0) -> None:
        ax.title.set_fontsize(title_size)
        ax.title.set_fontweight("semibold")
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=11.5, width=1.0, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    def _draw_1d_sweep_result(self, result, figure: Figure, canvas: FigureCanvasTkAgg) -> None:
        figure.clear()
        figure.set_size_inches(7.2, 2.65, forward=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(
            result.rgb_values[np.newaxis, :, :],
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_nm[0]),
                float(result.thickness_values_nm[-1]),
                0.0,
                1.0,
            ],
        )
        ax.set_yticks([])
        ax.set_xlabel(f"{result.layer_name} thickness (nm)")
        ax.set_title(
            self._sweep_report_title(
                f"Predicted colour vs {result.layer_name} thickness",
                result.stack_label,
            )
        )
        self._style_report_sweep_axis(ax)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.67)
        canvas.draw_idle()

    def _draw_2d_sweep_result(self, result, figure: Figure, canvas: FigureCanvasTkAgg) -> None:
        figure.clear()
        figure.set_size_inches(6.8, 5.8, forward=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(
            result.rgb_grid,
            aspect="auto",
            origin="lower",
            extent=[
                float(result.thickness_values_1_nm[0]),
                float(result.thickness_values_1_nm[-1]),
                float(result.thickness_values_2_nm[0]),
                float(result.thickness_values_2_nm[-1]),
            ],
        )
        ax.set_xlabel(f"{result.layer_name_1} thickness (nm)")
        ax.set_ylabel(f"{result.layer_name_2} thickness (nm)")
        ax.set_title(
            self._sweep_report_title(
                f"Predicted colour map: {result.layer_name_1} vs {result.layer_name_2}",
                result.stack_label,
            )
        )
        self._style_report_sweep_axis(ax)
        figure.subplots_adjust(left=0.13, right=0.985, bottom=0.12, top=0.82)
        canvas.draw_idle()

    def _draw_angle_sweep_result(self, result, figure: Figure, canvas: FigureCanvasTkAgg) -> None:
        figure.clear()
        figure.set_size_inches(7.2, 2.65, forward=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(
            result.rgb_values[np.newaxis, :, :],
            aspect="auto",
            origin="lower",
            extent=[
                float(result.angle_values_deg[0]),
                float(result.angle_values_deg[-1]),
                0.0,
                1.0,
            ],
        )
        ax.set_yticks([])
        ax.set_xlabel("Angle of incidence (deg)")
        ax.set_title(self._sweep_report_title("Predicted colour vs angle", result.stack_label))
        self._style_report_sweep_axis(ax)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.67)
        canvas.draw_idle()

    def _layer_thickness_value(self, row: LayerRow, layer_index: int | None = None) -> float:
        thickness_nm = self._try_float_variable(row.thickness_var)
        if thickness_nm is None:
            layer_text = "" if layer_index is None else f" {layer_index}"
            material = row.material_var.get().strip() or "unknown material"
            raise ValueError(f"Enter a numeric thickness for layer{layer_text} ({material}).")
        return thickness_nm

    def _build_stack_from_controls(self):
        deposited_layers = []
        for layer_index, row in enumerate(self.layer_rows, start=1):
            deposited_layers.append(
                Layer(
                    self.materials[row.material_var.get()],
                    self._layer_thickness_value(row, layer_index=layer_index),
                )
            )
        substrate_name = self.substrate_var.get()
        substrate = self.materials[substrate_name]
        native_oxide = self._native_oxide_from_controls(substrate_name)
        if self._use_effective_interfaces():
            return make_stack_with_interfaces(
                incident_medium=self.materials["air"],
                deposited_layers=deposited_layers,
                substrate=substrate,
                native_oxide=native_oxide,
                interface_thickness_nm=self.roughness_thickness_var.get(),
                interface_fraction=self.roughness_fraction_var.get(),
                name="GUI stack",
            )

        optical_layers = list(deposited_layers)
        if native_oxide is not None:
            optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
        return make_stack(
            incident_medium=self.materials["air"],
            substrate=substrate,
            layers=optical_layers,
            name="GUI stack",
            display_layers=deposited_layers,
        )

    def _model_from_controls(self):
        if "diffuse redistribution" in self.model_mode_var.get().lower():
            return TMMWithDiffuseRedistributionModel(
                DiffuseRedistributionSettings(
                    rms_roughness_nm=self.rms_roughness_var.get(),
                    scatter_scale=self.scatter_scale_var.get(),
                    wavelength_exponent=self.scatter_exponent_var.get(),
                    max_scatter_fraction=self.scatter_max_var.get(),
                    diffuse_angle_min_deg=0.0,
                    diffuse_angle_max_deg=80.0,
                    diffuse_angle_samples=17,
                )
            )
        if "RMS" in self.model_mode_var.get():
            return TMMWithRoughnessModel(
                RoughnessCorrectionSettings(rms_roughness_nm=self.rms_roughness_var.get())
            )
        return TMMModel()

    @staticmethod
    def _optical_model_labels() -> tuple[str, ...]:
        return (
            "Ideal TMM",
            "Effective interface TMM",
            "RMS roughness TMM",
            "Effective interface + RMS",
            "Diffuse redistribution TMM",
            "Effective interface + diffuse redistribution",
        )

    @staticmethod
    def _model_for_label(model_label: str, settings: dict[str, float]):
        label = model_label.lower()
        if "diffuse redistribution" in label:
            return TMMWithDiffuseRedistributionModel(
                DiffuseRedistributionSettings(
                    rms_roughness_nm=float(settings.get("rms_roughness_nm", 0.0)),
                    scatter_scale=float(settings.get("scatter_scale", 1.0)),
                    wavelength_exponent=float(settings.get("scatter_exponent", 0.0)),
                    max_scatter_fraction=float(settings.get("max_scatter_fraction", 0.85)),
                    diffuse_angle_min_deg=0.0,
                    diffuse_angle_max_deg=80.0,
                    diffuse_angle_samples=17,
                )
            )
        if "rms" in label:
            return TMMWithRoughnessModel(
                RoughnessCorrectionSettings(
                    rms_roughness_nm=float(settings.get("rms_roughness_nm", 0.0))
                )
            )
        return TMMModel()

    @staticmethod
    def _use_effective_interfaces_for_label(model_label: str) -> bool:
        return "effective interface" in model_label.lower()

    def _model_settings_signature(self) -> dict[str, float]:
        settings: dict[str, float] = {}
        if "diffuse redistribution" in self.model_mode_var.get().lower():
            settings["rms_roughness_nm"] = round(float(self.rms_roughness_var.get()), 6)
            settings["scatter_scale"] = round(float(self.scatter_scale_var.get()), 6)
            settings["scatter_exponent"] = round(float(self.scatter_exponent_var.get()), 6)
            settings["max_scatter_fraction"] = round(float(self.scatter_max_var.get()), 6)
        elif "RMS" in self.model_mode_var.get():
            settings["rms_roughness_nm"] = round(float(self.rms_roughness_var.get()), 6)
        return settings

    def _settings_variables(self) -> dict[str, tk.Variable]:
        variables: dict[str, tk.Variable] = {
            "substrate": self.substrate_var,
            "material_profile": self.material_profile_var,
            "angle_deg": self.angle_var,
            "model_mode": self.model_mode_var,
            "roughness_enabled": self.roughness_enabled_var,
            "interface_nm": self.roughness_thickness_var,
            "mix_fraction": self.roughness_fraction_var,
            "rms_roughness_nm": self.rms_roughness_var,
            "scatter_scale": self.scatter_scale_var,
            "scatter_exponent": self.scatter_exponent_var,
            "max_scatter_fraction": self.scatter_max_var,
            "native_oxide_enabled": self.native_oxide_enabled_var,
            "native_oxide_nm": self.native_oxide_thickness_var,
            "colour_metric": self.colour_metric_var,
            "sweep_layer_1": self.sweep_layer_1_var,
            "sweep_layer_2": self.sweep_layer_2_var,
            "sweep_min_nm": self.sweep_min_var,
            "sweep_max_nm": self.sweep_max_var,
            "sweep_points_1d": self.sweep_points_1d_var,
            "sweep_points_2d": self.sweep_points_2d_var,
            "sweep_quality": self.sweep_quality_var,
            "angle_sweep_min_deg": self.angle_sweep_min_var,
            "angle_sweep_max_deg": self.angle_sweep_max_var,
            "angle_sweep_points": self.angle_sweep_points_var,
            "constants_material": self.constants_material_var,
            "constants_source": self.constants_source_var,
            "experiment_data_path": self.experiment_data_path_var,
            "experiment_series_filter": self.experiment_series_filter_var,
            "experiment_substrate_filter": self.experiment_substrate_filter_var,
            "experiment_surface_filter": self.experiment_surface_filter_var,
            "experiment_kind_filter": self.experiment_kind_filter_var,
            "experiment_plot_text_scale": self.experiment_plot_text_scale_var,
            "fit_composition_filter": self.fit_composition_filter_var,
            "fit_sample_limit": self.fit_sample_limit_var,
            "thickness_opt_range_percent": self.thickness_opt_range_percent_var,
            "thickness_opt_step_percent": self.thickness_opt_step_percent_var,
            "thickness_fit_mode": self.thickness_fit_mode_var,
            "thickness_fit_scale_enabled": self.thickness_fit_scale_enabled_var,
            "thickness_fit_scale_min": self.thickness_fit_scale_min_var,
            "thickness_fit_scale_max": self.thickness_fit_scale_max_var,
            "empirical_fit_materials": self.empirical_fit_materials_var,
            "empirical_fit_k": self.empirical_fit_k_var,
            "empirical_fit_thickness_dependence": self.empirical_fit_thickness_dependence_var,
            "empirical_fit_time_dependence": self.empirical_fit_time_dependence_var,
            "empirical_validation_fraction": self.empirical_validation_fraction_var,
            "empirical_lab_weight": self.empirical_lab_weight_var,
            "empirical_max_evals": self.empirical_max_evals_var,
            "search_target_mode": self.search_target_mode_var,
            "search_target_hex": self.search_target_hex_var,
            "search_target_l": self.search_target_l_var,
            "search_target_a": self.search_target_a_var,
            "search_target_b": self.search_target_b_var,
            "search_min_nm": self.search_min_nm_var,
            "search_max_nm": self.search_max_nm_var,
            "search_points": self.search_points_var,
            "search_iterations": self.search_iterations_var,
            "search_strategy": self.search_strategy_var,
            "search_min_lightness": self.search_min_lightness_var,
            "search_brightness_weight": self.search_brightness_weight_var,
        }
        for key, variable in self.candidate_fit_group_vars.items():
            variables[f"candidate_fit_group_{key}"] = variable
        return variables

    def _load_gui_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        variables = self._settings_variables()
        for key, variable in variables.items():
            if key not in data:
                continue
            try:
                variable.set(data[key])
            except (tk.TclError, TypeError, ValueError):
                continue
        try:
            self.colour_metric_var.set(colour_metric_label(self.colour_metric_var.get()))
        except tk.TclError:
            pass
        layers = data.get("layers")
        if isinstance(layers, list):
            self._saved_layer_settings = [
                layer for layer in layers if isinstance(layer, dict)
            ]

    def _load_saved_layers(self) -> None:
        if not self._saved_layer_settings:
            return
        for row in list(self.layer_rows):
            row.frame.destroy()
        self.layer_rows.clear()
        for layer in self._saved_layer_settings:
            material = str(layer.get("material", "TiO2"))
            if material not in self.materials:
                material = "TiO2" if "TiO2" in self.materials else next(iter(self.materials))
            try:
                thickness = float(layer.get("thickness_nm", 80.0))
            except (TypeError, ValueError):
                thickness = 80.0
            self.add_layer(material, thickness)

    def _save_gui_settings(self) -> None:
        variables = self._settings_variables()
        data: dict[str, object] = {}
        for key, variable in variables.items():
            try:
                value = variable.get()
            except tk.TclError:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            data[key] = value
        layers: list[dict[str, object]] = []
        for row in self.layer_rows:
            thickness_nm = self._try_float_variable(row.thickness_var)
            if thickness_nm is None:
                return
            layers.append(
                {
                    "material": row.material_var.get(),
                    "thickness_nm": thickness_nm,
                }
            )
        data["layers"] = layers
        data["version"] = 1
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _schedule_save_gui_settings(self, *_args) -> None:
        if self.settings_save_job is not None:
            try:
                self.root.after_cancel(self.settings_save_job)
            except tk.TclError:
                pass
        self.settings_save_job = self.root.after(600, self._flush_scheduled_gui_settings)

    def _flush_scheduled_gui_settings(self) -> None:
        self.settings_save_job = None
        self._save_gui_settings()

    def _bind_settings_autosave(self) -> None:
        for variable in self._settings_variables().values():
            try:
                variable.trace_add("write", self._schedule_save_gui_settings)
            except tk.TclError:
                pass

    def _on_close(self) -> None:
        if self.settings_save_job is not None:
            try:
                self.root.after_cancel(self.settings_save_job)
            except tk.TclError:
                pass
            self.settings_save_job = None
        self._save_gui_settings()
        self.root.destroy()

    def _use_effective_interfaces(self) -> bool:
        return self.roughness_enabled_var.get() and "Effective interface" in self.model_mode_var.get()

    def _native_oxide_from_controls(self, substrate_name: str) -> NativeOxide | None:
        if not self.native_oxide_enabled_var.get():
            return None
        default_oxide = native_oxide_for_substrate(self.materials, substrate_name)
        if default_oxide is None:
            return None
        return NativeOxide(default_oxide.material, self.native_oxide_thickness_var.get())

    def _add_default_layers(self) -> None:
        self.add_layer("TiO2", 80.0)
        self.add_layer("SiO2", 120.0)
        self.add_layer("Ag", 40.0)

    def _refresh_all_layer_choices(self) -> None:
        names = self._material_names()
        for row in self.layer_rows:
            for child in row.frame.winfo_children():
                if isinstance(child, ttk.Combobox):
                    child.configure(values=names)
                    break

        layer_labels = self._display_layer_labels()
        self.sweep_layer_1_combo.configure(values=layer_labels)
        self.sweep_layer_2_combo.configure(values=layer_labels)
        if layer_labels:
            if self.sweep_layer_1_var.get() not in layer_labels:
                self.sweep_layer_1_var.set(layer_labels[0])
            if self.sweep_layer_2_var.get() not in layer_labels:
                self.sweep_layer_2_var.set(layer_labels[min(1, len(layer_labels) - 1)])
        if hasattr(self, "constants_material_combo"):
            self.constants_material_combo.configure(values=names)
            if self.constants_material_var.get() not in names and names:
                self.constants_material_var.set(names[0])
            self._refresh_material_source_choices()
        self._refresh_material_profile_choices()
        self._refresh_search_layers()

    def _display_layer_labels(self) -> list[str]:
        labels = []
        counts: dict[str, int] = {}
        for row in self.layer_rows:
            name = row.material_var.get()
            counts[name] = counts.get(name, 0) + 1
            labels.append(f"{name} #{counts[name]}")
        return labels

    def _selected_sweep_layer(self, variable: tk.StringVar) -> tuple[str, int]:
        label = variable.get()
        if not label:
            raise ValueError("Select a sweep layer.")
        name, occurrence_text = label.rsplit(" #", maxsplit=1)
        return name, int(occurrence_text) - 1

    def _current_layer_thickness_for_sweep_label(self, variable: tk.StringVar) -> float:
        material_name, occurrence = self._selected_sweep_layer(variable)
        seen = 0
        for row in self.layer_rows:
            if row.material_var.get() != material_name:
                continue
            if seen == occurrence:
                return self._layer_thickness_value(row)
            seen += 1
        raise ValueError(f"Could not find current thickness for {variable.get()}.")

    def _material_names(self) -> list[str]:
        return sorted(self.materials)

    def _substrate_names(self) -> list[str]:
        return [name for name in ("Si", "Ti", "substrate") if name in self.materials]

    def _on_stack_changed(self, *_args) -> None:
        self._refresh_all_layer_choices()
        self._update_sputter_time_estimate()
        self.schedule_reflectance_update()

    def _on_model_mode_changed(self, *_args) -> None:
        self.roughness_enabled_var.set("Effective interface" in self.model_mode_var.get())
        self._update_roughness_control_states()
        self.schedule_reflectance_update()

    def _on_colour_metric_changed(self, *_args) -> None:
        normalized_label = colour_metric_label(self._current_colour_metric())
        if self.colour_metric_var.get() != normalized_label:
            self.colour_metric_var.set(normalized_label)
            return
        self.experiment_cache_path = self._experiment_cache_path()
        self._update_delta_e_labels()
        if self.experiment_store is not None:
            if self.experiment_cache_path.exists():
                self.load_experiment_cache()
            else:
                self.experiment_cache = None
                self.plots_before_points_cache = None
                self.plots_after_points_cache = None
                self._clear_experiment_results_tree()
                self.experiment_info_var.set(
                    f"No saved {normalized_label} cache yet. Build / refresh saved results to calculate it."
                )
        if hasattr(self, "plots_map_combo"):
            self.plots_before_points_cache = None
            self.plots_after_points_cache = None
            try:
                if self.notebook.tab(self.notebook.select(), "text") == "Plots":
                    self.refresh_plots_map_choices(redraw_only=True)
            except tk.TclError:
                pass
        self.schedule_reflectance_update()

    def _update_delta_e_labels(self) -> None:
        label = self._delta_e_label()
        if hasattr(self, "experiment_results_tree"):
            self.experiment_results_tree.heading("delta_e", text=label)
        if hasattr(self, "colour_distance_tree"):
            self.colour_distance_tree.heading("model_delta", text=f"Model {label}")
            self.colour_distance_tree.heading("fit_delta", text=f"Fit {label}")
        if hasattr(self, "calibration_tree"):
            self.calibration_tree.heading("before", text=f"Before {label}")
            self.calibration_tree.heading("after", text=f"After {label}")

    def _update_roughness_control_states(self) -> None:
        mode = self.model_mode_var.get()
        effective_state = tk.NORMAL if "Effective interface" in mode else tk.DISABLED
        rough_state = tk.NORMAL if ("RMS" in mode or "diffuse redistribution" in mode.lower()) else tk.DISABLED
        for control in getattr(self, "effective_interface_controls", []):
            try:
                control.configure(state=effective_state)
            except tk.TclError:
                pass
        for control in getattr(self, "rms_roughness_controls", []):
            try:
                control.configure(state=rough_state)
            except tk.TclError:
                pass

    def _on_substrate_changed(self, *_args) -> None:
        default_oxide = native_oxide_for_substrate(self.materials, self.substrate_var.get())
        if default_oxide is not None:
            self.native_oxide_thickness_var.set(default_oxide.thickness_nm)
        self._on_stack_changed()

    def _on_material_profile_changed(self, *_args) -> None:
        profile = self.material_profile_var.get()
        show_missing_warning = bool(_args)
        active_layer_materials = [row.material_var.get() for row in self.layer_rows]
        active_substrate = self.substrate_var.get()
        if profile == "fitted_single_films":
            if not self.fitted_constants_path.exists():
                self._fallback_to_current_profile(
                    self._missing_material_profile_message(profile, self.fitted_constants_path),
                    show_missing_warning,
                )
                profile = "current"
            else:
                self.materials = load_fitted_materials(
                    built_in_materials("current"),
                    self.fitted_constants_path,
                )
        elif profile == "best_refractiveindex_candidates":
            if not self.best_candidate_profile_path.exists():
                self._fallback_to_current_profile(
                    self._missing_material_profile_message(profile, self.best_candidate_profile_path),
                    show_missing_warning,
                )
                profile = "current"
            else:
                self.materials = load_best_candidate_materials(
                    built_in_materials("current"),
                    self.best_candidate_profile_path,
                )
        elif profile.startswith("best_candidates_"):
            group_profile_path = self._group_candidate_profile_path_from_name(profile)
            if not group_profile_path.exists():
                self._fallback_to_current_profile(
                    self._missing_material_profile_message(profile, group_profile_path),
                    show_missing_warning,
                )
                profile = "current"
            else:
                self.materials = load_best_candidate_materials(
                    built_in_materials("current"),
                    group_profile_path,
                )
        else:
            self.materials = built_in_materials(profile)
        self.experiment_cache_path = self._experiment_cache_path()
        for row, material_name in zip(self.layer_rows, active_layer_materials):
            if material_name not in self.materials:
                row.material_var.set("TiO2" if "TiO2" in self.materials else self._material_names()[0])
        if active_substrate not in self._substrate_names():
            self.substrate_var.set("Si" if "Si" in self.materials else self._substrate_names()[0])
        self._refresh_all_layer_choices()
        self.load_constants_editor(self.constants_material_var.get())
        self.schedule_reflectance_update()
        if hasattr(self, "experiment_results_tree"):
            self.load_experiment_cache()
            if self.experiment_cache is None:
                self.experiment_info_var.set(
                    f"Constants profile set to {profile}. "
                    "Build experiment results to calculate this profile."
                )
        self.status_var.set(f"Constants profile set to {profile}.")

    def _on_quality_changed(self, *_args) -> None:
        settings = QUALITY_MODES[self.sweep_quality_var.get()]
        self.sweep_points_1d_var.set(int(settings["points_1d"]))
        self.sweep_points_2d_var.set(int(settings["points_2d"]))
        self.angle_sweep_points_var.set(int(settings["points_1d"]))
        self.status_var.set(
            "Quality set to "
            f"{self.sweep_quality_var.get()} "
            f"({settings['wavelength_step_nm']:g} nm wavelength step)."
        )

    def _spinbox(
        self,
        parent,
        variable,
        from_: float,
        to: float,
        increment: float,
        callback,
    ) -> ttk.Spinbox:
        spinbox = ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=10,
        )
        if callback is not None:
            spinbox.configure(command=callback)
            spinbox.bind("<KeyRelease>", callback)
            spinbox.bind("<FocusOut>", callback)
        return spinbox

    def _start_layer_drag(self, event, row: LayerRow) -> None:
        self.dragged_layer = row

    def _finish_layer_drag(self, event, row: LayerRow) -> None:
        pointer_y = self.layers_frame.winfo_pointery() - self.layers_frame.winfo_rooty()
        target_index = 0
        for index, candidate in enumerate(self.layer_rows):
            midpoint = candidate.frame.winfo_y() + candidate.frame.winfo_height() / 2
            if pointer_y > midpoint:
                target_index = index + 1
        old_index = self.layer_rows.index(row)
        if target_index > old_index:
            target_index -= 1
        target_index = max(0, min(len(self.layer_rows) - 1, target_index))
        if old_index != target_index:
            self.layer_rows.pop(old_index)
            self.layer_rows.insert(target_index, row)
            self._repack_layer_rows()
            self._refresh_all_layer_choices()
            self.schedule_reflectance_update()

    def _repack_layer_rows(self) -> None:
        for row in self.layer_rows:
            row.frame.pack_forget()
        for row in self.layer_rows:
            row.frame.pack(fill=tk.X, pady=2)

    def _parse_constants_table(self, text: str):
        wavelengths: list[float] = []
        n_values: list[float] = []
        k_values: list[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace("\t", ",").split(",")]
            if len(parts) == 1:
                parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Could not parse row: {line}")
            wavelengths.append(float(parts[0]))
            n_values.append(float(parts[1]))
            k_values.append(float(parts[2]) if len(parts) >= 3 else 0.0)
        if len(wavelengths) < 2:
            raise ValueError("Enter at least two wavelength rows.")
        order = np.argsort(wavelengths)
        wavelengths_array = np.asarray(wavelengths, dtype=float)[order]
        n_array = np.asarray(n_values, dtype=float)[order]
        k_array = np.asarray(k_values, dtype=float)[order]
        if np.any(np.diff(wavelengths_array) <= 0):
            raise ValueError("Wavelengths must be unique.")
        return wavelengths_array, n_array, k_array

    def _normalize_refractiveindex_url(self, url: str) -> str:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if parsed.netloc.endswith("refractiveindex.info") and "book" in query and "page" in query:
            shelf = query.get("shelf", ["main"])[0]
            book = query["book"][0]
            page = query["page"][0]
            datafile = f"database/data-nk/{shelf}/{book}/{page}.yml"
            return f"https://refractiveindex.info/data_csv.php?datafile={quote(datafile)}"
        return url

    def _parse_refractiveindex_payload(self, raw_text: str):
        try:
            return self._parse_refractiveindex_yaml(raw_text)
        except Exception:
            return self._parse_refractiveindex_csv(raw_text)

    def _parse_refractiveindex_yaml(self, raw_text: str):
        data = yaml.safe_load(raw_text)
        entries = data.get("DATA", []) if isinstance(data, dict) else []
        for entry in entries:
            data_type = str(entry.get("type", "")).lower()
            if "tabulated nk" in data_type or "tabulated n" in data_type:
                rows = []
                for line in str(entry.get("data", "")).splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) >= 2:
                        wavelength_um = float(parts[0])
                        n_value = float(parts[1])
                        k_value = float(parts[2]) if len(parts) >= 3 else 0.0
                        rows.append((wavelength_um * 1000.0, n_value, k_value))
                if len(rows) < 2:
                    raise ValueError("YAML table did not contain enough rows.")
                wavelengths, n_values, k_values = np.asarray(rows, dtype=float).T
                mask = (wavelengths >= 350.0) & (wavelengths <= 800.0)
                if np.count_nonzero(mask) >= 2:
                    return wavelengths[mask], n_values[mask], k_values[mask]
                return wavelengths, n_values, k_values
        raise ValueError(
            "No supported tabulated n/nk data found. Formula-based YAML is not imported yet; "
            "paste visible-spectrum wavelength,n,k rows into the table instead."
        )

    def _parse_refractiveindex_csv(self, raw_text: str):
        n_rows: list[tuple[float, float]] = []
        k_rows: list[tuple[float, float]] = []
        active = ""
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower in {"wl,n", "wl, n"}:
                active = "n"
                continue
            if lower in {"wl,k", "wl, k"}:
                active = "k"
                continue
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) < 2 or active not in {"n", "k"}:
                continue
            row = (float(parts[0]) * 1000.0, float(parts[1]))
            if active == "n":
                n_rows.append(row)
            else:
                k_rows.append(row)

        if len(n_rows) < 2:
            raise ValueError("No supported refractiveindex.info CSV data found.")

        wavelengths = np.asarray([row[0] for row in n_rows], dtype=float)
        n_values = np.asarray([row[1] for row in n_rows], dtype=float)
        if k_rows:
            k_wavelengths = np.asarray([row[0] for row in k_rows], dtype=float)
            k_values_raw = np.asarray([row[1] for row in k_rows], dtype=float)
            k_values = np.interp(wavelengths, k_wavelengths, k_values_raw)
        else:
            k_values = np.zeros_like(n_values)

        mask = (wavelengths >= 350.0) & (wavelengths <= 800.0)
        if np.count_nonzero(mask) >= 2:
            return wavelengths[mask], n_values[mask], k_values[mask]
        return wavelengths, n_values, k_values

    def _fill_reflectance_with_wavelength_colors(self, ax, wavelengths_nm, reflectance) -> None:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        values = np.asarray(reflectance, dtype=float)
        for left, right, y_left, y_right in zip(
            wavelengths[:-1],
            wavelengths[1:],
            values[:-1],
            values[1:],
        ):
            center = 0.5 * (left + right)
            ax.add_patch(
                Polygon(
                    [(left, 0.0), (left, y_left), (right, y_right), (right, 0.0)],
                    closed=True,
                    facecolor=self._wavelength_to_rgb(center),
                    edgecolor="none",
                    alpha=0.95,
                )
            )

    @staticmethod
    def _wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
        wavelength = float(wavelength_nm)
        if wavelength < 380 or wavelength > 780:
            return (0.0, 0.0, 0.0)
        if wavelength < 440:
            rgb = (-(wavelength - 440) / 60, 0.0, 1.0)
        elif wavelength < 490:
            rgb = (0.0, (wavelength - 440) / 50, 1.0)
        elif wavelength < 510:
            rgb = (0.0, 1.0, -(wavelength - 510) / 20)
        elif wavelength < 580:
            rgb = ((wavelength - 510) / 70, 1.0, 0.0)
        elif wavelength < 645:
            rgb = (1.0, -(wavelength - 645) / 65, 0.0)
        else:
            rgb = (1.0, 0.0, 0.0)

        if wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / 40
        elif wavelength <= 700:
            factor = 1.0
        else:
            factor = 0.3 + 0.7 * (780 - wavelength) / 80
        gamma = 0.8
        return tuple((max(channel, 0.0) * factor) ** gamma for channel in rgb)

    def _draw_stack_visualization(self, ax, stack) -> None:
        """Draw a small stylized layer stack using user-facing layers only."""

        ax.set_axis_off()
        ax.set_title("Layer stack", fontsize=11, fontweight="semibold", pad=4)

        display_layers = list(stack.display_layers or stack.layers)
        layers_from_bottom = [(stack.substrate.name, None), *[
            (layer.material.name, layer.thickness_nm) for layer in reversed(display_layers)
        ]]

        total_thickness = sum(thickness or 80.0 for _, thickness in layers_from_bottom)
        x0, width = 0.08, 0.54
        y0, total_height = 0.12, 0.55
        depth_x, depth_y = 0.14, 0.10
        y = y0
        label_targets: list[tuple[str, float, float]] = []

        for name, thickness in layers_from_bottom:
            if thickness is None:
                height = 0.18
            else:
                height = max(0.06, total_height * thickness / max(total_thickness, 1.0))
            color = self._material_display_color(name)
            self._draw_layer_block(ax, x0, y, width, height, depth_x, depth_y, color)
            label = name if thickness is None else f"{name}  {thickness:g} nm"
            label_targets.append(
                (
                    label,
                    x0 + width + depth_x * 0.5,
                    y + height * 0.5 + depth_y * 0.5,
                )
            )
            y += height

        self._draw_stack_labels(ax, label_targets)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 0.95)

    def _draw_stack_labels(self, ax, label_targets: list[tuple[str, float, float]]) -> None:
        label_x = 0.82
        min_gap = 0.085
        sorted_targets = sorted(label_targets, key=lambda item: item[2])
        adjusted: list[tuple[str, float, float]] = []
        last_y = -1.0
        for label, target_x, target_y in sorted_targets:
            text_y = max(target_y, last_y + min_gap)
            adjusted.append((label, target_x, min(text_y, 0.90)))
            last_y = text_y

        for label, target_x, text_y in adjusted:
            target_y = next(item[2] for item in label_targets if item[0] == label)
            ax.annotate(
                label,
                xy=(target_x, target_y),
                xytext=(label_x, text_y),
                ha="left",
                va="center",
                fontsize=8.5,
                color="#111827",
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#52606d",
                    "lw": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )

    def _draw_layer_block(self, ax, x, y, width, height, depth_x, depth_y, color) -> None:
        front = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        right = [
            (x + width, y),
            (x + width + depth_x, y + depth_y),
            (x + width + depth_x, y + height + depth_y),
            (x + width, y + height),
        ]
        top = [
            (x, y + height),
            (x + width, y + height),
            (x + width + depth_x, y + height + depth_y),
            (x + depth_x, y + height + depth_y),
        ]
        ax.add_patch(Polygon(front, closed=True, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.add_patch(
            Polygon(
                right,
                closed=True,
                facecolor=self._shade_color(color, 0.82),
                edgecolor="white",
                linewidth=0.6,
            )
        )
        ax.add_patch(
            Polygon(
                top,
                closed=True,
                facecolor=self._shade_color(color, 1.12),
                edgecolor="white",
                linewidth=0.6,
            )
        )

    @staticmethod
    def _material_display_color(material_name: str) -> str:
        colors = {
            "air": "#eef2ff",
            "TiO2": "#ef7b7b",
            "SiO2": "#f8fafc",
            "Ag": "#8aa4e6",
            "Au": "#f2c66d",
            "Si": "#8f9291",
            "Ti": "#9da7b3",
            "substrate": "#8f9291",
        }
        return colors.get(material_name, "#b7d7c5")

    @staticmethod
    def _shade_color(hex_color: str, factor: float) -> str:
        hex_color = hex_color.lstrip("#")
        channels = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
        shaded = [max(0, min(255, int(channel * factor))) for channel in channels]
        return "#{:02x}{:02x}{:02x}".format(*shaded)

    @staticmethod
    def _rgb_tuple_to_255(rgb) -> tuple[int, int, int]:
        values = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
        return tuple(int(round(channel * 255.0)) for channel in values)

    @staticmethod
    def _rgb_tuple_to_hex(rgb) -> str:
        return "#{:02x}{:02x}{:02x}".format(*ThinFilmDesignerApp._rgb_tuple_to_255(rgb))


def main() -> None:
    root = tk.Tk()
    ThinFilmDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
