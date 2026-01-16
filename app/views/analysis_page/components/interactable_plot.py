"""Interactive plot widget for feature visualization with user interaction.

Combines matplotlib canvas with Qt interactions for displaying audio features
with transition lines, click handling, and hover feedback.

Author: Hugo Demule
Date: January 2026
"""

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from PyQt6.QtWidgets import QWidget as QtWidget
from src.audio.signal import Signal

matplotlib.use("QtAgg")


class InteractablePlot(QWidget):
    """Interactive matplotlib plot widget with transition line support.

    Displays feature sequences with overlay transition lines and provides
    user interaction through click and hover events.

    Signals:
        s_canvas_clicked_position (float, int): Position clicked and mouse button
        s_canvas_hovered_position (float): Position being hovered
        s_figure_exited: Mouse left the figure area
    """

    # Signals for canvas interactions
    s_canvas_clicked_position = pyqtSignal(
        float, int
    )  # position in seconds, button (1=left, 2=middle, 3=right)
    s_canvas_hovered_position = pyqtSignal(float)
    s_figure_exited = pyqtSignal()

    color = False
    opacity = 0.4

    MAX_X_COUNT = 10000

    def __init__(self, name: str, signal: Signal):
        super().__init__()
        self.name = name
        self.signal: Signal = signal

        # Feature widget variables
        self.cursor_position = 0.0  # seconds
        self.preview_cursor_position = 0.0  # seconds
        self.is_plot_generated = False

        # Transition management variables
        self.transitions: list[float] = []  # List to store transition positions
        self.transition_lines = []  # List to store transition line objects
        self.transitions_hidden = False

        # Animation system for blinking transitions
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animated_line = None  # Currently animated line
        self.original_color = None  # Store original color to restore later
        self.animation_start_time = 0
        self.blink_frequency = 0.5  # Number of complete blinks per second

        # Controller reference (set by init method)
        self._controller = None

        # Static plot cache system for smooth resize performance
        self._static_plot_cache = None  # Cache for static plot elements
        self._resize_restore_timer = QTimer()  # Timer to restore content after resize
        self._resize_restore_timer.setSingleShot(True)
        self._resize_restore_timer.timeout.connect(self._restore_static_plot)
        self._is_resize_mode = False  # Track if we're currently in resize mode

        self.setup_ui()
        self.set_cursor_position(0.0)

    @staticmethod
    def init(name: str, signal: Signal, id: str, *args, **kwargs) -> "InteractablePlot":
        """Initialize Interactable Plot with MVC pattern."""
        from controllers.analysis_page.components.interactable_plot_c import (
            InteractablePlotController,
        )

        model = InteractablePlotModel(signal=signal, id=id)
        widget = InteractablePlot(name, signal, *args, **kwargs)
        controller = InteractablePlotController(widget, model)
        widget._controller = controller
        return widget

    def setup_ui(self):
        """Set up the UI layout combining both feature widget and transition functionality."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Title label
        if self.name != "":
            self.title_label = QLabel()
            self._set_title_style(self.name)
            layout.addWidget(self.title_label)

        # Matplotlib figure and layers - dual layer system for optimal performance
        self.figure = Figure(figsize=(8, 2.5))
        self.figure.set_visible(False)

        # Static layer: contains ONLY feature data (extremely rarely redrawn)
        self.static_layer = FigureCanvas(self.figure)
        self.static_layer.setStyleSheet(
            """
            QWidget {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            }
        """
        )
        self.static_layer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Dynamic figure and layer: contains only cursors (overlaid, transparent)
        self.dynamic_figure = Figure(figsize=(8, 2.5))
        self.dynamic_figure.patch.set_facecolor("none")
        self.dynamic_figure.set_visible(False)
        self.dynamic_layer = FigureCanvas(self.dynamic_figure)
        self.dynamic_layer.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
        )
        self.dynamic_layer.setStyleSheet(
            """
            QWidget {
            background: transparent;
            border: None;
            }
        """
        )
        self.dynamic_layer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # IMPORTANT: Make sure the dynamic layer can receive mouse events
        self.dynamic_layer.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
        )

        # Connect canvas events to dynamic layer (it's on top)
        self.dynamic_layer.mpl_connect("button_press_event", self.on_canvas_click)
        self.dynamic_layer.mpl_connect("motion_notify_event", self.on_canvas_hover)

        # Add optimized resize event handler with caching
        def on_canvas_resize(event):
            if (
                hasattr(self, "ax_static")
                and hasattr(self, "ax_dynamic")
                and self.is_plot_generated
            ):
                # Always restart the restore timer when resize is called
                self._resize_restore_timer.stop()

                # If not already in resize mode, cache the static plot and clear it
                if not self._is_resize_mode:
                    self._cache_and_clear_static_plot()
                    self._is_resize_mode = True

                # Perform minimal resize operations on empty axes (very fast)
                self.ax_static.margins(x=0, y=0)
                self.ax_dynamic.margins(x=0, y=0)

                # Apply tight layout to static figure (fast since it's empty)
                self.figure.tight_layout(pad=2.0)

                # Copy positioning from static to dynamic for perfect alignment
                static_pos = self.ax_static.get_position()
                self.ax_dynamic.set_position(static_pos)

                # Apply same layout to dynamic figure
                self.dynamic_figure.tight_layout(pad=2.0)

                # Ensure dynamic axis maintains the same position after tight layout
                self.ax_dynamic.set_position(static_pos)

                # Ensure dynamic axis matches static axis limits
                self.ax_dynamic.set_xlim(self.ax_static.get_xlim())
                self.ax_dynamic.set_ylim(self.ax_static.get_ylim())

                # Redraw both layers (static is empty, so very fast)
                self.static_layer.draw_idle()
                self.dynamic_layer.draw_idle()

                self._ensure_layer_overlay()

                # Start timer to restore static plot after 500ms of no resize
                self._resize_restore_timer.start(1000)

        self.static_layer.mpl_connect("resize_event", on_canvas_resize)
        self.dynamic_layer.mpl_connect("resize_event", on_canvas_resize)

        # Set up Qt right-click context menu on dynamic layer (it's on top)
        self.dynamic_layer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Create container widget for the two-layer system
        self.layer_container = QtWidget()
        self.layer_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        container_layout = QVBoxLayout(self.layer_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Add static layer as the base
        container_layout.addWidget(self.static_layer)

        # Add dynamic layer as overlay using absolute positioning
        self.dynamic_layer.setParent(self.layer_container)

        # Initially make the dynamic layer transparent to mouse events to allow button clicks
        self.dynamic_layer.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

        # Create the generate button with explicit z-order management
        self.generate_button = QPushButton(f"Compute {self.name}", self.dynamic_layer)

        self.generate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_button.clicked.connect(self.on_generate_button_clicked)
        from constants.colors import ACCENT_COLOR_HOVER

        self.generate_button.setStyleSheet(
            f"""
            QPushButton {{
            background-color: gray;
            color: white;
            font-size: 13px;
            border-radius: 8px;
            padding: 2px 12px;
            }}
            QPushButton:hover {{
            background-color: {ACCENT_COLOR_HOVER};
            }}
        """
        )
        self.generate_button.setParent(self.layer_container)

        # Use explicit window flags to ensure button stays on top
        self.generate_button.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.generate_button.raise_()
        self.generate_button.show()
        self.generate_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        def resize_button_and_layers(a0):
            if self.generate_button is None:
                return
            w = self.layer_container.width() * 0.8
            h = self.layer_container.height() * 0.4
            self.generate_button.resize(int(w), int(h))
            self.generate_button.move(
                int((self.layer_container.width() - w) / 2),
                int((self.layer_container.height() - h) / 2),
            )

            # Ensure dynamic layer overlay is properly positioned and sized
            if hasattr(self, "dynamic_layer") and hasattr(
                self, "_ensure_layer_overlay"
            ):
                self._ensure_layer_overlay()

                # Also ensure axes alignment
                if hasattr(self, "ax_static") and hasattr(self, "ax_dynamic"):
                    static_pos = self.ax_static.get_position()
                    self.ax_dynamic.set_position(static_pos)

            if a0:
                a0.accept()

        self.layer_container.resizeEvent = resize_button_and_layers
        layout.addWidget(self.layer_container, stretch=1)

        # Set up dual-axis system for performance optimization
        # Static axis: contains ONLY feature data (never redrawn during cursor updates)
        # Dynamic axis: contains cursors, transitions, thresholds, animations
        self.ax_static = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor("none")
        self.ax_static.set_facecolor((0.1, 0.1, 0.1, 0.3))

        # Dynamic axis: contains only cursors and animated elements (overlaid, transparent)
        self.ax_dynamic = self.dynamic_figure.add_subplot(111)
        self.dynamic_figure.patch.set_facecolor("none")
        self.ax_dynamic.set_facecolor("none")  # Transparent background

        # Copy positioning from static axis for perfect alignment
        self.ax_dynamic.set_position(self.ax_static.get_position())

        # Set the main axis reference (for backward compatibility)
        self.ax = self.ax_static

        # Initialize layer overlay positioning after widget is fully setup
        self._ensure_layer_overlay()

    def create_feature_plot(
        self,
        t: np.ndarray,
        values: np.ndarray,
        threshold: float | None = None,
        color: bool = color,
        opacity: float = opacity,
    ):
        """
        Create the feature plot with T x D shaped data using dual-axis system for optimal performance.
        Args:
            t (np.ndarray): Time axis (1D array of shape (T,)) that starts at 0 seconds, if there is an offset time, it will be computed automatically.
            values (np.ndarray): Feature values (1D array of shape (T,) or 2D array of shape (T, D)).
            threshold (float | None): Optional threshold value to display as a horizontal line.
            color (bool): Whether to use color mapping for multi-dimensional data.
            opacity (float): Opacity of the feature plot (0.0 to 1.0).
        """
        print("(VIEW): Creating feature plot...")

        self.set_confirmed_state(True)

        # Remove the downsampling limitation - we can now handle any number of points efficiently
        # The dual-axis system allows us to plot millions of points without cursor performance issues

        t = t + self.signal.offset_time()

        self.ax_static.clear()
        self.ax_dynamic.clear()

        # Plot all static elements on the static axis
        self._plot_static_elements(t, values, threshold, color, opacity)

        # Set up dynamic elements on the dynamic axis
        self._setup_dynamic_elements()

        # Configure axis appearance
        self._configure_axes_appearance()

        self.figure.tight_layout(pad=2.0)
        self.figure.set_visible(True)
        self.dynamic_figure.tight_layout(pad=2.0)
        self.dynamic_figure.set_visible(True)

        # Draw both layers initially
        self.static_layer.draw()
        self.dynamic_layer.draw()

        # Ensure layer overlay is properly positioned after drawing
        # Copy positioning from static to dynamic for perfect alignment
        static_pos = self.ax_static.get_position()
        self.ax_dynamic.set_position(static_pos)

        # Ensure dynamic axis matches static axis limits
        self.ax_dynamic.set_xlim(self.ax_static.get_xlim())
        self.ax_dynamic.set_ylim(self.ax_static.get_ylim())
        self._ensure_layer_overlay()

    def _set_title_style(self, name: str = "", warning: bool = False):
        if not hasattr(self, "title_label"):
            return

        if name == "":
            name = self.name

        self.title_label.setText(name + " ⚠" if warning else name)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.title_label.setStyleSheet(
            "color: rgba(255, 255, 255, 0.8); font-size: 16px; font-weight: bold;"
        )

        if not warning:
            self.title_label.setToolTip("")
            self.title_label.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.title_label.setToolTip(
                "Parameters have changed since last computation. Please recompute to confirm."
            )
            self.title_label.setCursor(Qt.CursorShape.WhatsThisCursor)

    def _plot_static_elements(
        self,
        t: np.ndarray,
        values: np.ndarray,
        threshold: float | None,
        color: bool,
        opacity: float,
    ):
        """Plot ONLY static elements (feature data) on the static axis. Dynamic elements go to dynamic axis."""
        # Handle T x D shaped values
        if values.ndim == 2:
            T, D = values.shape
            if D == 1:
                # Single dimension - plot as normal curve
                line_color = "#4facfe" if color else "#888888"
                (self.one_dim_data,) = self.ax_static.plot(
                    t, values[:, 0], color=line_color, linewidth=5, alpha=opacity
                )
            else:
                # Multiple dimensions - create spectrogram-like visualization
                # Normalize values to [0, 1] for opacity mapping
                values_norm = (values - values.min()) / (
                    values.max() - values.min() + 1e-8
                )

                # Create frequency/component bin edges (D+1 edges for D bins)
                freq_bin_edges = np.arange(D + 1) - 0.5

                # For pcolormesh with shading='flat', we need to extend the time array
                dt = t[1] - t[0] if len(t) > 1 else 1.0
                t_extended = np.append(t, t[-1] + dt)

                # Create meshgrid for pcolormesh using extended arrays
                T_mesh, F_mesh = np.meshgrid(t_extended, freq_bin_edges)

                # Transpose values to match meshgrid dimensions (D x T)
                values_mesh = values_norm.T

                # Choose colormap based on color flag
                colormap = "viridis" if color else "gray"

                # Create spectrogram-like plot using pcolormesh
                self.multi_dim_plot = self.ax_static.pcolormesh(
                    T_mesh,
                    F_mesh,
                    values_mesh,
                    cmap=colormap,
                    alpha=opacity,
                    shading="flat",
                )

                # Store mesh data for caching purposes
                self.multi_dim_data = {
                    "T_mesh": T_mesh,
                    "F_mesh": F_mesh,
                    "values_mesh": values_mesh,
                    "colormap": colormap,
                }

                # Set y-axis to show bin numbers centered in bins
                self.ax_static.set_ylim(-0.5, D - 0.5)
                # Create ticks at integer positions (center of bins)
                tick_positions = np.arange(0, D, max(1, D // 10))
                self.ax_static.set_yticks(tick_positions)
                self.ax_static.set_yticklabels(
                    [str(int(pos)) for pos in tick_positions]
                )
        else:
            # Handle 1D case (T,) shaped values
            line_color = "#4facfe" if color else "#888888"
            (self.one_dim_data,) = self.ax_static.plot(
                t, values, color=line_color, linewidth=1.2, alpha=opacity
            )

        # Add threshold line on dynamic axis - can change during session
        self.threshold_line = self.ax_dynamic.axhline(
            y=0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.25
        )
        if threshold is not None:
            self.threshold_line.set_ydata([threshold, threshold])
            self.threshold_line.set_visible(True)
        else:
            self.threshold_line.set_visible(False)

    def _setup_dynamic_elements(self):
        """Set up dynamic elements (cursors) on the separate dynamic canvas."""
        # Match both X and Y limits from static axis for perfect alignment
        self.ax_dynamic.set_xlim(self.ax_static.get_xlim())
        self.ax_dynamic.set_ylim(self.ax_static.get_ylim())

        # Create cursor lines on the dynamic axis spanning full Y range for visibility
        y_min, y_max = self.ax_dynamic.get_ylim()

        self.cursor_line = self.ax_dynamic.axvline(
            x=self.cursor_position, color="#ff6b6b", linewidth=1.5, alpha=0.8
        )
        self.cursor_line.set_visible(True)
        self.preview_cursor_line = self.ax_dynamic.axvline(
            x=self.preview_cursor_position, color="#ff6b6b", linewidth=1.2, alpha=0.4
        )
        self.preview_cursor_line.set_visible(False)  # Initially hidden

        # Hide all dynamic axis decorations since it's just for cursors
        self.ax_dynamic.set_xticks([])
        self.ax_dynamic.set_yticks([])
        self.ax_dynamic.set_xticklabels([])
        self.ax_dynamic.set_yticklabels([])
        self.ax_dynamic.tick_params(
            axis="both", which="both", left=False, right=False, top=False, bottom=False
        )

        # add a highlight rectangle for current page (initially invisible)
        self.page_highlight = self.ax_dynamic.axvspan(0, 0, color="yellow", alpha=0.05)
        self.page_highlight.set_visible(False)

        # Remove spines/borders from dynamic axis to keep it invisible
        for spine in self.ax_dynamic.spines.values():
            spine.set_visible(False)

    def _configure_axes_appearance(self):
        """Configure the appearance of both axes ensuring perfect alignment."""
        # Configure static axis appearance
        self.ax_static.set_xlim(
            self.signal.offset_time(),
            self.signal.offset_time() + self.signal.duration_seconds(),
        )
        self.ax_static.set_xlabel("Time (seconds)", color="#b2b2b2", fontsize=9)
        self.ax_static.tick_params(colors="white", labelsize=9)
        self.ax_static.grid(
            True, alpha=0.2, color="white", linestyle="--", linewidth=0.5
        )

        # Apply tight layout to static figure first
        self.figure.tight_layout(pad=2.0)

        # Ensure dynamic axis matches static axis positioning exactly
        static_pos = self.ax_static.get_position()
        self.ax_dynamic.set_position(static_pos)

        # Apply same layout to dynamic figure
        self.dynamic_figure.tight_layout(pad=2.0)

        # Force the dynamic axis to use the same position again after tight layout
        self.ax_dynamic.set_position(static_pos)

    def _ensure_layer_overlay(self):
        """
        Ensure the dynamic layer perfectly overlays the static layer.
        This method resizes the dynamic layer to match the static layer size
        and ensures it is on top for proper interaction handling.
        """
        self.dynamic_layer.resize(self.static_layer.size())
        self.dynamic_layer.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
        )

    def _cache_and_clear_static_plot(self):
        """Cache only the essential plot data (xdata, ydata) and clear the static axis for smooth resize."""
        if self._static_plot_cache is not None:
            # Already cached, don't do it again
            return

        # Store only the essential data that causes performance issues
        cache_data = {}
        cache_data["xlim"] = self.ax_static.get_xlim()
        cache_data["ylim"] = self.ax_static.get_ylim()

        # Cache only the actual data arrays
        if hasattr(self, "one_dim_data"):
            # Store line plot data
            cache_data["line_xdata"] = np.array(self.one_dim_data.get_xdata())
            cache_data["line_ydata"] = np.array(self.one_dim_data.get_ydata())

        if hasattr(self, "multi_dim_data"):
            # Store only the values mesh data (the heavy part)
            cache_data["spectrogram_values"] = self.multi_dim_data["values_mesh"]

        self._static_plot_cache = cache_data

        # Clear the static axis but keep its properties
        self.ax_static.clear()

        # Restore basic axis properties (keeping it minimal for performance)
        self.ax_static.set_xlim(cache_data["xlim"])
        self.ax_static.set_ylim(cache_data["ylim"])

        # Add loading text in the middle of the plot
        x_mid = (cache_data["xlim"][0] + cache_data["xlim"][1]) / 2
        y_mid = (cache_data["ylim"][0] + cache_data["ylim"][1]) / 2
        self.ax_static.text(
            x_mid,
            y_mid,
            "Loading data...",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            alpha=0.7,
        )

    def _restore_static_plot(self):
        """Restore the static plot content from cache after resize is complete."""
        if self._static_plot_cache is None:
            # Nothing to restore
            self._is_resize_mode = False
            return

        # Clear the axis again
        self.ax_static.clear()

        # Recreate line plot if data exists
        if (
            "line_xdata" in self._static_plot_cache
            and "line_ydata" in self._static_plot_cache
        ):
            # Use the same styling as in _plot_static_elements
            line_color = "#4facfe" if self.color else "#888888"
            (self.one_dim_data,) = self.ax_static.plot(
                self._static_plot_cache["line_xdata"],
                self._static_plot_cache["line_ydata"],
                color=line_color,
                linewidth=1.2,
                alpha=self.opacity,
            )

        # Recreate spectrogram if data exists
        if "spectrogram_values" in self._static_plot_cache and hasattr(
            self, "_spectrogram_mesh_data"
        ):
            try:
                # Use the original mesh data but with cached values
                self.multi_dim_plot = self.ax_static.pcolormesh(
                    self.multi_dim_data["T_mesh"],
                    self.multi_dim_data["F_mesh"],
                    self._static_plot_cache["spectrogram_values"],
                    cmap=self.multi_dim_data["colormap"],
                    alpha=self.opacity,
                    shading="flat",
                )

                # Restore y-axis settings for spectrogram
                D = self._static_plot_cache["spectrogram_values"].shape[0]
                self.ax_static.set_ylim(-0.5, D - 0.5)
                tick_positions = np.arange(0, D, max(1, D // 10))
                self.ax_static.set_yticks(tick_positions)
                self.ax_static.set_yticklabels(
                    [str(int(pos)) for pos in tick_positions]
                )
            except Exception as e:
                print(f"Failed to restore spectrogram: {e}")

        # Restore axis properties
        self.ax_static.set_xlim(self._static_plot_cache["xlim"])
        self.ax_static.set_ylim(self._static_plot_cache["ylim"])

        # Restore appearance settings (same as _configure_axes_appearance)
        self.ax_static.set_xlabel("Time (seconds)", color="#b2b2b2", fontsize=9)
        self.ax_static.tick_params(colors="white", labelsize=9)
        self.ax_static.grid(
            True, alpha=0.2, color="white", linestyle="--", linewidth=0.5
        )

        # Clear cache and exit resize mode
        self._static_plot_cache = None
        self._is_resize_mode = False

        # Redraw the static layer with restored content
        self.static_layer.draw()

    # Dynamic canvas event handlers
    def on_canvas_click(self, event):
        """Handle mouse click events on the canvas."""
        # Check if click is on either axis (static or dynamic)
        if event.inaxes == self.ax_static or event.inaxes == self.ax_dynamic:
            self.s_canvas_clicked_position.emit(event.xdata, event.button)

    def on_canvas_hover(self, event):
        """Handle mouse hover events on the canvas."""
        # Check if hover is on either axis (static or dynamic)
        if event.inaxes == self.ax_static or event.inaxes == self.ax_dynamic:
            if event.xdata is not None:
                self.s_canvas_hovered_position.emit(event.xdata)
        else:
            self.s_figure_exited.emit()

    def on_generate_button_clicked(self):
        """Handle generate button click."""
        # Delete the generate button after click
        if self.generate_button is not None:
            self.generate_button.deleteLater()
            self.generate_button = None

        self.show_cursor(True)

        self.is_plot_generated = True

    def activate_plot(self):
        """Activate the plot by hiding the generate button and showing the cursor."""
        if self.generate_button is not None:
            self.generate_button.deleteLater()
            self.generate_button = None
        self.show_cursor(True)
        self.is_plot_generated = True

    # Feature widget methods - now using separate layers for maximum performance
    def set_threshold_line(self, threshold: float | None):
        """
        Set or update the threshold line on the plot.
        if threshold is None, hide the line.
        """
        if threshold is not None:
            self.threshold_line.set_ydata([threshold, threshold])
            self.threshold_line.set_visible(True)
        else:
            self.threshold_line.set_visible(False)

        # Only redraw dynamic layer since threshold is now on dynamic axis - ultra fast!
        self.dynamic_layer.draw_idle()

    def set_cursor_position(self, position_seconds: float):
        """Set the cursor position marker (vertical red line) - ultra-fast with separate dynamic layer."""
        self.cursor_position = max(
            self.signal.offset_time(),
            min(
                position_seconds,
                self.signal.offset_time() + self.signal.duration_seconds(),
            ),
        )
        if hasattr(self, "cursor_line"):
            self.cursor_line.set_xdata([self.cursor_position, self.cursor_position])
            # Only redraw the dynamic layer - extremely fast regardless of data size!
            self.dynamic_layer.draw_idle()

    def show_cursor(self, show: bool):
        """Show or hide the cursor - ultra-fast with separate dynamic layer."""
        if hasattr(self, "cursor_line"):
            self.cursor_line.set_visible(show)
            # Only redraw dynamic layer - never touches static data!
            self.dynamic_layer.draw_idle()

    def show_preview_cursor(self, show: bool):
        """Show or hide the preview cursor - ultra-fast with separate dynamic layer."""
        if hasattr(self, "preview_cursor_line"):
            self.preview_cursor_line.set_visible(show)
            # Only redraw dynamic layer - never touches static data!
            self.dynamic_layer.draw_idle()

    def set_preview_cursor_position(self, position_seconds: float):
        """Set the preview cursor position (for hover effects) - ultra-fast with separate dynamic layer."""
        self.preview_cursor_position = max(
            self.signal.offset_time(),
            min(
                position_seconds,
                self.signal.offset_time() + self.signal.duration_seconds(),
            ),
        )
        if hasattr(self, "preview_cursor_line"):
            self.preview_cursor_line.set_xdata(
                [self.preview_cursor_position, self.preview_cursor_position]
            )
            # Only redraw the dynamic layer - extremely fast regardless of data size!
            self.dynamic_layer.draw_idle()

    # Transition management methods - using static layer (only redrawn when transitions change)
    def add_transition_line(self, pos_seconds, color="yellow"):
        """Add a vertical line at the specified position to indicate a transition."""
        pos_seconds = pos_seconds + self.signal.offset_time()
        print("(VIEW): Adding transition line at", pos_seconds)
        if pos_seconds in self.transitions:
            return
        # Add transition to DYNAMIC axis since transitions are dynamic content
        line = self.ax_dynamic.axvline(
            pos_seconds, color=color, linestyle="--", linewidth=1.5, alpha=1.0
        )
        self.transition_lines.append(line)
        if pos_seconds not in self.transitions:
            self.transitions.append(float(pos_seconds))

        self.dynamic_layer.draw_idle()

    def remove_transition_line(self, pos_seconds):
        """Remove the transition line at the specified position."""
        if pos_seconds in self.transitions:
            index = self.transitions.index(pos_seconds)
            line = self.transition_lines[index]
            line.remove()  # Remove line from axes
            del self.transition_lines[index]
            del self.transitions[index]

            # Only redraw the DYNAMIC layer when transitions change - ultra fast!
            self.dynamic_layer.draw_idle()

    def remove_all_transition_lines(self):
        """Remove all transition lines from the canvas."""
        for line in self.transition_lines:
            try:
                # Check if line is still in the axes before removing
                if hasattr(line, "axes") and line.axes is not None:
                    line.remove()
                elif hasattr(line, "figure") and line.figure is not None:
                    # Try removing from figure if not in axes
                    line.figure.delaxes(line)
            except (NotImplementedError, ValueError, AttributeError) as e:
                # Some artists cannot be removed - just ignore
                print(f"Warning: Could not remove transition line: {e}")
                pass
        self.transition_lines.clear()
        self.transitions.clear()

        # Only redraw the DYNAMIC layer when transitions change - ultra fast!
        self.dynamic_layer.draw_idle()

    def hide_transition_lines(self, hide: bool):
        """Show or hide all transition lines."""
        for line in self.transition_lines:
            line.set_visible(not hide)
        self.transitions_hidden = hide

        # Only redraw the DYNAMIC layer when transitions change - ultra fast!
        self.dynamic_layer.draw_idle()

    def start_transition_animation(self, line_index):
        """Start blinking animation for the transition line at the given index."""
        if line_index < len(self.transition_lines):
            # Stop any current animation
            self.stop_animation()

            # Set up new animation
            self.animated_line = self.transition_lines[line_index]
            # Store original color to restore later
            self.original_color = self.animated_line.get_color()
            # Change color to white
            self.animated_line.set_color("white")
            self.animation_start_time = 0

            # Start the animation timer (update every 50ms for smooth animation)
            self.animation_timer.start(50)

    def stop_animation(self):
        """Stop the current animation and restore line to original state."""
        if self.animation_timer.isActive():
            self.animation_timer.stop()

        if self.animated_line is not None:
            # Restore full opacity and original color
            self.animated_line.set_alpha(1.0)
            if self.original_color is not None:
                self.animated_line.set_color(self.original_color)

            # Only redraw dynamic layer since animated line is now on dynamic axis - ultra fast!
            self.dynamic_layer.draw_idle()

            self.animated_line = None
            self.original_color = None

    def update_animation(self):
        """Update the animation frame - called by timer. Very fast with separate layers."""
        if self.animated_line is None:
            self.stop_animation()
            return

        # Calculate elapsed time
        elapsed_time = self.animation_start_time
        self.animation_start_time += 50  # Add 50ms for each frame

        # Animation runs indefinitely until stopped by another click
        # Calculate opacity based on sine wave for smooth blinking
        time_in_seconds = elapsed_time / 1000.0  # Convert to seconds
        blink_cycle = time_in_seconds * self.blink_frequency * 2 * np.pi
        opacity = 0.3 + 0.7 * abs(np.sin(blink_cycle))  # Oscillates between 0.3 and 1.0

        # Apply opacity to the animated line
        self.animated_line.set_alpha(opacity)

        # With separate layers, we only need to redraw the dynamic layer for animations
        # since animated transitions are now on the dynamic layer - ultra fast updates!
        self.dynamic_layer.draw_idle()

    def set_cursor_type(self, cursor_type):
        """Set the cursor type for the dynamic layer (it's on top)."""
        self.dynamic_layer.setCursor(cursor_type)

    def show_context_menu(self, position: float, transition_pos: float | None):
        pass

    def set_confirmed_state(self, confirmed: bool):
        self._set_title_style(warning=not confirmed, name=self.name)
        print("(INTERACTABLE PLOT VIEW): Setting confirmed state to", confirmed)
