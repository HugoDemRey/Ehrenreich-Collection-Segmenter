"""Interactive waveform widget with matplotlib integration for audio visualization.

Provides a Qt widget that displays audio waveforms with interactive capabilities
including position selection, range selection, and playback position tracking.

Author: Hugo Demule
Date: January 2026
"""

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

matplotlib.use("QtAgg")


class InteractiveWaveformWidget(QWidget):
    """Interactive waveform display with matplotlib backend.

    Provides audio waveform visualization with click-to-seek functionality,
    range selection capabilities, and playback position tracking.

    Signals:
        position_clicked (float): Emitted when user clicks on waveform (position in seconds).
    """

    # Signals
    position_clicked = pyqtSignal(float)  # Position in seconds clicked

    def __init__(self):
        """Initialize the waveform widget with default state."""
        super().__init__()

        # Sample data (will be replaced with real audio data later)
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self.current_position = 0.0

        # Range selection attributes
        self.range_start_line = None
        self.range_end_line = None
        self.range_rectangle = None

        self.setup_ui()
        self.create_sample_plot()

    def setup_ui(self):
        """Setup matplotlib canvas and widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setMinimumHeight(400)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(
            """
            QWidget {
                background: rgba(255, 255, 255, 0.05);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
            }
        """
        )

        # Set up the plot
        self.ax = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor("none")  # Transparent background
        self.ax.set_facecolor((0.1, 0.1, 0.1, 0.3))  # Dark semi-transparent background

        # Connect click events
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        # Connect hover events
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_hover)
        # Connect leave event to hide hover line
        self.canvas.mpl_connect("axes_leave_event", self.on_canvas_exit)

        layout.addWidget(self.canvas)

        # Control panel
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Zoom controls
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.setStyleSheet(self.get_control_button_style())
        zoom_in_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_in_btn.clicked.connect(self.zoom_in)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.setStyleSheet(self.get_control_button_style())
        zoom_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_out_btn.clicked.connect(self.zoom_out)

        reset_zoom_btn = QPushButton("Reset View")
        reset_zoom_btn.setStyleSheet(self.get_control_button_style())
        reset_zoom_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_zoom_btn.clicked.connect(self.reset_zoom)

        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(zoom_out_btn)
        controls_layout.addWidget(reset_zoom_btn)
        controls_layout.addStretch()

        # Position info
        self.position_label = QLabel("Position: 0.00s")
        self.position_label.setStyleSheet(
            "color: rgba(255,255,255,0.8); font-size: 12px;"
        )
        controls_layout.addWidget(self.position_label)

        layout.addLayout(controls_layout)

        # Store original view limits for reset
        self.original_xlim = None
        self.original_ylim = None

    def get_control_button_style(self):
        """Get styling for control buttons to match overall application style."""
        return """
        QPushButton {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            min-width: 100px;
            min-height: 30px;
        }
        QPushButton:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        QPushButton:pressed {
            background: rgba(255, 255, 255, 0.05);
        }
        """

    def create_sample_plot(self):
        """Create a sample interactive plot for demonstration."""
        # Generate sample waveform data
        duration = 30.0  # 30 seconds
        sample_rate = 1000  # 1000 samples per second for visualization
        t = np.linspace(0, duration, int(duration * sample_rate))

        # Create a complex waveform with multiple components
        frequency1 = 2.0  # Low frequency component
        frequency2 = 8.0  # Higher frequency component
        noise_level = 0.3

        # Generate realistic-looking audio waveform
        waveform = (
            0.7 * np.sin(2 * np.pi * frequency1 * t) * np.exp(-t / 20)  # Decaying sine
            + 0.4
            * np.sin(2 * np.pi * frequency2 * t)
            * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # Modulated sine
            + noise_level * np.random.normal(0, 1, len(t))  # Noise
        )

        # Add some "musical" structure
        for i in range(5):
            start_idx = int(i * len(t) / 5)
            end_idx = int((i + 0.8) * len(t) / 5)
            amplitude = 0.8 if i % 2 == 0 else 0.5  # Alternate loud/soft sections
            waveform[start_idx:end_idx] *= amplitude

        self.duration = duration

        # Clear the axes
        self.ax.clear()

        # Plot the waveform
        (self.waveform_line,) = self.ax.plot(
            t, waveform, color="#4facfe", linewidth=0.8, alpha=0.9
        )

        # Style the plot
        self.ax.set_xlabel("Time (seconds)", color="white", fontsize=11)
        self.ax.set_ylabel("Amplitude", color="white", fontsize=11)
        self.ax.set_title(
            "🎵 Interactive Audio Waveform (Sample Data)",
            color="white",
            fontsize=13,
            pad=15,
        )

        # Customize appearance
        self.ax.tick_params(colors="white", labelsize=9)
        self.ax.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)
        self.ax.set_facecolor((0.1, 0.1, 0.1, 0.3))

        # Add position marker (vertical line)
        self.position_line = self.ax.axvline(
            x=0, color="#ff6b6b", linewidth=2, alpha=0.8, label="Current Position"
        )

        # Store original limits
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        # Tight layout
        self.figure.tight_layout(pad=2.0)

        # Refresh canvas
        self.canvas.draw()

    def on_canvas_click(self, event):
        """Handle canvas click events."""
        if event.inaxes == self.ax and event.xdata is not None:
            clicked_time = event.xdata
            self.set_position(clicked_time)
            self.position_clicked.emit(clicked_time)

    def on_canvas_hover(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            hover_time = event.xdata
            self.set_hover_position(hover_time)

    def on_canvas_exit(self, event):
        if hasattr(self, "hover_position_line"):
            self.hover_position_line.set_visible(False)
            self.set_position_label()
            self.canvas.draw_idle()

    def set_hover_position(self, position_seconds):
        self.current_hover_position = max(0, min(position_seconds, self.duration))

        # Update position line
        if hasattr(self, "hover_position_line"):
            self.hover_position_line.set_visible(True)
            self.hover_position_line.set_xdata(
                [self.current_hover_position, self.current_hover_position]
            )

        self.set_position_label()

        self.canvas.draw_idle()

    def set_position(self, position_seconds):
        """Set the current position marker."""
        self.current_position = max(0, min(position_seconds, self.duration))

        # Update position line
        if hasattr(self, "position_line"):
            self.position_line.set_xdata([self.current_position, self.current_position])

        # Update position label
        self.set_position_label()

        # Refresh canvas
        self.canvas.draw_idle()

    def set_position_label(self):
        if (
            hasattr(self, "hover_position_line")
            and self.hover_position_line.get_visible()
            and hasattr(self, "current_hover_position")
        ):
            self.position_label.setText(
                f"Hover Position: {self.current_hover_position:.2f}s - Position: {self.current_position:.2f}s"
            )
        else:
            self.position_label.setText(f"Position: {self.current_position:.2f}s")

    def zoom_in(self):
        """Zoom into the waveform centered on the current position."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom factor
        zoom_factor = 0.8

        # Center zoom on current position instead of view center
        center_x = self.current_position
        center_y = (ylim[0] + ylim[1]) / 2

        width_x = (xlim[1] - xlim[0]) * zoom_factor / 2
        width_y = (ylim[1] - ylim[0]) / 2

        self.ax.set_xlim(center_x - width_x, center_x + width_x)
        self.ax.set_ylim(center_y - width_y, center_y + width_y)
        self.canvas.draw()

    def zoom_out(self):
        """Zoom out of the waveform."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom factor
        zoom_factor = 1.25
        center_x = self.current_position
        center_y = (ylim[0] + ylim[1]) / 2

        width_x = (xlim[1] - xlim[0]) * zoom_factor / 2
        width_y = (ylim[1] - ylim[0]) / 2

        # Don't zoom out beyond original limits
        new_xlim = [center_x - width_x, center_x + width_x]
        new_ylim = [center_y - width_y, center_y + width_y]

        if self.original_xlim:
            new_xlim[0] = max(new_xlim[0], self.original_xlim[0])
            new_xlim[1] = min(new_xlim[1], self.original_xlim[1])

        if self.original_ylim:
            new_ylim[0] = max(new_ylim[0], self.original_ylim[0] * 1.1)
            new_ylim[1] = min(new_ylim[1], self.original_ylim[1] * 1.1)

        self.ax.set_xlim(new_xlim[0], new_xlim[1])
        self.ax.set_ylim(new_ylim[0], new_ylim[1])
        self.canvas.draw()

    def reset_zoom(self):
        """Reset zoom to original view."""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw()

    def load_audio_data(self, audio_data, sample_rate):
        """Load real audio data and update the plot."""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate

        # Downsample for visualization if needed
        max_points = 10000  # Maximum points to display for performance
        if len(audio_data) > max_points:
            step = len(audio_data) // max_points
            display_data = audio_data[::step]
            display_time = np.linspace(0, self.duration, len(display_data))
        else:
            display_data = audio_data
            display_time = np.linspace(0, self.duration, len(display_data))

        # Clear and replot
        self.ax.clear()

        # Plot real waveform
        (self.waveform_line,) = self.ax.plot(
            display_time, display_data, color="#4facfe", linewidth=0.6, alpha=0.9
        )

        # Style the plot
        self.ax.set_xlabel("Time (seconds)", color="white", fontsize=11)
        self.ax.set_ylabel("Amplitude", color="white", fontsize=11)
        self.ax.set_title("Audio Waveform", color="white", fontsize=13, pad=15)

        # Customize appearance
        self.ax.tick_params(colors="white", labelsize=9)
        self.ax.grid(True, alpha=0.3, color="white", linestyle="--", linewidth=0.5)
        self.ax.set_facecolor((0.1, 0.1, 0.1, 0.3))

        # Add position marker
        self.position_line = self.ax.axvline(
            x=0, color="#ff6b6b", linewidth=2, alpha=0.8, label="Current Position"
        )
        self.hover_position_line = self.ax.axvline(
            x=0, color="#ff6b6b", linewidth=2, alpha=0.2, label="Hover Position"
        )
        self.hover_position_line.set_visible(False)

        # Store original limits
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        # Tight layout
        self.figure.tight_layout(pad=2.0)

        # Refresh canvas
        self.canvas.draw()

    def load_segmentation_timestamps(self, timestamps):
        """Load segmentation timestamps and display them on the waveform."""
        timestamps_filtered = [ts for ts in timestamps if 0 <= ts <= self.duration]
        for ts in timestamps_filtered:
            self.ax.axvline(
                x=ts, color="#00ff00", linestyle="--", linewidth=1, alpha=0.7
            )
        self.canvas.draw()

    def clear_audio_data(self):
        """Clear audio data to free memory."""
        # Clear large audio data
        self.audio_data = None

        # Reset other attributes
        self.sample_rate = None
        self.duration = 0.0
        self.current_position = 0.0

        # Clear the plot
        if hasattr(self, "ax"):
            self.ax.clear()
            self.canvas.draw()

        # Force garbage collection
        import gc

        gc.collect()

    def clear_range_selection(self):
        """Clear any existing range selection visualization."""
        if self.range_start_line:
            self.range_start_line.remove()
            self.range_start_line = None
        if self.range_end_line:
            self.range_end_line.remove()
            self.range_end_line = None
        if self.range_rectangle:
            self.range_rectangle.remove()
            self.range_rectangle = None
        self.canvas.draw()

    def set_range_start(self, start_seconds):
        """Set the start point of range selection."""
        # Clear existing range visualization
        self.clear_range_selection()

        # Add start line
        self.range_start_line = self.ax.axvline(
            x=start_seconds,
            color="white",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            label="Range Start",
        )
        self.canvas.draw()

    def set_range_end(self, end_seconds):
        """Set the end point of range selection."""
        if self.range_start_line is None:
            return  # No start point set

        start_x = self.range_start_line.get_xdata()[0]

        # Add end line
        self.range_end_line = self.ax.axvline(
            x=end_seconds,
            color="white",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            label="Range End",
        )

        # Add highlighted rectangle between start and end
        ylim = self.ax.get_ylim()
        rect_start = min(start_x, end_seconds)
        rect_width = abs(end_seconds - start_x)

        self.range_rectangle = Rectangle(
            (rect_start, ylim[0]),
            rect_width,
            ylim[1] - ylim[0],
            facecolor="#ffc107",
            alpha=0.2,
            edgecolor="none",
        )
        self.ax.add_patch(self.range_rectangle)

        self.canvas.draw()

    def set_range(self, start_seconds, end_seconds):
        """Set both start and end points of range selection."""
        self.clear_range_selection()

        # Add start and end lines
        self.range_start_line = self.ax.axvline(
            x=start_seconds,
            color="white",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            label="Range Start",
        )
        self.range_end_line = self.ax.axvline(
            x=end_seconds,
            color="white",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            label="Range End",
        )

        # Add highlighted rectangle
        ylim = self.ax.get_ylim()
        rect_start = min(start_seconds, end_seconds)
        rect_width = abs(end_seconds - start_seconds)

        self.range_rectangle = Rectangle(
            (rect_start, ylim[0]),
            rect_width,
            ylim[1] - ylim[0],
            facecolor="#ffc107",
            alpha=0.2,
            edgecolor="none",
        )
        self.ax.add_patch(self.range_rectangle)

        self.canvas.draw()
