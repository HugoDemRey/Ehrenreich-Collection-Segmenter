def get_starting_page_style() -> str:
    """Get the analysis page styling to match the look and feel."""
    return """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                       stop:0 #2C3E50, stop:1 #4A6741);
        }
        
        QWidget {
            background: transparent;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Add these missing widget styles */
        QStackedWidget {
            background: transparent;
        }
        
        QButtonGroup {
            background: transparent;
        }
        
        QHBoxLayout, QVBoxLayout {
            background: transparent;
        }
        
        /* Frame styles - important for your analysis modules */
        QFrame {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }
        
        QFrame#card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
        }

        /* Button styling with all states */
        QPushButton {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            color: white;
            padding: 8px 15px;
            font-weight: 500;
        }
        
        QPushButton:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        QPushButton:pressed {
            background: rgba(255, 255, 255, 0.3);
        }
        
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.4);
        }
        
        QPushButton:checked {
            background: #4facfe;
            color: white;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            color: white;
            padding: 8px;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 2px solid #4facfe;
        }
        
        /* Combo boxes and spinboxes */
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            color: white;
            padding: 5px;
            min-height: 20px;
        }
        
        QComboBox::drop-down, QSpinBox::drop-down, QDoubleSpinBox::drop-down {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        
        QComboBox::down-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            color: white;
        }
        
        QComboBox QAbstractItemView {
            background: #2b2b2b;
            color: white;
            selection-background-color: #4facfe;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Checkboxes and radio buttons */
        QCheckBox, QRadioButton {
            color: white;
            spacing: 8px;
        }
        
        QCheckBox::indicator, QRadioButton::indicator {
            width: 18px;
            height: 18px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 3px;
        }
        
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background: #4facfe;
            border: 1px solid #4facfe;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: #4facfe;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        
        QSlider::sub-page:horizontal {
            background: #4facfe;
            border-radius: 3px;
        }
        
        /* Tab widgets */
        QTabWidget::pane {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 5px;
        }
        
        QTabBar::tab {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        
        QTabBar::tab:selected {
            background: #4facfe;
        }
        
        QTabBar::tab:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Menu bars and menus */
        QMenuBar {
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        QMenuBar::item:selected {
            background: #4facfe;
        }
        
        QMenu {
            background: #2b2b2b;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        QMenu::item:selected {
            background: #4facfe;
        }
        
        /* Existing styles */
        QLabel {
            color: white;
            font-size: 16px;
            font-weight: 500;
            background: transparent;
            border: none;
        }
        
        QLabel#title {
            font-size: 32px;
            font-weight: bold;
            margin: 20px 0;
        }
        
        QLabel#subtitle {
            font-size: 18px;
            font-weight: 300;
            margin: 10px 0;
            color: rgba(255, 255, 255, 0.9);
        }
    
        
        QScrollArea {
            background: transparent;
            border: none;
        }
        
        QProgressBar {
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            text-align: center;
            color: white;
            font-weight: bold;
            min-height: 25px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #4facfe, stop:1 #00f2fe);
            border-radius: 8px;
        }
        
        /* Tooltip styling for parameter descriptions */
        QToolTip {
            background: rgba(40, 40, 40, 0.95);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        """


def get_analysis_page_style() -> str:
    """Minimal dark theme that only overrides system theme without affecting custom component styling."""
    return """
        /* Only style the main analysis page background */
        AnalysisPageView {
            background: black;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        QMessageBox QPushButton {
            min-width: 80px;
            min-height: 30px;
            margin: 5px;
        }

        QPushButton {
            border: none;
        }
    
        QWidget {
            background: transparent;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Menu styling */
        QMenuBar {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: none;
        }
        
        QMenuBar::item {
            background: transparent;
            color: white;
            padding: 5px 10px;
        }
        
        QMenuBar::item:selected {
            background: #4facfe;
            border-radius: 3px;
        }
        
        QMenu {
            background: #2b2b2b;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        
        QMenu::item {
            padding: 5px 20px;
        }
        
        QMenu::item:selected {
            background: #4facfe;
        }
        
        /* Basic scroll area styling */
        QScrollArea {
            background: transparent;
            border: none;
        }
        
        /* Stacked widget styling */
        QStackedWidget {
            background: transparent;
        }
        
        /* Tooltip styling for parameter descriptions */
        QToolTip {
            background: rgba(40, 40, 40, 0.95);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        """
