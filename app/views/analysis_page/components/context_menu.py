"""Generic context menu components for interactive analysis widgets.

Provides configurable context menus with flexible action handling
for various analysis page components.

Author: Hugo Demule
Date: January 2026
"""

from typing import Callable, List, Optional

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import QMenu, QWidget


class ContextMenu(QObject):
    """Configurable context menu for analysis widgets.

    Provides a flexible context menu system that can be configured with
    custom button labels and corresponding signals or callbacks.

    Example:
        >>> menu = ContextMenu(
        ...     ["Add Transition", "Remove All"],
        ...     [add_signal, remove_signal],
        ...     parent_widget
        ... )
        >>> menu.show_menu()
    """

    def __init__(
        self,
        button_names: List[str],
        signals_or_callbacks: List[Callable],
        parent: QWidget = None,
        cleanup_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the generic context menu.

        Args:
            button_names (List[str]): List of button labels to display in the menu.
            signals_or_callbacks (List[Callable]): List of signals or callable functions to
                                                  trigger when corresponding buttons are clicked.
            parent (QWidget, optional): Parent widget for the context menu.
            cleanup_callbacks (List[Callable], optional): List of callbacks to execute
                                                         when menu is about to hide.
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.button_names = button_names
        self.signals_or_callbacks = signals_or_callbacks
        self.cleanup_callbacks = cleanup_callbacks or []

        # Validate input
        if len(button_names) != len(signals_or_callbacks):
            raise ValueError(
                "button_names and signals_or_callbacks must have the same length"
            )

    def show_menu(self, button_args: List[List] = None):
        """
        Show the context menu at cursor position.

        Args:
            button_args (List[List], optional): List of argument lists for each button.
                                              Each inner list contains the arguments to pass
                                              to the corresponding button's callback.
                                              Empty list [] means no arguments.
        """
        if button_args is None:
            button_args = [[] for _ in self.button_names]

        if len(button_args) != len(self.button_names):
            raise ValueError("button_args must have the same length as button_names")

        context_menu = QMenu(self.parent_widget)

        # Create actions for each button
        for name, signal_or_callback, args in zip(
            self.button_names, self.signals_or_callbacks, button_args
        ):
            action = QAction(name, context_menu)

            # Connect action to signal or callback with specific arguments for this button
            if hasattr(signal_or_callback, "emit"):
                # It's a pyqtSignal
                action.triggered.connect(
                    lambda checked=False,
                    sig=signal_or_callback,
                    button_args=args: sig.emit(*button_args)
                )
            elif callable(signal_or_callback):
                # It's a callable function
                action.triggered.connect(
                    lambda checked=False,
                    func=signal_or_callback,
                    button_args=args: func(*button_args)
                )
            else:
                raise ValueError(
                    f"Item {signal_or_callback} is neither a signal nor a callable"
                )

            context_menu.addAction(action)

        # Connect cleanup callbacks
        for cleanup_callback in self.cleanup_callbacks:
            context_menu.aboutToHide.connect(cleanup_callback)

        # Show menu at cursor position
        global_pos = QCursor.pos()
        context_menu.exec(global_pos)


class ConditionalContextMenu(ContextMenu):
    """
    Extended context menu that can show different buttons based on conditions.

    This class allows for dynamic menu creation where different sets of buttons
    can be shown based on the context or state.
    """

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the conditional context menu.

        Args:
            parent (QWidget, optional): Parent widget for the context menu.
        """
        # Initialize with empty lists - will be populated dynamically
        if parent is None:
            super().__init__([], [])
        else:
            super().__init__([], [], parent)
        self.menu_configs = {}

    def add_menu_config(
        self,
        condition_key: str,
        button_names: List[str],
        signals_or_callbacks: List[Callable],
        cleanup_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Add a menu configuration for a specific condition.

        Args:
            condition_key (str): Key to identify this menu configuration.
            button_names (List[str]): List of button labels for this configuration.
            signals_or_callbacks (List[Callable]): List of signals or callbacks for this configuration.
            cleanup_callbacks (List[Callable], optional): List of cleanup callbacks for this configuration.
        """
        self.menu_configs[condition_key] = {
            "button_names": button_names,
            "signals_or_callbacks": signals_or_callbacks,
            "cleanup_callbacks": cleanup_callbacks or [],
        }

    def show_conditional_menu(self, condition_key: str, button_args: List[List]):
        """
        Show menu based on the specified condition.

        Args:
            condition_key (str): Key identifying which menu configuration to use.
            button_args (List[List]): List of argument lists for each button.
                                    Each inner list contains the arguments to pass
                                    to the corresponding button's callback.
                                    Empty list [] means no arguments.

        Example:
            # For "transition_specific" with 2 buttons:
            # Button 1 gets [transition_pos], Button 2 gets [] (no args)
            self.context_menu.show_conditional_menu("transition_specific", [[transition_pos], []])
        """
        if condition_key not in self.menu_configs:
            raise ValueError(f"Menu configuration '{condition_key}' not found")

        config = self.menu_configs[condition_key]

        # Temporarily set the configuration
        self.button_names = config["button_names"]
        self.signals_or_callbacks = config["signals_or_callbacks"]
        self.cleanup_callbacks = config["cleanup_callbacks"]

        # Show the menu with specific arguments for each button
        self.show_menu(button_args)
