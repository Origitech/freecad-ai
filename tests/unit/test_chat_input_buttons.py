"""The chat panel's Send button must never be squeezed below its own label."""

import os

import pytest

# Must be set before the first QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
except ImportError:
    try:
        from PySide2 import QtWidgets  # noqa: F401
    except ImportError:
        pytest.skip("PySide6/PySide2 not available", allow_module_level=True)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture(params=["", "QPushButton { padding: 6px 12px; border: 2px solid gray; }"],
                ids=["plain", "padded-theme"])
def dock(request, qapp, tmp_config_dir):
    # A theme that pads its buttons is what made the old 30 px minimum fall
    # below the button's own size hint.
    previous = qapp.styleSheet()
    qapp.setStyleSheet(request.param)
    from freecad_ai.ui.chat_widget import ChatDockWidget
    widget = ChatDockWidget()
    yield widget
    widget.deleteLater()
    qapp.setStyleSheet(previous)
    qapp.processEvents()


def test_send_never_goes_below_its_size_hint(dock):
    # A minimum under the hint is what let a short dock squeeze Send alone -
    # its neighbours keep their hints - cutting the label in half.
    send = dock.send_btn
    assert send.minimumHeight() >= send.sizeHint().height()


def test_send_is_taller_than_its_neighbours(dock):
    assert dock.send_btn.minimumHeight() >= 40
    assert dock.send_btn.minimumHeight() > dock._attach_btn.sizeHint().height()
