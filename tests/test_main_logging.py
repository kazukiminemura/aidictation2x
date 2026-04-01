import logging

from main import setup_logging


def test_setup_logging_suppresses_noisy_display_loggers() -> None:
    setup_logging("INFO")

    assert logging.getLogger("screen_brightness_control").level == logging.ERROR
    assert logging.getLogger("screen_brightness_control.windows").level == logging.ERROR
    assert logging.getLogger("screen_brightness_control.windows.VCP").level == logging.ERROR
    assert logging.getLogger("x_wmi").level == logging.ERROR
