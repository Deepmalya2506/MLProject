# exception.py
# This file is responsible for handling custom exceptions and formatting error messages.

import sys
import traceback


def error_message_detail(error: Exception, error_detail) -> str:
    """
    Returns a detailed error message including:
    - filename
    - line number
    - original error message
    """
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in Python script: [{filename}] "
        f"at line number: [{lineno}] "
        f"with error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class that wraps the original exception
    and provides detailed error information.
    """

    def __init__(self, error: Exception, error_detail):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
