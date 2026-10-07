"""
Define custom exceptions for errors that occur within the project.
Provide meaningful error messages while preserving the original cause.
Allow calling code to identify and handle specific project failures.
"""

import sys

from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occured in script: [{file_name}] at line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


"""
This code wraps an existing error in a custom exception with a detailed message.

- error_message_detail(error, error_detail) receives the original error and the sys module.
- Passing sys as the second argument makes error_detail refer to sys inside the function.
- ': sys' does not pass sys automatically; remove it because sys is a module, not a type.
- exc_info() returns the current exception's type, object, and traceback.
- _, _, exc_tb keeps the traceback; the underscores mark values we do not need.
- tb_frame.f_code.co_filename gets the filename; tb_lineno gets the line number.
- These describe the first traceback entry, which may be a calling line for nested errors.
- The f-string combines the filename, line number, and original error into readable text.

- CustomException(Exception) creates an exception class based on Python's Exception.
- __init__ runs when CustomException(e, sys) creates an exception object.
- self refers to that object; super().__init__ initializes its parent Exception.
- self.error_message stores the detailed message returned by the helper function.
- __str__ returns that message when the exception is printed or converted to a string.

Use inside an except block: raise CustomException(e, sys) from e.
An active exception is needed here; otherwise exc_info() provides no traceback.
"""