"""Extension to the Cerberus Validator with common checks."""

from json import JSONDecodeError, load
from logging import NullHandler, getLogger
from os import R_OK, W_OK, X_OK, access
from os.path import isdir, isfile
from pprint import pformat

from cerberus import Validator
from cerberus.errors import UNKNOWN_FIELD

_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


class base_validator(Validator):
    """Additional format checks."""

    def error_str(self):
        """Prettier format to a list of errors."""
        return '\n'.join((field + ': ' + pformat(error) for field, error in self.errors.items()))

    def _isdir(self, field, value):
        """Validate value is a valid, existing directory."""
        if not isdir(value):
            self._error(field, "{} is not a valid directory or does not exist.".format(value))
            return False
        return True

    def _isfile(self, field, value):
        """Validate value is a valid, existing file."""
        if not isfile(value):
            self._error(field, "{} is not a valid file or does not exist.".format(value))
            return False
        return True

    def _isreadable(self, field, value):
        """Validate value is a readable file."""
        if not access(value, R_OK):
            self._error(field, "{} is not readable.".format(value))
            return False
        return True

    def _iswriteable(self, field, value):
        """Validate value is a writeable file."""
        if not access(value, W_OK):
            self._error(field, "{} is not writeable.".format(value))
            return False
        return True

    def _isexecutable(self, field, value):
        """Validate value is an executable file."""
        if not access(value, X_OK):
            self._error(field, "{} is not executable.".format(value))
            return False
        return True

    def _isjsonfile(self, field, value):
        """Validate the JSON file is decodable."""
        if self._isfile(field, value) and self._isreadable(field, value):
            with open(value, "r") as file_ptr:
                try:
                    schema = load(file_ptr)
                except JSONDecodeError as ex:
                    self._error(field, "The file is not decodable JSON: {}".format(ex))
                else:
                    return schema
        return None

    def str_errors(self, error):
        """Create an error string."""
        if error.code == UNKNOWN_FIELD.code:
            error.rule == 'unknown field'
        str_tuple = (
            'Value: ' + str(error.value),
            'Rule: ' + str(error.rule),
            'Constraint: ' + str(error.constraint)
        )
        return ', '.join(str_tuple)
