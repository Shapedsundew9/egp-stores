"""Tokenisation of user strings.

This module is used to map tokens to user text.
"""


from copy import deepcopy
from logging import getLogger


""" Valid token code prefixes
    'D': Debug
    'I': Information
    'W': Warning
    'E': Error
    'F': Fatal
    'X': Unknown
"""
_CODE_PREFIXES = ('D', 'I', 'W', 'E', 'F', 'X')


"""The token library maps token codes to formatted strings.

The code:fmt_str pairs are added using register_token_code()
When a token with a code in the token_library is converted to a string
The fmt_str is looked up and formatted with the token parameters.
"""
token_library = {
    'E00000': 'Unknown error code {token} with parameters {parameters}.' 
}


def _valid_code(code):
    """Sanity of the token code.

    Args
    ----
        code(str): The code to be validated.

    Returns
    -------
        (bool): True if the code is valid else False
    """
    if not code[0] in _CODE_PREFIXES: return False
    if len(code) != 6: return False
    code_num = int(code[1:])
    if code_num < 0 or code_num > 99999: return False
    if code_num in token_library: return False
    return True


def register_token_code(code, fmt_str):
    """Register a token code and text in the token_library.

    The registered code can then be used to generate a human readable
    string when the _str_() function is called on a token with that code.

    Args
    ----
        code (str): Format "E<i><i><i><i><i>" where <i> is a digit 0 to 9. Every code is unique.
        fmt_str (str): A human readable string for the code with optional formatting parameters.

    Returns
    -------
        (bool): True if the token is valid else False
    """
    assert _valid_code(code)
    assert code not in token_library
    token_library[code] = fmt_str


class text_token():
    """Maintains a relationship between a code and a human readable string.

    A token has the structure:
    { '<code>', {<parameters>} }
    where <code> is the same as in register_token_code() and <parameters> is a
    set of key:value pairs defining the values of the formatting keys in the
    fmt_str for the code in the token_library.
    """

    _logger = getLogger(__name__)


    def __init__(self, token):
        self.code = tuple(token.keys())[0]
        self.parameters = token[self.code]


    def __str__(self):
        """Convert the token to a human readbale string.
        
        This can be recursive if a parameter is of type text_token.
        """
        if self.code not in token_library: return token_library['E00000'].format(vars(self))
        # text_token._logger.debug("Code {}: Parameters: {} Library string: {}".format(self.code, self.parameters, token_library[self.code]))
        return self.code + ": " + token_library[self.code].format(**self.parameters)
