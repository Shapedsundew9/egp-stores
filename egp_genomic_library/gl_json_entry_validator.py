"""Validate & normalise JSON Genetic Code definitions."""

from datetime import datetime
from json import load
from os.path import dirname, join
from copy import deepcopy
from .gl_entry_validator import GL_ENTRY_SCHEMA, _gl_entry_validator, merge
from .conversions import str_to_datetime, str_to_sha256, str_to_UUID, encode_properties


GL_JSON_ENTRY_SCHEMA = deepcopy(GL_ENTRY_SCHEMA)
with open(join(dirname(__file__), "formats/gl_json_entry_format.json"), "r") as file_ptr:
    merge(GL_JSON_ENTRY_SCHEMA, load(file_ptr))


class _gl_json_entry_validator(_gl_entry_validator):

    # TODO: Make errors ValidationError types for full disclosure
    # https://docs.python-cerberus.org/en/stable/customize.html#validator-error


    def _normalize_coerce_signature_str_to_binary(self, value):
        return str_to_sha256(value)

    def _normalize_coerce_datetime_str_to_datetime(self, value):
        return str_to_datetime(value)

    def _normalize_coerce_UUID_str_to_UUID(self, value):
        return str_to_UUID(value)

    def _normalize_coerce_properties_dict_to_int(self, value):
        return encode_properties(value)


entry_validator = _gl_json_entry_validator(GL_JSON_ENTRY_SCHEMA)


"""
Some benchmarking on SHA256 generation
======================================
Python 3.8.5

>>> def a():
...     start = time()
...     for _ in range(10000000): int(sha256("".join(string.split()).encode()).hexdigest(), 16)
...     print(time() - start)
...
>>> a()
8.618626356124878
>>> def b():
...     start = time()
...     for _ in range(10000000): int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big')
...     print(time() - start)
...
>>> b()
7.211490631103516
>>> def c():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).hexdigest()
...     print(time() - start)
...
>>> c()
6.463267803192139
>>> def d():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).digest()
...     print(time() - start)
...
>>> d()
6.043259143829346
>>> def e():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).digest(): "Test"}
...     print(time() - start)
...
>>> e()
6.640311002731323
>>> def f():
...     start = time()
...     for _ in range(10000000): {int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big'): "Test"}
...     print(time() - start)
...
>>> f()
7.6320412158966064
>>> def g():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).hexdigest(): "Test"}
...     print(time() - start)
...
>>> g()
7.144319295883179
>>> def h1():
...     start = time()
...     for _ in range(10000000): getrandbits(256)
...     print(time() - start)
...
>>> h1()
1.0232288837432861
>>> def h2():
...     start = time()
...     for _ in range(10000000): getrandbits(128)
...     print(time() - start)
...
>>> h2()
0.8551476001739502
>>> def h3():
...     start = time()
...     for _ in range(10000000): getrandbits(64)
...     print(time() - start)
...
>>> h3()
0.764052152633667
>>> def i():
...     start = time()
...     for _ in range(10000000): getrandbits(256).to_bytes(32, 'big')
...     print(time() - start)
...
>>> i()
2.038336753845215
"""


"""
Some Benchmarking on hashing SHA256
===================================
Python 3.8.5

>>> a =tuple( (getrandbits(256).to_bytes(32, 'big') for _ in range(10000000)))
>>> b =tuple( (int(getrandbits(63)) for _ in range(10000000)))
>>> start = time(); c=set(a); print(time() - start)
1.8097834587097168
>>> start = time(); d=set(b); print(time() - start)
1.0908379554748535
"""
