"""Validate & normalise JSON Genetic Code definitions."""

from datetime import datetime
from hashlib import sha256
from json import load
from os.path import dirname, join
from pprint import pformat
from cerberus import Validator
from .gc_type import validate, INVALID_VALUE


with open(join(dirname(__file__), "formats/entry_format.json"), "r") as file_ptr:
    _ENTRY_SCHEMA = schema = load(file_ptr)


def define_signature(gc):
    """Define the signature of a genetic code.

    The signature for a codon GC is slightly different to a regular GC.

    Args
    ----
    gc(dict): Must at least be an mCodon.

    Returns
    -------
    (str): Lowercase hex SHA256 string.
    """
    # NOTE: This needs to be very specific and stand the test of time!
    gca_hex = '0' * 64 if gc['gca'] is None else gc['gca']
    gcb_hex = '0' * 64 if gc['gcb'] is None else gc['gcb']
    string = pformat(gc['graph'], indent=0, sort_dicts=True, width=65535, compact=True) + gca_hex + gcb_hex

    # If it is a codon glue on the mandatory definition
    if "generation" in gc and gc["generation"] == 0:
        if "meta_data" in gc and "function" in gc["meta_data"]:
            string += gc["meta_data"]["function"]["python3"]["0"]["inline"]
            if 'code' in gc["meta_data"]["function"]["python3"]["0"]:
                string += gc["meta_data"]["function"]["python3"]["0"]["code"]
    return sha256(string.encode()).hexdigest()


class _entry_validator(Validator):

    # TODO: Make errors ValidationError types for full disclosure
    # https://docs.python-cerberus.org/en/stable/customize.html#validator-error
    def _check_with_valid_alpha_class(self, field, value):

        if not value:
            # Valid codon
            if "beta_class" in self.document and self.document['beta_class']:
                self._error(
                    field, "If alpha_class == 0, beta_class must == 0.")
            if "codon_depth" in self.document and self.document['codon_depth'] != 1:
                self._error(
                    field, "If alpha_class == 0, codon_depth must == 1.")
            if "code_depth" in self.document and self.document['code_depth'] != 1:
                self._error(
                    field, "If alpha_class == 0, code_depth must == 1.")
            if "gca" in self.document and not self.document['gca'] is None:
                self._error(
                    field, "If alpha_class == 0, gca must == None")
            if "gcb" in self.document and not self.document['gcb'] is None:
                self._error(
                    field, "If alpha_class == 0, gcb must == None")
            if "num_codes" in self.document and self.document['num_codes'] != 1:
                self._error(field, "If alpha_class == 0, num_codes must == 1.")
            if "num_unique_codes" in self.document and self.document['num_unique_codes'] != 1:
                self._error(
                    field, "If alpha_class == 0, num_unique_codes must == 1.")
            if "raw_num_codons" in self.document and self.document['raw_num_codons'] != 1:
                self._error(
                    field, "If alpha_class == 0, raw_num_codons must == 1.")
            if "opt_num_codons" in self.document and self.document['opt_num_codons'] != 1:
                self._error(
                    field, "If alpha_class == 0, opt_num_codons must == 1.")
            if "meta_data" in self.document and "parents" in self.document['meta_data']:
                self._error(
                    field, "If alpha_class == 0 then there are no parents.")
            if "meta_data" in self.document and not "function" in self.document['meta_data']:
                self._error(
                    field, "If alpha_class == 0 then there must be a 'function' definition.")
            elif not "python3" in self.document['meta_data']['function']:
                self._error(
                    field, "If alpha_class == 0 then there must be a 'python3' definition.")
        else:
            # Valid non-codon
            if "beta_class" in self.document and not self.document['beta_class']:
                self._error(
                    field, "If alpha_class != 0, beta_class must != 0.")
            if "meta_data" in self.document and not "parents" in self.document['meta_data']:
                self._error(
                    field, "If alpha_class != 0 then there must be at least one parent.")


    def _check_with_valid_created(self, field, value):
        try:
            date_time_obj = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            self._error(
                field, "Created date-time is not valid. Unknown error parsing.")
            return

        if date_time_obj > datetime.utcnow():
            self._error(field, "Created date-time cannot be in the future.")


    def _check_with_valid_type(self, field, value):
        if not validate(value) and value != INVALID_VALUE:
            self._error(field, 'Does not exist')


    def _check_with_valid_inline(self, field, value):
        # TODO: Check right number of return parameters and arguments
        pass


    def _check_with_valid_callable(self, field, value):
        # TODO: Check right number of return parameters and arguments. Check arguments all have default=None.
        pass


    def _normalize_default_setter_set_signature(self, document):
        return define_signature(self.document)


    def _normalize_default_setter_set_input_types(self, document):
        # Sort the input endpoints by index then return the types as a list
        inputs = []
        for row in document["graph"].values():
            inputs.extend((ep for ep in filter(lambda x: x[0] == 'I', row)))
            inputs.sort(key=lambda x: x[1])
        return [x[2] for x in inputs]


    def _normalize_default_setter_set_output_types(self, document):
        return [ep[2] for ep in sorted(document["graph"].get("O", tuple()), key=lambda x:x[1])]


    def _normalize_default_setter_set_num_inputs(self, document):
        return len(document["graph"].get("I", tuple()))


    def _normalize_default_setter_set_num_outputs(self, document):
        return len(document["graph"].get("O", tuple()))


    def _normalize_default_setter_set_opt_num_codons(self, document):
        return 1 if document['gca'] is None and document['gcb'] is None else 0


    def _normalize_default_setter_set_created(self, document):
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


entry_validator = _entry_validator(_ENTRY_SCHEMA)


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