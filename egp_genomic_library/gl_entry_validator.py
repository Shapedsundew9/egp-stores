"""Validate & normalise JSON Genetic Code definitions."""

from copy import deepcopy
from json import load
from os.path import dirname, join
from uuid import UUID
from hashlib import sha256
from pprint import pformat
from cerberus import TypeDefinition
from .gms_entry_validator import _gms_entry_validator, GMS_ENTRY_SCHEMA, merge


GL_ENTRY_SCHEMA = deepcopy(GMS_ENTRY_SCHEMA)
with open(join(dirname(__file__), "formats/gl_entry_format.json"), "r") as file_ptr:
    merge(GL_ENTRY_SCHEMA, load(file_ptr))


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
    return sha256(string.encode()).digest()


class _gl_entry_validator(_gms_entry_validator):

    types_mapping = _gms_entry_validator.types_mapping.copy()
    types_mapping['uuid'] = TypeDefinition('uuid', (UUID,), ())

    # TODO: Make errors ValidationError types for full disclosure
    # https://docs.python-cerberus.org/en/stable/customize.html#validator-error

    def _check_with_valid_ancestor_a(self, field, value):
        if value is None and self.document['generation']:
            self._error(field, f'GC has no primary parent (ancestor A) but is not a codon (0th generation).')
        if value is not None and not self.document['generation']:
            self._error(field, f'GC has a primary parent (ancestor A) but is a codon (0th generation).')

    def _check_with_valid_ancestor_b(self, field, value):
        if value is not None and self.document['ancestor_a'] is None:
            self._error(field, f'GC has a secondary parent (ancestor B) but no primary parent (ancestor A).')

    def _check_with_valid_gca(self, field, value):
        if 'A' in self.document['graph'] and value is None:
            self._error(field, f'graph references row A but gca is None.')

    def _check_with_valid_gcb(self, field, value):
        if 'B' in self.document['graph'] and value is None:
            self._error(field, f'graph references row B but gcb is None.')
        if value is not None and self.document['gca'] is None:
            self._error(field, f'gcb is defined but gca is None.')

    def _check_with_valid_pgc(self, field, value):
        if self.document['generation'] and value is None:
            self._error(field, f'Generation is > 0 but pgc is None.')
        if not self.document['generation'] and value is not None:
            self._error(field, f'Generation is 0 but pgc is defined as {value}.')
        if self.document['ancestor_a'] is None and value is not None:
            self._error(field, f'GC has no primary parent (ancestor A) but pgc is defined as {value}.')

    def _normalize_default_setter_set_signature(self, document):
        return define_signature(self.document)


gl_entry_validator = _gl_entry_validator(GL_ENTRY_SCHEMA)
