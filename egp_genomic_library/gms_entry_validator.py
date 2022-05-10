"""Validate & normalise JSON Genetic Code definitions."""

from datetime import datetime
from json import load
from os.path import dirname, join
# TODO: Make utils its own python package
from .utils.base_validator import base_validator
from egp_physics.ep_type import validate
from egp_physics.gc_graph import gc_graph
from egp_physics.gc_type import PROPERTIES


with open(join(dirname(__file__), "formats/gms_entry_format.json"), "r") as file_ptr:
    GMS_ENTRY_SCHEMA = load(file_ptr)


# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def merge(a, b, path=None):
    """Merge dict b into a recursively. a is modified.

    This function is equivilent to a.update(b) if b contains no dictionary values with
    the same key as in a.

    If there are dictionaries
    in b that have the same key as a then those dictionaries are merged in the same way.
    Keys in a & b (or common key'd sub-dictionaries) where one is a dict and the other
    some other type raise an exception.

    Args
    ----
    a (dict): Dictionary to merge in to.
    b (dict): Dictionary to merge.

    Returns
    -------
    a (modified)
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


class _gms_entry_validator(base_validator):

    # TODO: Make errors ValidationError types for full disclosure
    # https://docs.python-cerberus.org/en/stable/customize.html#validator-error

    def _check_with_valid__e_count(self, field, value):
        if not value and self.document['_evolvability'] > 0.0:
            self._error(field, f'_e_count cannot be 0 if _evolvability is non-zero.')
        if value > self.document['e_count']:
            self._error(field, f"_e_count ({value}) cannot be greater than e_count ({self.document['e_count']})")

    def _check_with_valid__evolvability(self, field, value):
        if value > 0.0 and not self.document['_e_count']:
            self._error(field, f'_e_count cannot be 0 if _evolvability is non-zero.')

    def _valid_pgc(self, field, value):
        pgc_none = {k:v is None for k, v in self.document.items() if 'pgc_' in k and k != 'pgc_ref'}
        if (value is None and not all(pgc_none.values())) or (value is not None and any(pgc_none.values())):
            pgc_defined = [k for k, v in pgc_none.items() if not v]
            pgc_undefined = [k for k, v in pgc_none.items() if v]
            self._error(field, f'pGC fields only partially defined. Defined: {pgc_defined}, Undefined: {pgc_undefined}.')
            return False
        return value is not None

    def _check_with_valid__pgc_e_count(self, field, value):
        if self._valid_pgc(field, value):
            e = self.document['_pgc_evolvability']
            invalid = {i: v == 0 and e[i] > 0.0 for i, v in enumerate(value)}
            if any(invalid.values()):
                indices = [i for i, v in invalid.items() if not v]
                self._error(field, f'_pgc_e_count cannot be 0 if _pgc_evolvability is non-zero at indices {indices}.')

    def _check_with_valid__pgc_evolvability(self, field, value):
        self._valid_pgc(field, value)

    def _check_with_valid__pgc_f_count(self, field, value):
        if self._valid_pgc(field, value):
            invalid = {i: v == 0 and self.document['_pgc_fitness'][i] > 0.0 for i, v in enumerate(value)}
            if any(invalid.values()):
                indices = [i for i, v in invalid.items() if not v]
                self._error(field, f'_pgc_f_count cannot be 0 if _pgc_fitness is non-zero at indices {indices}.')

    def _check_with_valid__pgc_fitness(self, field, value):
        self._valid_pgc(field, value)

    def _check_with_valid__reference_count(self, field, value):
        if value > self.document['reference_count']:
            self._error(f"_reference_count ({value}) cannot be higher than reference_count {self.document['reference_count']}.")

    def _check_with_valid_created(self, field, value):
        if value > datetime.utcnow():
            self._error(field, "Created date-time cannot be in the future. Is the system clock correct?")

    def _check_with_valid_e_count(self, field, value):
        if value == 1 and self.document['evolvability'] < 1.0:
            self._error(field, f'e_count cannot be 1 if evolvability has changed (is not 1.0).')
        if value < self.document['_e_count']:
            self._error(field, f"e_count ({value}) cannot be less than _e_count ({self.document['_e_count']})")

    def _check_with_valid_evolvability(self, field, value):
        if value < 1.0 and self.document['e_count'] == 1:
            self._error(field, f'e_count cannot be 1 if evolvability has changed (is not 1.0).')

    def _check_with_valid_graph(self, field, value):
        graph = gc_graph(value)
        if not graph.validate():
            self._error(field, f'graph is invalid: {graph.status}')

    def _check_with_valid_type(self, field, value):
        if not validate(value):
            self._error(field, f'ep_type {value} does not exist.')

    def _check_with_valid_input_types(self, field, value):
        all_types = set(range(len(value)))
        all_refs = set((idx for idx in self.document['inputs']))
        if all_types != all_refs:
            self._error(field, f'Input types at indices {all_types - all_refs} are not referenced by inputs.')

    def _check_with_valid_inputs(self, field, value):
        num_types = len(self.document['input_types'])
        invalid_indices = [idx for idx in value if idx > num_types]
        if invalid_indices:
            self._error(field, f'Invalid inputs indices: {invalid_indices}')

    def _check_with_valid_num_inputs(self, field, value):
        if value != len(self.document['inputs']):
            self._error(field, f"num_inputs ({value}) != length of inputs ({len(self.document['inputs'])}.")

    def _check_with_valid_num_outputs(self, field, value):
        if value != len(self.document['outputs']):
            self._error(field, f"num_outputs ({value}) != length of outputs ({len(self.document['outputs'])}.")

    def _check_with_valid_output_types(self, field, value):
        all_types = set(range(len(value)))
        all_refs = set((idx for idx in self.document['outputs']))
        if all_types != all_refs:
            self._error(field, f'Output types at indices {all_types - all_refs} are not referenced by outputs.')

    def _check_with_valid_outputs(self, field, value):
        num_types = len(self.document['output_types'])
        invalid_indices = [idx for idx in value if idx > num_types]
        if invalid_indices:
            self._error(field, f'Invalid outputs indices: {invalid_indices}')

    def _check_with_valid_pgc_e_count(self, field, value):
        if self._valid_pgc(field, value):
            invalid = {i: v == 1 and self.document['pgc_evolvability'][i] < 1.0 for i, v in enumerate(value)}
            if any(invalid.values()):
                indices = [i for i, v in invalid.items() if not v]
                self._error(field, f'pgc_e_count cannot be 1 if pgc_evolvability has changed (is not 1.0) at indices {indices}.')

            invalid = [idx for idx, pgc_e_count in enumerate(value) if pgc_e_count < self.document['_pgc_e_count'][idx]]
            if invalid:
                self._error(field, f'_pgc_e_count is greater than pgc_e_count at indices: {invalid}')

    def _check_with_valid_pgc_f_count(self, field, value):
        if self._valid_pgc(field, value):
            invalid = {i: v == 1 and self.document['pgc_fitness'][i] < 1.0 for i, v in enumerate(value)}
            if any(invalid.values()):
                indices = [i for i, v in invalid.items() if not v]
                self._error(field, f'pgc_f_count {list(invalid.values())} cannot be 1 if pgc_fitness has changed (is not 1.0) at indices {indices}.')

            invalid = [idx for idx, pgc_f_count in enumerate(value) if pgc_f_count < self.document['_pgc_f_count'][idx]]
            if invalid:
                self._error(field, f'_pgc_f_count is greater than pgc_f_count at indices: {invalid}')

    def _check_with_valid_properties(self, field, value):
        valid_property_mask = 0
        for valid_property in PROPERTIES.values():
            valid_property_mask |= valid_property
        invalid_properties = (valid_property_mask & value) ^ value
        if invalid_properties:
            self._error(field, f'Invalid properties set in the positions: {hex(invalid_properties)}.')

    def _check_with_valid_reference_count(self, field, value):
        if value < self.document['_reference_count']:
            self._error(f"reference_count ({value}) cannot be lower than _reference_count {self.document['_reference_count']}.")

    def _check_with_valid_updated(self, field, value):
        if value > datetime.utcnow():
            self._error(field, "Updated date-time cannot be in the future. Is the system clock correct?")

    def _normalize_default_setter_set_input_types(self, document):
        # Gather all the input endpoint types. Reduce in a set then order the list.
        inputs = []
        for row in document["graph"].values():
            inputs.extend([ep[2] for ep in filter(lambda x: x[0] == 'I', row)])
        return sorted(set(inputs))

    def _normalize_default_setter_set_output_types(self, document):
        # Gather all the output endpoint types. Reduce in a set then order the list.
        return sorted(set([ep[2] for ep in document["graph"].get("O", tuple())]))

    def _normalize_default_setter_set_input_indices(self, document):
        # Get the type list then find all the inputs in order & look them up.
        type_list = self._normalize_default_setter_set_input_types(document)
        inputs = []
        for row in document["graph"].values():
            inputs.extend((ep for ep in filter(lambda x: x[0] == 'I', row)))
        return bytes((type_list.index(ep[2]) for ep in sorted(inputs, key=lambda x:x[1])))

    def _normalize_default_setter_set_output_indices(self, document):
        # Get the type list then find all the inputs in order & look them up.
        type_list = self._normalize_default_setter_set_output_types(document)
        bytea = (type_list.index(ep[2]) for ep in sorted(document["graph"].get("O", tuple()), key=lambda x:x[1]))
        return bytes(bytea)

    def _normalize_default_setter_set_num_inputs(self, document):
        inputs = set()
        for row in document["graph"].values():
            for ep in filter(lambda x:x[0] == 'I', row):
                inputs.add(ep[1])
        return len(inputs)

    def _normalize_default_setter_set_num_outputs(self, document):
        return len(document["graph"].get("O", tuple()))

    def _normalize_default_setter_set_updated(self, document):
        return datetime.utcnow()

    def _normalize_coerce_memoryview_to_bytes(self, value):
        return bytes(value)


gms_entry_validator = _gms_entry_validator(GMS_ENTRY_SCHEMA)
