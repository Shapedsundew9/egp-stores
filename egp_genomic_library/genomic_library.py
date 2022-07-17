"""The Genomic Library class wraps the database_table."""

from copy import deepcopy
from json import load
from logging import NullHandler, getLogger
from os.path import dirname, join
from pprint import pformat
from functools import partial

from pypgtable import table

from .gl_json_entry_validator import entry_validator, merge
from .genetic_material_store import genetic_material_store, GMS_TABLE_SCHEMA
from .utils.text_token import register_token_code, text_token
from .conversions import *


_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# No longer used but still here just in case of future need.
_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 0.0::REAL, 0::INTEGER)')
_WEIGHTED_FIXED_UPDATE_RAW = ('weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})')
_FIXED_UPDATE_RAW = 'fixed_array_update({CSCC}, {PSCC}, {CSPC})'
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_WEIGHTED_SCALAR_UPDATE = '({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / ' + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_evolvability}',
    'CSCC': 'EXCLUDED.{pgc_e_count}',
    'PSCV': '"__table__".{pgc_evolvability}',
    'PSCC': '"__table__".{pgc_e_count}',
    'CSPV': 'EXCLUDED.{_pgc_evolvability}',
    'CSPC': 'EXCLUDED.{_pgc_e_count}'
}
_PGC_EVO_UPDATE_STR = '{pgc_evolvability} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR = '{pgc_e_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_fitness}',
    'CSCC': 'EXCLUDED.{pgc_f_count}',
    'PSCV': '"__table__".{pgc_fitness}',
    'PSCC': '"__table__".{pgc_f_count}',
    'CSPV': 'EXCLUDED.{_pgc_fitness}',
    'CSPC': 'EXCLUDED.{_pgc_f_count}'
}
_PGC_FIT_UPDATE_STR = '{pgc_fitness} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR = '{pgc_f_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR = '{evolvability} = ' + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR = '{e_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR = ','.join((
    '{updated} = NOW()',
    _PGC_EVO_UPDATE_STR,
    _PGC_E_COUNT_UPDATE_STR,
    _PGC_FIT_UPDATE_STR,
    _PGC_F_COUNT_UPDATE_STR,
    _EVO_UPDATE_STR,
    _EVO_COUNT_UPDATE_STR,
    _REF_UPDATE_STR))
_NULL_GC_DATA = {
    'code_depth': 0,
    'num_codes': 0,
    'raw_num_codons': 1,
    'generation': 0,
    'properties': 0,
    '_stored': True
}
_DATA_FILE_FOLDER = join(dirname(__file__), 'data')
_DATA_FILES = ['codons.json', 'mutations.json']


# Tree structure
_LEL = 'gca'
_REL = 'gcb'
_NL = 'signature'
_PTR_MAP = {
    _LEL: _NL,
    _REL: _NL
}


register_token_code("I03000", "Adding data to table {table} from {file}.")
register_token_code('E03000', 'Query is not valid: {errors}: {query}')
register_token_code('E03001', 'Entry is not valid: {errors}: {entry}')
register_token_code('E03002', 'Referenced GC(s) {references} do not exist. Entry:\n{entry}:')


_CONVERSIONS = (
    ('graph', compress_json, decompress_json),
    ('meta_data', compress_json, decompress_json),
    ('signature', str_to_sha256, memoryview_to_bytes),
    ('gca', str_to_sha256, memoryview_to_bytes),
    ('gcb', str_to_sha256, memoryview_to_bytes),
    ('ancestor_a', str_to_sha256, memoryview_to_bytes),
    ('ancestor_b', str_to_sha256, memoryview_to_bytes),
    ('pgc', str_to_sha256, memoryview_to_bytes),
    ('inputs', None, memoryview_to_bytes),
    ('outputs', None, memoryview_to_bytes),
    ('creator', str_to_UUID, None),
    ('created', str_to_datetime, None),
    ('updated', str_to_datetime, None),
    ('properties', encode_properties, None)
)


_PTR_MAP = {
    'gca': 'signature',
    'gcb': 'signature'
}

_GL_TABLE_SCHEMA = deepcopy(GMS_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gl_table_format.json"), "r") as file_ptr:
    merge(_GL_TABLE_SCHEMA, load(file_ptr))
HIGHER_LAYER_COLS = tuple((key for key in filter(lambda x: x[0] == '_', _GL_TABLE_SCHEMA)))
UPDATE_RETURNING_COLS = tuple((x[1:] for x in HIGHER_LAYER_COLS)) + ('updated', 'created', 'signature')


# The default config
_DEFAULT_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'genomic_library',
    'ptr_map': _PTR_MAP,
    'schema': _GL_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
    'conversions': _CONVERSIONS
}


def default_config():
    """Return a deepcopy of the default genomic library configuration.

    The copy may be modified and used to create a genomic library instance.

    Returns
    -------
    (dict): The default genomic_library() configuration.
    """
    return deepcopy(_DEFAULT_CONFIG)


def sql_functions():
    """Load the SQL functions used by the genomic library & dependent repositiories of genetic material.

    SQL functions are used to update genetic code entries as they are an easy way to guarantee
    atomic behaviour.
    """
    with open(join(dirname(__file__), 'data/gl_functions.sql'), 'r') as fileptr:
        return fileptr.read()


class genomic_library(genetic_material_store):
    """Store of genetic codes & associated data.

    The genomic_library is responsible for:
        1. Populating calculable entry fields.
        2. Validating entries to be added to the store.
        3. Providing an application interface to the fields.

    The genomic library must be self consistent i.e. no entry can reference a genetic code
    that is not in the genomic library.
    """

    def __init__(self, config=_DEFAULT_CONFIG):
        """Connect to or create a genomic library.

        The genomic library data persists in a postgresql database. Multiple
        instances of the genomic_library() class can connect to the same database
        (use the same configuration).

        Args
        ----
        config(pypgtable config): The config is deep copied by pypgtable.
        """
        super().__init__(node_label=_NL, left_edge_label=_LEL, right_edge_label=_REL)
        self.library = table(config)
        self._update_str = UPDATE_STR.replace('__table__', config['table'])
        self.encode_value = self.library.encode_value
        self.select = self.library.select
        self.recursive_select = self.library.recursive_select
        self.hl_copy = partial(super().hl_copy, field_names=HIGHER_LAYER_COLS)
        if self.library.raw.creator:
            self.library.raw.arbitrary_sql(sql_functions(), read=False)
            for data_file in _DATA_FILES:
                abspath = join(_DATA_FILE_FOLDER, data_file)
                _logger.info(text_token({'I03000': {'table': self.library.raw.config['table'], 'file': abspath}}))
                with open(abspath, "r") as file_ptr:
                    self.library.insert((entry_validator.normalized(entry) for entry in load(file_ptr)))

    def __getitem__(self, signature):
        """Recursively select genetic codes starting with 'signature'.

        Args
        ----
        signature (obj): signature of GC to select.

        Returns
        -------
        (dict(dict)): All genetic codes & codons constructing & including signature gc.
        """
        return self.library[self.library.encode_value('signature', signature)]


    def _check_references(self, references, check_list=set()):
        """Verify all the references exist in the genomic library.

        Genetic codes reference each other. A debugging check is to verify the
        existence of all the references.

        Args
        ----
        references(list): List of genetic code signatures to look up.
        check_list(set): A set of known existing genetic codes signatures.

        Returns
        -------
        Empty list if all references exist else the signatures of missing references.
        """
        naughty_list = []
        for reference in references:
            if self.library[reference] is None and reference not in check_list:
                naughty_list.append(reference)
            else:
                check_list.add(reference)
        return naughty_list

    def _calculate_fields(self, entry, entries=None):
        """Calculate the derived genetic code fields.

        Cerberus normalisation can only set fields based on the contents of the genetic code dictionary.
        However, some fields are derived from GCA & GCB. Entries may be stored in batch and so
        may reference other, as yet to be stored, genetic code dictionaries.

        The entry dictionary is modified.

        Args
        ----
        entry(dict): A genetic code dictionary. That is present in entries.
        entries(dict): A dictionary entry['signature']: entry of genetic code dictionaries to be
            stored or updated in the genomic library.
        """
        # TODO: Need to update references.
        # TODO: Check consistency for GC's that are already stored.
        gca = _NULL_GC_DATA
        if not entry['gca'] is None:
            if entry['gca'] not in entries.keys():
                if not self.library[entry['gca']]:
                    self._logger.error(
                        'entry["gca"] = {} does not exist in the list to be stored or genomic library!'.format(entry['gca']))
                    self._logger.error('Entries signature list: {}'.format(entries.keys()))
            else:
                gca = entries[entry['gca']]
                if not gca['_calculated']:
                    self._calculate_fields(gca, entries)

        gcb = _NULL_GC_DATA
        if not entry['gcb'] is None:
            if entry['gcb'] not in entries.keys():
                if not self.library[entry['gcb']]:
                    self._logger.error(
                        'entry["gcb"] = {} does not exist in the list to be stored or genomic library!'.format(entry['gcb']))
                    self._logger.error('Entries signature list: {}'.format(entries.keys()))
            else:
                gcb = entries[entry['gca']]
                if not gcb['_calculated']:
                    self._calculate_fields(gca, entries)

        if not (entry['gca'] is None and entry['gcb'] is None):
            entry['code_depth'] = max((gca['code_depth'], gcb['code_depth'])) + 1
            entry['num_codes'] = gca['num_codes'] + gcb['num_codes']
            entry['raw_num_codons'] = gca['raw_num_codons'] + gcb['raw_num_codons']
            entry['generation'] = max((gca['generation'] + 1, gcb['generation'] + 1, entry['generation']))
            entry['properties'] = gca['properties'] | gcb['properties']
        entry['_calculated'] = True

    def _normalize(self, entries):
        """Normalize entries before storing. The entries are modified in place.

        Genetic code statistics and meta data are updated / created for storage
        and checked for consistency. This can be a heavy process.

        Args
        ----
        entries(dict): A dictionary entry['signature']: entry of genetic code dictionaries to be
            stored or updated in the genomic library.
        """
        self._logger.debug("Normalizing {} entries.".format(len(entries)))
        for signature, entry in entries.items():
            entries[signature] = entry_validator.normalized(entry)
            entries[signature]['_calculated'] = False
        for entry in entries.values():
            self._calculate_fields(entry, entries)

        self._logger.debug("Validating normalised entries before storing.")
        check_list = set(entries.keys)
        for entry in entries.values():
            del entry['_calculated']
            if not entry_validator.validate(entry):
                self._logger.error(str(text_token({'E03001': {
                    'errors': pformat(entry_validator.errors, width=180),
                    'entry': pformat(entry, width=180)}})))
                raise ValueError('Genomic library entry invalid.')
            references = [entry['gca'], entry['gcb']]
            problem_references = self._check_references(references, check_list)
            if problem_references:
                self._logger.error(str(text_token({'E03002': {
                    'entry': pformat(entry, width=180),
                    'references': problem_references}})))
                raise ValueError('Genomic library entry invalid.')


    def upsert(self, entries):
        """Insert or update into the genomic library.

        Validates, normalises and updates genetic code entries prior to storage. All input entries
        are updated with values as they were stored (but not encoded).

        Args
        ----
        entries (dict(dict)): keys are signatures and dicts are genetic code
            entries. Values will be normalised & updated in place
        """
        self._normalize(entries)
        updated_entries = self.library.upsert(entries.values(), self._update_str, {}, UPDATE_RETURNING_COLS)
        for updated_entry in updated_entries:
            entry = entries[updated_entry['signature']]
            entry.update(updated_entry)
