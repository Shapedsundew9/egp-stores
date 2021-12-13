"""The Genomic Library class wraps the database_table."""

from copy import deepcopy
from json import dumps, load, loads
from logging import NullHandler, getLogger
from os.path import dirname, join
from pprint import pformat
from zlib import compress, decompress
from uuid import UUID
from datetime import datetime

from pypgtable import table

from .entry_validator import entry_validator
from .utils.text_token import register_token_code, text_token

_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 0.0::DOUBLE PRECISION, 0::BIGINT)')
_WEIGHTED_FIXED_UPDATE_RAW = ('weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 0.0::DOUBLE PRECISION, 0::BIGINT)')
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR = '{evolvability} = ' + _WEIGHTED_VARIABLE_UPDATE_RAW.format(**_EVO_UPDATE_MAP)
_E_COUNT_UPDATE_STR = '{e_count} = variable_vector_weights_update(EXCLUDED.{e_count}, "__table__".{e_count}, EXCLUDED.{_e_count}, 1::BIGINT)'
_FIT_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{fitness}',
    'CSCC': 'EXCLUDED.{f_count}',
    'PSCV': '"__table__".{fitness}',
    'PSCC': '"__table__".{f_count}',
    'CSPV': 'EXCLUDED.{_fitness}',
    'CSPC': 'EXCLUDED.{_f_count}'
}
_FIT_UPDATE_STR = '{fitness} = ' + _WEIGHTED_VARIABLE_UPDATE_RAW.format(**_FIT_UPDATE_MAP)
_F_COUNT_UPDATE_STR = '{f_count} = variable_vector_weights_update(EXCLUDED.{f_count}, "__table__".{f_count}, EXCLUDED.{_f_count}, 1::BIGINT)'
_AC_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{alpha_class}',
    'CSCC': 'EXCLUDED.{ac_count}',
    'PSCV': '"__table__".{alpha_class}',
    'PSCC': '"__table__".{ac_count}',
    'CSPV': 'EXCLUDED.{_alpha_class}',
    'CSPC': 'EXCLUDED.{_ac_count}'
}
_AC_UPDATE_STR = '{alpha_class} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format(**_AC_UPDATE_MAP)
_AC_COUNT_UPDATE_STR = '{ac_count} = ' + _SCALAR_COUNT_UPDATE.format(**_AC_UPDATE_MAP)
_REF_UPDATE_MAP = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format(**_REF_UPDATE_MAP)
UPDATE_STR = ','.join((
    '{updated} = NOW()',
    _EVO_UPDATE_STR,
    _E_COUNT_UPDATE_STR,
    _FIT_UPDATE_STR,
    _F_COUNT_UPDATE_STR,
    _AC_UPDATE_STR,
    _AC_COUNT_UPDATE_STR,
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


register_token_code("I03000", "Adding data to table {table} from {file}.")
register_token_code('E03000', 'Query is not valid: {errors}: {query}')
register_token_code('E03001', 'Entry is not valid: {errors}: {entry}')
register_token_code('E03002', 'Referenced GC(s) {references} do not exist. Entry:\n{entry}:')


def compress_json(obj):
    """Compress a JSON dict object.

    Args
    ----
    obj (dict): Must be a JSON compatible dict.

    Returns
    -------
    (bytes): zlib compressed JSON string.
    """
    # TODO: Since the vast majority of data looks the same but is broken into many objects
    # it would be more efficient to use a compression algorithm that does not embedded the
    # compression token dictionary.
    if isinstance(obj, dict):
        return compress(dumps(obj).encode())
    if isinstance(obj, memoryview) or isinstance(obj, bytearray) or isinstance(obj, bytes):
        return obj
    if obj is None:
        return None
    raise TypeError("Un-encodeable type '{}': Expected 'dict' or byte type.".format(type(obj)))


def decompress_json(obj):
    """Decompress a compressed JSON dict object.

    Args
    ----
    obj (bytes): zlib compressed JSON string.

    Returns
    -------
    (dict): JSON dict.
    """
    return None if obj is None else loads(decompress(obj).decode())


def str_to_sha256(obj):
    """Convert a hexidecimal string to a bytearray.

    Args
    ----
    obj (str): Must be a hexadecimal string.

    Returns
    -------
    (bytearray): bytearray representation of the string.
    """
    if isinstance(obj, str):
        return bytearray.fromhex(obj)
    if isinstance(obj, memoryview) or isinstance(obj, bytearray) or isinstance(obj, bytes):
        return obj
    if obj is None:
        return None
    raise TypeError("Un-encodeable type '{}': Expected 'str' or byte type.".format(type(obj)))


def str_to_UUID(obj):
    """Convert a UUID formated string to a UUID object.

    Args
    ----
    obj (str): Must be a UUID formated hexadecimal string.

    Returns
    -------
    (uuid): UUID representation of the string.
    """
    if isinstance(obj, str):
        return UUID(obj)
    if isinstance(obj, UUID):
        return obj
    if obj is None:
        return None
    raise TypeError("Un-encodeable type '{}': Expected 'str' or UUID type.".format(type(obj)))


def str_to_datetime(obj):
    """Convert a datetime formated string to a datetime object.

    Args
    ----
    obj (str): Must be a datetime formated string.

    Returns
    -------
    (datetime): datetime representation of the string.
    """
    if isinstance(obj, str):
        return datetime.strptime(obj, "%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(obj, datetime):
        return obj
    if obj is None:
        return None
    raise TypeError("Un-encodeable type '{}': Expected 'str' or datetime type.".format(type(obj)))


def sha256_to_str(obj):
    """Convert a bytearray to its lowercase hexadecimal string representation.

    Args
    ----
    obj (bytearray): bytearray representation of the string.

    Returns
    -------
    (str): Lowercase hexadecimal string.
    """
    return None if obj is None else obj.hex()


def UUID_to_str(obj):
    """Convert a UUID to its lowercase hexadecimal string representation.

    Args
    ----
    obj (UUID): UUID representation of the string.

    Returns
    -------
    (str): Lowercase hexadecimal UUID string.
    """
    return None if obj is None else str(obj)


def datetime_to_str(obj):
    """Convert a datetime to its string representation.

    Args
    ----
    obj (datetime): datetime representation of the string.

    Returns
    -------
    (str): datetime string.
    """
    return None if obj is None else obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def encode_properties(obj):
    """Encode the properties dictionary into its integer representation.

    The properties field is a dictionary of properties to boolean values. Each
    property maps to a specific bit of a 64 bit value as defined
    by the _PROPERTIES dictionary.

    Args
    ----
    obj(dict): Properties dictionary.

    Returns
    -------
    (int): Integer representation of the properties dictionary.
    """
    if isinstance(obj, dict):
        bitfield = 0
        for k, v in filter(lambda x: x[1], obj.items()):
            bitfield |= 1 << _PROPERTIES[k]
        return bitfield
    if isinstance(obj, int):
        return obj
    raise TypeError("Un-encodeable type '{}': Expected 'dict' or integer type.".format(type(obj)))


def decode_properties(obj):
    """Decode the properties dictionary from its integer representation.

    The properties field is a dictionary of properties to boolean values. Each
    property maps to a specific bit of a 64 bit value as defined
    by the _PROPERTIES dictionary.

    Args
    ----
    obj(int): Integer representation of the properties dictionary.

    Returns
    -------
    (dict): Properties dictionary.
    """
    return {b: bool((1 << f) & obj) for b, f in _PROPERTIES.items()}


_CONVERSIONS = (
    ('graph', compress_json, decompress_json),
    ('meta_data', compress_json, decompress_json),
    ('signature', str_to_sha256, None),
    ('gca', str_to_sha256, None),
    ('gcb', str_to_sha256, None),
    ('ancestor_a', str_to_sha256, None),
    ('ancestor_b', str_to_sha256, None),
    ('pgc', str_to_sha256, None),
    ('creator', str_to_UUID, None),
    ('created', str_to_datetime, None),
    ('updated', str_to_datetime, None),
    ('properties', encode_properties, None)
)


# _PROPERTIES must define the bit position of all the properties listed in
# the "properties" field of the entry_format.json definition.
_PROPERTIES = {
    "extended": 0,
    "constant": 1,
    "conditional": 2,
    "deterministic": 3,
    "memory_modify": 4,
    "object_modify": 5,
    "physical": 6,
    "arithmetic": 16,
    "logical": 17,
    "bitwise": 18,
    "boolean": 19,
    "sequence": 20
}


_PTR_MAP = {
    'gca': 'signature',
    'gcb': 'signature'
}


with open(join(dirname(__file__), "formats/table_format.json"), "r") as file_ptr:
    _GL_TABLE_SCHEMA = load(file_ptr)
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


class genomic_library():
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
        self.library = table(config)
        self._update_str = UPDATE_STR.replace('__table__', config['table'])
        self.encode_value = self.library.encode_value
        self.select = self.library.select
        self.recursive_select = self.library.recursive_select
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
        for_insert = (x for x in filter(lambda x: not x.get('_stored', False), entries.values()))
        for_update = (x for x in filter(lambda x: x.get('_stored', False), entries.values()))
        updated_entries = list(self.library.upsert(for_insert, self._update_str, {}, UPDATE_RETURNING_COLS, HIGHER_LAYER_COLS))
        updated_entries.extend(self.library.update(for_update, self._update_str, {}, UPDATE_RETURNING_COLS))
        for updated_entry in updated_entries:
            entry = entries[updated_entry['signature']]
            entry.update(updated_entry)
            for hlk in HIGHER_LAYER_COLS:
                entry[hlk] = entry[hlk[1:]]
        for entry in entries:
            entry['_stored'] = True
