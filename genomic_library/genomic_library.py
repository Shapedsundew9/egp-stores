"""The Genomic Library class wraps the database_table."""

from copy import deepcopy
from json import dumps, load, loads
from logging import NullHandler, getLogger
from os.path import dirname, join
from pprint import pformat
from zlib import compress, decompress

from pypgtable.table import ID_FUNC, table

from .entry_validator import entry_validator
from .utils.text_token import register_token_code, text_token

_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


_EVO_UPDATE_RAW = ('SELECT total_weighted_values({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 1.0::DOUBLE PRECISION, 1::BIGINT)')
_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_count}',
    'PSCV': '{evolvability}',
    'PSCC': '{count}'
}
_EVO_UPDATE_STR = _EVO_UPDATE_RAW.format(**_EVO_UPDATE_MAP)
_COUNT_UPDATE_STR = '{count} = total_counts({count}, EXCLUDED.{count}, EXCLUDED.{_count}, 1::BIGINT)'
_REF_UPDATE_STR = '{references} = total_counts({references}, EXCLUDED.{references}, EXCLUDED.{_references}, 1::BIGINT)'
_UPDATE_STR = '{updated} = NOW(), ' + _EVO_UPDATE_STR + _COUNT_UPDATE_STR + _REF_UPDATE_STR
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


def _compress_json(obj):
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


def _decompress_json(obj):
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
    ('graph', _compress_json, _decompress_json),
    ('meta_data', _compress_json, _decompress_json),
    ('signature', str_to_sha256, ID_FUNC),
    ('gca', str_to_sha256, ID_FUNC),
    ('gcb', str_to_sha256, ID_FUNC),
    ('ancestor', str_to_sha256, ID_FUNC),
    ('creator', str_to_sha256, ID_FUNC),
    ('properties', encode_properties, decode_properties)
)


# _PROPERTIES must definition the bit position of all the properties listed in
# the "properties" feild of the entry_format.json definition.
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
    _DB_TABLE_SCHEMA = load(file_ptr)


# The default config
_DEFAULT_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'genomic_library',
    'ptr_map': _PTR_MAP,
    'schema': _DB_TABLE_SCHEMA,
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


class genomic_library():
    """Store of genetic codes & associated data.

    The genomic_library is responsible for:
        1. Populating calculable entry fields.
        2. Validating entries to be added to the store.
        3. Providing an application interface to the fields.

    The geneomic library must be self consistent i.e. no entry can reference a genetic code
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
        self._library = table(config)
        if self._library.raw.creator:
            with open(join(dirname(__file__), 'data/array_functions.sql'), 'r') as fileptr:
                self._library.arbitrary_sql(fileptr.read())
            for data_file in _DATA_FILES:
                abspath = join(_DATA_FILE_FOLDER, data_file)
                _logger.info(text_token({'I03000': {'table': self._library.raw.config['table'], 'file': abspath}}))
                with open(abspath, "r") as file_ptr:
                    self._library.insert((entry_validator.normalized(entry) for entry in load(file_ptr)))

    def __getitem__(self, signature):
        """Recursively select genetic codes starting with 'signature'.

        Args
        ----
        signature (obj): signature of GC to select.

        Returns
        -------
        (dict(dict)): All genetic codes & codons constructing & including signature gc.
        """
        return self._library[self._library.encode_value('signature', signature)]

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
            if self._library[reference] is None and reference not in check_list:
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
        gca = _NULL_GC_DATA
        if not entry['gca'] is None:
            if entry['gca'] not in entries.keys():
                if not self._library[entry['gca']]:
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
                if not self._library[entry['gcb']]:
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

    def encode_value(self, column, value):
        """Encode value using the registered conversion function for column.

        If no conversion function is registered value is returned unchanged.
        This function is provided to create encoded literals in query functions.

        Args
        ----
        column (str): Column name for value.
        value  (obj): Value to encode.

        Returns
        -------
        (obj): Encoded value
        """
        return self._library.encode_value(column, value)

    def select(self, query_str='', literals={}):
        """Select genetic codes from the genomic library.

        The select is done recursively returning all constituent genetic codes and
        codons.

        Args
        ----
        query_str (str): Query SQL: e.g. '{input_types} @> {inputs}'.
        literals (dict): Keys are labels used in query_str. Values are literals to replace the labels.
            NOTE: Literal values for encoded columns must be encoded. See encode_value().

        Returns
        -------
        (dict(dict)): 'signature' keys with gc values.
        """
        return self._library.recursive_select(query_str, literals, container='pkdict')

    def upsert(self, entries):
        """Insert or update into the genomic library.

        Validates, normalises and updates genetic code entries prior to storage. All input entries
        are updated with values as they were stored (but not encoded).

        Args
        ----
        entries (dict(dict)): keys are signatures and dicts are genetic code
            entries. Values will be normalised & updated in place
            then encoded just for storage.
        """
        self._normalize(entries)
        for updated_entry in self._library.upsert(entries.values(), _UPDATE_STR, {}, '*'):
            entry = entries[updated_entry['signature']]
            entry.update(updated_entry)
            entry['_references'] = entry['references']
            entry['_evolvability'] = entry['evolvability']
            entry['_count'] = entry['count']
        for entry in entries:
            entry['_stored'] = True
