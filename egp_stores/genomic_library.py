"""The Genomic Library class wraps the database_table."""

from copy import deepcopy
from json import load
from logging import Logger, NullHandler, getLogger
from os.path import dirname, join
from pprint import pformat
from typing import Any, Callable, Iterable, Literal, LiteralString

from egp_types.conversions import (compress_json, decompress_json,
                                   encode_properties, memoryview_to_bytes,
                                   str_to_datetime, str_to_sha256, str_to_UUID)
from egp_types.xgc_validator import LGC_json_load_entry_validator
from egp_utils.common import merge
from egp_utils.text_token import register_token_code, text_token
from pypgtable import table
from pypgtable.validators import raw_table_config_validator
from pypgtable.typing import RowIter, TableConfig, TableConfigNorm

from .genetic_material_store import (GMS_RAW_TABLE_SCHEMA, UPDATE_STR,
                                     genetic_material_store)

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


_NULL_GC_DATA: dict[str, int | bool] = {
    'code_depth': 0,
    'num_codes': 0,
    'raw_num_codons': 1,
    'generation': 0,
    'properties': 0,
    '_stored': True
}
_DATA_FILE_FOLDER: str = join(dirname(__file__), 'data')
_DATA_FILES: list[str] = ['codons.json', 'mutations.json']


# Tree structure
_LEL: Literal['gca'] = 'gca'
_REL: Literal['gcb'] = 'gcb'
_NL: Literal['signature'] = 'signature'
_PTR_MAP: dict[str, str] = {
    _LEL: _NL,
    _REL: _NL
}


register_token_code("I03000", "Adding data to table {table} from {file}.")
register_token_code('E03000', 'Query is not valid: {errors}: {query}')
register_token_code('E03001', 'Entry is not valid: {errors}: {entry}')
register_token_code('E03002', 'Referenced GC(s) {references} do not exist. Entry:\n{entry}:')


_CONVERSIONS: tuple[tuple[LiteralString, Callable[..., Any] | None, Callable[..., Any] | None], ...] = (
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


GL_RAW_TABLE_SCHEMA: dict[str, Any] = deepcopy(GMS_RAW_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gl_table_format.json"), "r") as file_ptr:
    merge(GL_RAW_TABLE_SCHEMA, load(file_ptr))
GL_HIGHER_LAYER_COLS: tuple[str, ...] = tuple((key for key in filter(lambda x: x[0] == '_', GL_RAW_TABLE_SCHEMA)))
GL_UPDATE_RETURNING_COLS: tuple[str, ...] = tuple((x[1:] for x in GL_HIGHER_LAYER_COLS)) + ('updated', 'created', 'signature')
GL_SIGNATURE_COLUMNS: tuple[str, ...] = tuple((key for key, _ in filter(
    lambda x: x[1].get('signature', False), GL_RAW_TABLE_SCHEMA.items())))

# The default config
_DEFAULT_CONFIG: TableConfigNorm = raw_table_config_validator.normalized({
    'database': {
        'dbname': 'erasmus_db'
    },
    'table': 'genomic_library',
    'ptr_map': _PTR_MAP,
    'schema': GL_RAW_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
    'conversions': _CONVERSIONS
})


def default_config() -> TableConfigNorm:
    """Return a deepcopy of the default genomic library configuration.

    The copy may be modified and used to create a genomic library instance.

    Returns
    -------
    (dict): The default genomic_library() configuration.
    """
    return deepcopy(_DEFAULT_CONFIG)


def sql_functions() -> str:
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

    def __init__(self, config: TableConfig | TableConfigNorm = _DEFAULT_CONFIG) -> None:
        """Connect to or create a genomic library.

        The genomic library data persists in a postgresql database. Multiple
        instances of the genomic_library() class can connect to the same database
        (use the same configuration).

        Args
        ----
        config(pypgtable config): The config is deep copied by pypgtable.
        """
        super().__init__()
        self.library: table = table(config)
        self._update_str: str = UPDATE_STR.replace('__table__', config['table'])
        self.encode_value: Callable[..., Any] = self.library.encode_value
        self.select: Callable[..., RowIter] = self.library.select
        self.recursive_select: Callable[..., RowIter] = self.library.recursive_select
        if self.library.raw.creator:
            self.library.raw.arbitrary_sql(sql_functions(), read=False)
            for data_file in _DATA_FILES:
                abspath: str = join(_DATA_FILE_FOLDER, data_file)
                _logger.info(text_token({'I03000': {'table': self.library.raw.config['table'], 'file': abspath}}))
                with open(abspath, "r") as file_ptr:
                    self.library.insert((LGC_json_load_entry_validator.normalized(entry) for entry in load(file_ptr)))

    def __getitem__(self, signature) -> Any:
        """Recursively select genetic codes starting with 'signature'.

        Args
        ----
        signature (obj): signature of GC to select.

        Returns
        -------
        (dict(dict)): All genetic codes & codons constructing & including signature gc.
        """
        return self.library[self.library.encode_value('signature', signature)]

    def _check_references(self, references: Iterable[bytes], check_list: set[bytes] | None = None) -> list[bytes]:
        """Verify all the references exist in the genomic library.

        Genetic codes reference each other. A debugging check is to verify the
        existence of all the references.

        Args
        ----
        references: List of genetic code signatures to look up.
        check_list: A set of known existing genetic codes signatures.

        Returns
        -------
        Empty list if all references exist else the signatures of missing references.
        """
        if check_list is None:
            check_list = set()
        naughty_list: list[bytes] = []
        for reference in references:
            if self.library[reference] is None and reference not in check_list:
                naughty_list.append(reference)
            else:
                check_list.add(reference)
        return naughty_list

    def _calculate_fields(self, entry: dict[str, Any], entries: dict[bytes, dict[str, Any]]) -> None:
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
        gca: dict[str, int | bool] = _NULL_GC_DATA
        if not entry['gca'] is None:
            if entry['gca'] not in entries.keys():
                if not self.library[entry['gca']]:
                    _logger.error('entry["gca"] = {} does not exist in the list to be stored or genomic library!'.format(entry['gca']))
                    _logger.error('Entries signature list: {}'.format(entries.keys()))
            else:
                gca = entries[entry['gca']]
                if not gca['_calculated']:
                    self._calculate_fields(gca, entries)

        gcb: dict[str, int | bool] = _NULL_GC_DATA
        if not entry['gcb'] is None:
            if entry['gcb'] not in entries.keys():
                if not self.library[entry['gcb']]:
                    _logger.error('entry["gcb"] = {} does not exist in the list to be stored or genomic library!'.format(entry['gcb']))
                    _logger.error('Entries signature list: {}'.format(entries.keys()))
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

    def _normalize(self, entries: dict[bytes, dict[str, Any]]) -> None:
        """Normalize entries before storing. The entries are modified in place.

        Genetic code statistics and meta data are updated / created for storage
        and checked for consistency. This can be a heavy process.

        Args
        ----
        entries: A dictionary entry['signature']: entry of genetic code dictionaries to be
            stored or updated in the genomic library.
        """
        _logger.debug("Normalizing {} entries.".format(len(entries)))
        for signature, entry in entries.items():
            entries[signature] = LGC_json_load_entry_validator.normalized(entry)
            entries[signature]['_calculated'] = False
        for entry in entries.values():
            self._calculate_fields(entry, entries)

        _logger.debug("Validating normalised entries before storing.")
        check_list: set[bytes] = set(entries.keys())
        for entry in entries.values():
            del entry['_calculated']
            if not LGC_json_load_entry_validator.validate(entry):
                _logger.error(
                    str(text_token({'E03001': {'errors': LGC_json_load_entry_validator.error_str(),
                        'entry': pformat(entry, width=180)}})))
                raise ValueError('Genomic library entry invalid.')
            references: list[bytes] = [entry['gca'], entry['gcb']]
            problem_references: list[bytes] = self._check_references(references, check_list)
            if problem_references:
                _logger.error(str(text_token({'E03002': {'entry': pformat(entry, width=180), 'references': problem_references}})))
                raise ValueError('Genomic library entry invalid.')

    def upsert(self, entries: dict[bytes, dict[str, Any]]) -> None:
        """Insert or update into the genomic library.

        Validates, normalises and updates genetic code entries prior to storage. All input entries
        are updated with values as they were stored (but not encoded).

        Args
        ----
        entries: Keys are signatures and dicts are genetic code
            entries. Values will be normalised & updated in place
        """
        self._normalize(entries)
        updated_entries: RowIter = self.library.upsert(entries.values(), self._update_str, {}, GL_UPDATE_RETURNING_COLS)
        for updated_entry in updated_entries:
            entry: dict[str, Any] = entries[updated_entry['signature']]
            entry.update(updated_entry)
