"""The Gene Pool (GP)."""

from copy import deepcopy
from json import load
from logging import NullHandler, getLogger, DEBUG
from os.path import dirname, join
from pprint import pformat
from functools import partial

from pypgtable import table

from .genetic_material_store import genetic_material_store, GMS_TABLE_SCHEMA, UPDATE_STR
from .utils.text_token import register_token_code, text_token
from egp_types.conversions import *
from egp_types.gc_type_tools import merge
from egp_types.xgc_validator import LGC_json_entry_validator
from egp_types._GC import ref_str
from egp_types.gc_graph import gc_graph


_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)


def compress_igraph(x):
    """Extract the internal representation and compress."""
    return compress_json(x.save())


def decompress_igraph(x):
    """Create a gc_graph() from an decompressed internal representation."""
    return gc_graph(internal=decompress_json(x))


_CONVERSIONS = (
    ('graph', compress_json, decompress_json),
    ('meta_data', compress_json, decompress_json),  # TODO: Why store this?
    ('inputs', None, memoryview_to_bytes),
    ('outputs', None, memoryview_to_bytes),
    ('igraph', compress_igraph, decompress_igraph)
)


# Tree structure
_LEL = 'gca_ref'
_REL = 'gcb_ref'
_NL = 'ref'
_PTR_MAP = {
    _LEL: _NL,
    _REL: _NL
}


_GP_TABLE_SCHEMA = deepcopy(GMS_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gp_table_format.json"), "r") as file_ptr:
    merge(_GP_TABLE_SCHEMA, load(file_ptr))


# The default config
# The gene pool default config
_DEFAULT_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'gene_pool',
    'ptr_map': _PTR_MAP,
    'schema': _GP_TABLE_SCHEMA,
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


class gene_pool(genetic_material_store):
    """Store of transient genetic codes & associated data for domain era populations.

    The gene_pool is responsible for:
        1. Populating calcuable entry fields.
        2. Providing an application interface to the fields.
        3. Persistence of updates to the gene pool.

    The gene pool must be consistent i.e. no entry can depend on a genetic code
    that is not in the gene_pool.

    The primary difference with the genomic_library is the presence of transient data
    and the fast (and space saving) UID method of referencing and storing active GC's.
    For the same reasons validation is not performed unless in debug mode.

    The gene_pool is more local (faster to access) than the genomic library
    due to the heavy transaction load. It is also designed to be multi-process
    memory efficient.

    The public member self.pool is the local cache of gGC's.
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
        self.pool = table(config)
        self._update_str = UPDATE_STR.replace('__table__', config['table'])
        self.encode_value = self.pool.encode_value
        self.select = self.pool.select
        self.recursive_select = self.pool.recursive_select
        if self._pool.raw.creator:
            self._pool.raw.arbitrary_sql(sql_functions(), read=False)

    def __getitem__(self, ref:int) -> Tuple[gGC):
        """Recursively select genetic codes starting with 'ref'.

        Args
        ----
        ref (int): Reference of GC to select.

        Returns
        -------
        (gGC): All genetic codes & codons constructing & including signature gc.
        """
        return self.pool[self.pool.encode_value('signature', signature)]

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
            if self.pool[reference] is None and reference not in check_list:
                naughty_list.append(reference)
            else:
                check_list.add(reference)
        return naughty_list

