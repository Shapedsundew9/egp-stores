""" EGP stores typing."""
from typing import NotRequired, TypedDict
from pypgtable.pypgtable_typing import TableConfigNorm, TableConfig


class GenePoolConfig(TypedDict):
    """Gene Pool Config structure prior to normalization."""

    gene_pool: NotRequired[TableConfig]
    meta_data: NotRequired[TableConfig]
    gp_metrics: NotRequired[TableConfig]
    pgc_metrics: NotRequired[TableConfig]


class GenePoolConfigNorm(TypedDict):
    """Gene Pool Config structure post normalization."""

    gene_pool: TableConfigNorm
    meta_data: TableConfigNorm
    gp_metrics: TableConfigNorm
    pgc_metrics: TableConfigNorm


class AncestryKeys(TypedDict):
    """Definition of ancestry keys in the graph."""

    edge_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    lost_keys: tuple[str, ...]


class StructureKeys(TypedDict):
    """Definition of structure keys in the graph"""

    edge_keys: tuple[str, ...]


class GmsGraphViews(TypedDict):
    """Definition of the views in the graph."""

    structure: StructureKeys
    ancestry: AncestryKeys
