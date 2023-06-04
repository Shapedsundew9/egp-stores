from typing import NotRequired, TypedDict
from pypgtable.typing import TableConfigNorm, TableConfig


class GenePoolConfig(TypedDict):
    gene_pool: NotRequired[TableConfig]
    meta_data: NotRequired[TableConfig]
    gp_metrics: NotRequired[TableConfig]
    pgc_metrics: NotRequired[TableConfig]


class GenePoolConfigNorm(TypedDict):
    gene_pool: TableConfigNorm
    meta_data: TableConfigNorm
    gp_metrics: TableConfigNorm
    pgc_metrics: TableConfigNorm


class AncestryKeys(TypedDict):
    edge_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    lost_keys: tuple[str, ...]


class StructureKeys(TypedDict):
    edge_keys: tuple[str, ...]


class GmsGraphViews(TypedDict):
    structure: StructureKeys
    ancestry: AncestryKeys
