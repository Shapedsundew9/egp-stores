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
    edges: tuple[str, ...]
    missing: tuple[str, ...]
    lost: tuple[str, ...]


class StructureKeys(TypedDict):
    edges: tuple[str, ...]


class GmsGraphViews(TypedDict):
    structure: StructureKeys
    ancestory: AncestryKeys
