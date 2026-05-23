"""Analysis utilities for regulatory program decomposition.

This subpackage provides tools to:
1. Decompose learned latent embeddings into interpretable feature weights
   via regularized linear regression (ElasticNet)
2. Build Gene Regulatory Networks (GRNs) from ATAC peaks using snapatac2

Examples
--------
>>> from solid_recover.analysis import decompose_latent_to_features, GRNBuilder
>>> 
>>> # Decompose RNA features
>>> W_rna = decompose_latent_to_features(X_rna, Z_embed, gene_names)
>>> 
>>> # Build GRN
>>> grn = GRNBuilder(genome="hg38")
>>> df_links = grn.build_peak_gene_network(peaks, genes)
>>> df_tf_re = grn.add_tf_binding(tf_filter=top_genes)
"""

from solid_recover.analysis.decomposition import (
    compare_with_mlp_error,
    compute_reconstruction_error,
    decompose_latent_to_features,
    get_top_features_by_latent,
)
from solid_recover.analysis.grn_builder import (
    GRNBuilder,
    filter_mutual_elements,
    parse_genomic_location,
)

__all__ = [
    "decompose_latent_to_features",
    "compute_reconstruction_error",
    "compare_with_mlp_error",
    "get_top_features_by_latent",
    "GRNBuilder",
    "parse_genomic_location",
    "filter_mutual_elements",
]
