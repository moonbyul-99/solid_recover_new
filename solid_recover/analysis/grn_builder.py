"""Gene Regulatory Network (GRN) construction from ATAC peaks.

This module wraps snapatac2's network annotation and TF motif analysis
functionality to build TF→RE(regulatory element)→TG(target gene)
triplets for downstream GRN visualization and analysis.

The workflow:
1. Build peak-gene network from genomic coordinates
2. Add TF binding information via motif scanning
3. Extract TF→RE and RE→TG relationships

Note: This module requires snapatac2 to be installed. Install via:
    pip install solid-recover[analysis]
or
    pip install snapatac2 rustworkx
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _check_snapatac2():
    """Check if snapatac2 is installed, raise informative error if not."""
    try:
        import snapatac2 as sp
        return sp
    except ImportError:
        raise ImportError(
            "snapatac2 is required for GRN construction. "
            "Install it via: pip install solid-recover[analysis]\n"
            "or: pip install snapatac2 rustworkx"
        )


def parse_genomic_location(loc_str: str) -> Tuple[str, int, int]:
    """Parse genomic location string into (chrom, start, end).

    Parameters
    ----------
    loc_str : str
        Genomic location in format "chrN:start-end", e.g., "chr1:1000-2000"

    Returns
    -------
    tuple
        (chromosome, start_position, end_position)

    Examples
    --------
    >>> parse_genomic_location("chr1:1000-2000")
    ('chr1', 1000, 2000)
    """
    match = re.match(r'(.+):(\d+)-(\d+)', str(loc_str))
    if not match:
        raise ValueError(
            f"Invalid genomic location format: {loc_str}. "
            f"Expected format: 'chr:start-end'"
        )
    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    return chrom, start, end


class GRNBuilder:
    """Build Gene Regulatory Networks from ATAC peaks using snapatac2.

    This class wraps snapatac2's peak-gene annotation and TF motif
    enrichment analysis to construct TF→RE→TG regulatory triplets.

    Parameters
    ----------
    genome : str, default="hg38"
        Reference genome identifier (passed to snapatac2).
    motif_db : str, default="cis_bp"
        Transcription factor motif database to use.

    Attributes
    ----------
    network : rustworkx.PyDiGraph
        The constructed regulatory network graph from snapatac2.

    Examples
    --------
    >>> from solid_recover.analysis import GRNBuilder
    >>> grn = GRNBuilder(genome="hg38")
    >>> 
    >>> # Build peak-gene links
    >>> df_links = grn.build_peak_gene_network(top_peaks, top_genes)
    >>> 
    >>> # Add TF binding and extract TF→RE relationships
    >>> df_tf_re = grn.add_tf_binding(tf_filter=top_genes)
    """

    def __init__(
        self,
        genome: str = "hg38",
        motif_db: str = "cis_bp",
    ):
        self.sp = _check_snapatac2()
        self.genome = genome
        self.motif_db = motif_db
        self.network = None

    def build_peak_gene_network(
        self,
        peaks: List[str],
        genes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build peak-gene regulatory network from genomic annotations.

        Uses snapatac2's init_network_from_annotation to link peaks to
        their nearest/target genes based on genomic proximity and
        regulatory annotations.

        Parameters
        ----------
        peaks : list of str
            List of peak coordinates in "chr:start-end" format.
            Example: ["chr1:1000-2000", "chr2:5000-6000"]
        genes : list of str, optional
            List of gene names to restrict the network to.
            If None, all genes annotated to peaks are included.

        Returns
        -------
        pd.DataFrame
            Peak-gene links with columns:
            - 'peak': peak coordinate string
            - 'gene': target gene name

        Examples
        --------
        >>> peaks = ["chr1:1000-2000", "chr1:5000-6000"]
        >>> genes = ["BRCA1", "TP53", "MYC"]
        >>> df_links = grn.build_peak_gene_network(peaks, genes)
        """
        sp = self.sp

        # Initialize network from peak annotations
        self.network = sp.tl.init_network_from_annotation(
            peaks,
            anno_file=getattr(sp.genome, self.genome),
        )

        # Extract edges from the network graph
        df_links = self._extract_peak_gene_links(self.network)

        # Filter to specified genes if provided
        if genes is not None:
            gene_set = set(genes)
            df_links = df_links[df_links['gene'].isin(gene_set)]

        return df_links

    def add_tf_binding(
        self,
        tf_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Add TF binding information to the network via motif scanning.

        Scans peaks for enriched TF motifs using the specified motif
        database and adds TF→RE edges to the network.

        Parameters
        ----------
        tf_filter : list of str, optional
            List of TF names to restrict analysis to.
            If None, all TFs in the motif database are considered.

        Returns
        -------
        pd.DataFrame
            TF-RE links with columns:
            - 'TF': transcription factor name
            - 'Peak_ID': peak coordinate string
            - 'Target_Type': always "region"

        Notes
        -----
        Must call build_peak_gene_network() first to initialize the network.
        """
        if self.network is None:
            raise RuntimeError(
                "Network not initialized. Call build_peak_gene_network() first."
            )

        sp = self.sp

        # Add TF binding to network
        sp.tl.add_tf_binding(
            self.network,
            motifs=sp.datasets.cis_bp(unique=True),
            genome_fasta=getattr(sp.genome, self.genome),
        )

        # Extract TF→RE relationships
        df_tf_re = self.extract_tf_re_links(tf_filter=tf_filter)

        return df_tf_re

    def extract_tf_re_links(
        self,
        network=None,
        tf_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract TF→RE (regulatory element) edges from the network.

        Traverses the network graph to find all motif→region edges,
        optionally filtering to a specific set of TFs.

        Parameters
        ----------
        network : rustworkx.PyDiGraph, optional
            Network graph to extract from. If None, uses self.network.
        tf_filter : list of str, optional
            Filter to only include these TFs.

        Returns
        -------
        pd.DataFrame
            TF-RE links with columns:
            - 'TF': transcription factor name
            - 'Peak_ID': peak coordinate string
            - 'Target_Type': node type (usually "region")

        Examples
        --------
        >>> df_tf_re = grn.extract_tf_re_links(tf_filter=['IRF8', 'CEBPB', 'SPI1'])
        """
        if network is None:
            network = self.network

        if network is None:
            raise RuntimeError(
                "No network available. Call build_peak_gene_network() first."
            )

        links = []

        # Traverse all nodes in the network
        for node_idx in network.node_indices():
            node_data = network[node_idx]

            # Check if this node is a TF (motif)
            if hasattr(node_data, 'type') and node_data.type == "motif":
                tf_name = node_data.id

                # Get all downstream targets
                for target in network.successors(node_idx):
                    # Handle both integer indices and node objects
                    if isinstance(target, int):
                        target_data = network[target]
                    else:
                        target_data = target

                    # Only include region (peak) targets
                    if hasattr(target_data, 'type') and target_data.type == "region":
                        links.append({
                            "TF": tf_name,
                            "Peak_ID": target_data.id,
                            "Target_Type": target_data.type
                        })

        df_tf_re = pd.DataFrame(links)

        # Apply TF filter if specified
        if tf_filter is not None:
            tf_set = set(tf_filter)
            df_tf_re = df_tf_re[df_tf_re['TF'].isin(tf_set)]

        return df_tf_re

    def _extract_peak_gene_links(self, network) -> pd.DataFrame:
        """Extract all peak→gene edges from the network graph.

        Internal helper method to traverse the network and collect
        all region→gene relationships.

        Parameters
        ----------
        network : rustworkx.PyDiGraph
            The regulatory network graph.

        Returns
        -------
        pd.DataFrame
            Peak-gene links with columns ['peak', 'gene'].
        """
        edges = []
        nodes = network.nodes()

        for edge_indices in network.edge_indices():
            source_idx, target_idx = network.get_edge_endpoints_by_index(edge_indices)
            u = nodes[source_idx]
            v = nodes[target_idx]

            edges.append({
                'peak': u.id,
                'gene': v.id
            })

        return pd.DataFrame(edges)


def filter_mutual_elements(
    df_links: pd.DataFrame,
    top_genes: List[str],
    top_peaks: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Filter network links to only include specified genes and peaks.

    Helper function to intersect network edges with top-ranked features
    from decomposition analysis.

    Parameters
    ----------
    df_links : pd.DataFrame
        Peak-gene links DataFrame with columns ['peak', 'gene'].
    top_genes : list of str
        List of top gene names (e.g., from decomposition analysis).
    top_peaks : list of str
        List of top peak coordinates.

    Returns
    -------
    tuple
        (filtered_df, mutual_genes, mutual_peaks)
        - filtered_df: DataFrame with only mutual genes and peaks
        - mutual_genes: list of genes present in both df_links and top_genes
        - mutual_peaks: list of peaks present in both df_links and top_peaks

    Examples
    --------
    >>> df_filtered, mutual_genes, mutual_peaks = filter_mutual_elements(
    ...     df_links, top_genes, top_peaks
    ... )
    """
    mutual_genes = np.intersect1d(df_links['gene'].values, top_genes)
    mutual_peaks = np.intersect1d(df_links['peak'].values, top_peaks)

    idx_gene = df_links['gene'].isin(mutual_genes)
    idx_peak = df_links['peak'].isin(mutual_peaks)
    df_filtered = df_links[idx_gene & idx_peak]

    return df_filtered, mutual_genes.tolist(), mutual_peaks.tolist()


__all__ = [
    "GRNBuilder",
    "parse_genomic_location",
    "filter_mutual_elements",
]
