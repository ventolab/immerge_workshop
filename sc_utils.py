# Libraries

import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import math
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter


# This function takes an anndata and scores each cell'se status for stress and healthiness
def score_cell_status(adata, health_score_only=False):
    import scanpy as sc

    if health_score_only:
        healthy_genes = [x.strip() for x in open('/nfs/team292/lg18/utils/nondying_cells_genes.txt')]
        sc.tl.score_genes(adata, score_name='health_score', gene_list=healthy_genes)
        
    else:   
        SenMayo_genes = [x.strip() for x in open('/nfs/team292/lg18/utils/senescence_SenMayo.tsv')]
        sc.tl.score_genes(adata, score_name='senescence_score', gene_list=SenMayo_genes)
        
        stress_genes = [x.strip() for x in open('/nfs/team292/lg18/utils/stress_mmu_vandenBrink.tsv')]
        sc.tl.score_genes(adata, score_name='stress_score_vandenBrink', gene_list=stress_genes)
        
        healthy_genes = [x.strip() for x in open('/nfs/team292/lg18/utils/nondying_cells_genes.txt')]
        sc.tl.score_genes(adata, score_name='health_score', gene_list=healthy_genes)
        
        metabolic_shutdown_genes = [x.strip() for x in open('/nfs/team292/lg18/utils/metabolic_shutdown_genes_curated.txt')]
        sc.tl.score_genes(adata, score_name='metabolic_activity', gene_list=metabolic_shutdown_genes)
    
    return adata


def cc_scoring(adata, genes='seurat'): #cell cycle scoring
    if genes=='seurat': #default genes from Seurat
        
        s_genes = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","HELLS",
                   "RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45",
                   "CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
        
        g2m_genes =["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF",
                    "TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP",
                    "CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR",
                    "AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]
    
    if genes=='macosko':
        cc_genes = pd.read_table(cc_genes_file, delimiter='\t')
        s_genes = cc_genes['S'].dropna()
        g2m_genes = cc_genes['G2.M'].dropna()
    
    s_genes_filt = adata.var_names[np.in1d(adata.var_names, s_genes)]
    g2m_genes_filt = adata.var_names[np.in1d(adata.var_names, g2m_genes)]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_filt, g2m_genes=g2m_genes_filt)


def bh(pvalues):
    """
    Computes the Benjamini-Hochberg FDR correction.
    
    Input:
        * pvals - vector of p-values to correct
    """
    pvalues = np.array(pvalues)
    n = int(pvalues.shape[0])
    new_pvalues = np.empty(n)
    values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = n - i
        pvalue, index = vals
        new_values.append((n/rank) * pvalue)
    for i in range(0, int(n)-1):
        if new_values[i] < new_values[i+1]:
            new_values[i+1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        new_pvalues[index] = new_values[i]
    return new_pvalues


def bonf(pvalues):
    """
    Computes the Bonferroni FDR correction.
    
    Input:
        * pvals - vector of p-values to correct
    """
    new_pvalues = np.array(pvalues) * len(pvalues)
    new_pvalues[new_pvalues>1] = 1
    return new_pvalues


def obsm_to_csv(adata,X_name,csv):
    X_out = adata.obsm[X_name]
    df_out = pd.DataFrame(
    	X_out,
    	index=adata.obs_names,
    	columns=[f"X_{i}" for i in range(X_out.shape[1])])
	# Save to CSV
    df_out.to_csv(csv)
    
def Barplot(which_var, adata, var='lineage', height=3, color = False, pdf=False, pdfpath=os.getcwd(), legendcols=1):
    plotdata = pd.crosstab(adata.obs[var], adata.obs[which_var], normalize='index') * 100

    if 'category' in plotdata.index.dtype.name:
        plotdata.index.reorder_categories(adata.obs[var].cat.categories[::-1])

    if not color:
        ax1 = plotdata.plot.barh(stacked = True, edgecolor = 'none', zorder = 3, figsize = (4,height), fontsize = 14, grid = False)
    else:
        ax1 = plotdata.plot.barh(stacked = True, edgecolor = 'none', zorder = 3, figsize = (4,height), fontsize = 14, grid = False, color = color)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_xlim(0, 100)
    ax1.set_title(which_var+' %')
    ax1.set_ylabel(var)
    horiz_offset = 1
    vert_offset = 1.
    ax1 = ax1.legend(bbox_to_anchor = (horiz_offset, vert_offset),ncol=legendcols)
    which_var_name=which_var.replace(" ", "")
    if pdf==True:
        ax1.figure.savefig(pdfpath+'_'+
                           which_var_name+'proportions_per'+var+'.pdf', bbox_inches='tight',
                       dpi=300, orientation='landscape', format= 'pdf')



def quick_markers(adata, cluster_key, cell_groups=None, layer=None, n_markers=10, fdr=0.01, express_cut=0.9, r_output=False):
    """
    Identifies top N markers for each cluster in an AnnData object using a TF-IDF-based strategy.
    Implemented as in the SoupX library for R.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix from Scanpy.

    cluster_key : str
        Key in adata.obs for the cluster labels.

    cell_groups : list, optional (default: None)
        List of cell groups to be compared in the analysis.

    layer : str, optional (default: None)
        Layer to use for the analysis. If None, uses adata.X.

    n_markers : int, optional (default: 10)
        Number of marker genes to return per cluster.

    fdr : float, optional (default: 0.01)
        False discovery rate for the hypergeometric test.

    express_cut : float, optional (default: 0.9)
        Value above which a gene is considered expressed.

    r_output : bool, optional (default: False)
        Whether reporting the same exact column names as the SoupX version.

    Returns
    -------
    markers : pandas.DataFrame
        A pandas.DataFrame with top N markers for each cluster and their statistics.
    """
    if cell_groups is not None:
        adata_ = adata[adata.obs[cluster_key].isin(cell_groups)]
    else:
        adata_ = adata

    # Convert to CSR matrix if necessary and binarize the expression data
    if layer is not None:
        toc = csr_matrix(adata_.layers[layer]) if not issparse(adata_.layers[layer]) else adata_.layers[layer]
    else:
        toc = csr_matrix(adata_.X) if not issparse(adata_.X) else adata_.X

    toc_bin = (toc > express_cut).astype(int)

    # Cluster information
    clusters = pd.Categorical(adata_.obs[cluster_key]).codes
    unique_clusters = np.unique(clusters)
    cl_counts = np.asarray([np.sum(clusters == cl) for cl in unique_clusters]).reshape(-1, 1)

    # Calculate observed and total frequency
    n_obs = np.asarray([np.asarray(toc_bin[clusters == cl, :].sum(axis=0)).flatten() for cl in unique_clusters])
    n_tot = n_obs.sum(axis=0)

    # Term Frequency (TF), Inverse Document Frequency (IDF) and TF-IDF
    tf = n_obs / cl_counts
    idf = np.log(len(clusters) / n_tot)
    tf_idf = tf * idf

    # Calculate additional metrics
    gene_freq_outside_cluster = (n_tot - n_obs) / (len(clusters) - cl_counts)
    gene_freq_global = n_tot / len(clusters)

    # Calculate second-best TF score and corresponding cluster name
    second_best_tf = np.zeros_like(tf)
    second_best_cluster_idx = np.zeros(tf.shape[1], dtype=int)
    for gene_idx in range(tf.shape[1]):
        tf_scores = tf[:, gene_idx]
        second_best_idx = np.argsort(tf_scores)[-2]  # Get index of second-highest value
        second_best_tf[:, gene_idx] = tf_scores[second_best_idx]
        second_best_cluster_idx[gene_idx] = unique_clusters[second_best_idx]

    # P-values
    p_values = np.array \
        ([hypergeom.sf(n_obs[i] - 1, len(clusters), n_tot, cl_counts[i]) for i in range(len(unique_clusters))])

    # FDR correction using statsmodels (global across all gene-cluster pairs)
    p_flat = p_values.flatten()
    reject, q_flat, _, _ = multipletests(p_flat, alpha=fdr, method='fdr_bh')
    q_values = q_flat.reshape(p_values.shape)

    # Select top N markers by iterating over columns of p-values matrix
    top_markers = {cl: [] for cl in unique_clusters}
    for gene_idx in range(tf_idf.shape[1]):
        for cl in unique_clusters:
            # Filter genes by FDR (statsmodels handles NaN automatically)
            q_val = q_values[cl, gene_idx]
            if not np.isnan(q_val) and q_val < fdr:
                top_markers[cl].append((gene_idx, tf_idf[cl, gene_idx]))

    # Sort and select top genes for each cluster
    for cl in top_markers:
        top_markers[cl].sort(key=lambda x: x[1], reverse=True)  # Sort by TF-IDF
        top_markers[cl] = [gene_idx for gene_idx, _ in top_markers[cl][:n_markers]]

    # Constructing the output DataFrame
    marker_data = []
    for cl, markers in top_markers.items():
        for gene_idx in markers:
            gene = adata.var_names[gene_idx]
            second_best_cl = second_best_cluster_idx[gene_idx]
            marker_data.append({
                'gene': gene,
                'cluster': adata.obs[cluster_key].cat.categories[cl],
                'tf': tf[cl, gene_idx],
                'idf': idf[gene_idx],
                'tf_idf': tf_idf[cl, gene_idx],
                'gene_frequency_outside_cluster': gene_freq_outside_cluster[cl, gene_idx],
                'gene_frequency_global': gene_freq_global[gene_idx],
                'second_best_tf': second_best_tf[cl, gene_idx],
                'second_best_cluster': adata.obs[cluster_key].cat.categories[second_best_cl],
                'pval': p_values[cl, gene_idx],
                'qval': q_values[cl, gene_idx]
            })

    markers = pd.DataFrame(marker_data)
    if markers.shape == (0, 0):
        markers = pd.DataFrame(columns=['gene', 'cluster', 'tf', 'idf', 'tf_idf', 'gene_frequency_outside_cluster',
                                        'gene_frequency_global', 'second_best_tf', 'second_best_cluster', 'pval',
                                        'qval'])

    if r_output:
        cols = ['gene', 'cluster', 'tf', 'gene_frequency_outside_cluster', 'second_best_tf', 'gene_frequency_global', 'second_best_cluster', 'tf_idf', 'idf', 'qval']
        markers = markers[cols]
        markers.columns = ['gene', 'cluster', 'geneFrequency', 'geneFrequencyOutsideCluster',
                           'geneFrequencySecondBest', 'geneFrequencyGlobal', 'secondBestClusterName', 'tfidf', 'idf',
                           'qval']
    return markers


def filterTFIDF(mrks, N=20, genes_to_filter=None):
    top_n = N # Number of top-genes per cell type
    genes = set()
    markers = dict()
    if genes_to_filter is not None:
        mrks = mrks[[ i in genes_to_filter for i in mrks.gene ]]
            
    for cluster, df in mrks.groupby('cluster'):
        top_markers = [g for g in df['gene']][:top_n]
        markers[cluster] = top_markers
        
    return(markers)


# Plotting functions
def plotUMAP_per_value(adata, obs_variable='leiden', pdf_file='', **kwargs):
    adata.obs['obs_variable_rm'] = adata.obs[obs_variable]
    for va in set(adata.obs[obs_variable]):
        va = str(va)
        print(va)
        sc.pl.umap(adata,
            groups=[va], legend_loc='on data',
            color=['obs_variable_rm'],
            palette=['#eb4034', '#5169e0'],
            frameon=True, legend_fontsize=10, save=pdf_file+'.pdf',
            **kwargs,
        )
    del adata.obs['obs_variable_rm']

    
def plotUMAP_per_value_v2(adata, clust_key, frameon=False, legend_loc=None, pdf_file='', **kwargs):
    tmp = adata.copy()
    tmp = tmp[ tmp.obs[clust_key].notna() ]

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#f2f2f2", "#fa0000"]
    
        sc.pl.umap(
            tmp,
            groups=tmp.obs[clust].cat.categories[1:].values,
            color=[clust_key],
            frameon=frameon,
            legend_loc=legend_loc, legend_fontsize=7, save=f"{pdf_file}_{clust}.pdf",
            **kwargs,
        )

        
def load_metadata(
    metadata_csv: str,
    key_col: str = "Accession",
) -> dict:
    """
    Load a metadata CSV into a dict keyed by a chosen column.
 
    Parameters
    ----------
    metadata_csv : Path to the metadata CSV file.
    key_col      : Column to use as the dictionary key (default: 'Accession').
 
    Returns
    -------
    dict mapping key_col values → row dicts.
    """
    metadata = {}
    with open(metadata_csv, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row.get(key_col, "").strip()
            if key:
                metadata[key] = row
    print(f"Loaded {len(metadata)} metadata entries from '{metadata_csv}'")
    return metadata


# Helper to filter markers present in adata and report status
def get_available_markers(marker_dict, name, adata):
    if not marker_dict:
        return {}
    
    avail = {
        ct: [g for g in genes if g in adata.var_names]
        for ct, genes in marker_dict.items()
    }
    # Remove empty categories
    avail = {ct: g for ct, g in avail.items() if g}
    
    all_genes = [g for genes in marker_dict.values() for g in genes]
    found_count = sum(len(v) for v in avail.values())
    missing = [g for g in all_genes if g not in adata.var_names]
    
    print(f"{name:20} found: {found_count} / {len(all_genes)}")
    if missing:
        print(f"  Not found: {missing}")
    return avail

def plot_umap_highlight(adata, column, ncols=3, point_size=0.5, highlight_color='red'):
    
    values = adata.obs[column].unique()
    n_values = len(values)
    nrows = math.ceil(n_values / ncols)
    umap = adata.obsm['X_umap']
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
    
    for i, val in enumerate(values):
        ax = axes[i]
        mask = adata.obs[column] == val
        
        ax.scatter(umap[~mask, 0], umap[~mask, 1], c='lightgrey', s=point_size, rasterized=True)
        ax.scatter(umap[mask, 0], umap[mask, 1], c=highlight_color, s=point_size, rasterized=True, label=val)
        
        ax.legend(markerscale=5, frameon=False, fontsize=8)
        ax.set_title(val, fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_upset(
    gene_sets,
    title="",
    min_subset_size=1,
    max_n_bars=30,
    bar_color="#4C72B0",
    dot_color="#2c2c2c",
    set_bar_color="#DD8452",
    save_path=None,
):
    """
    Pure-matplotlib UpsetPlot.

    Parameters
    ----------
    gene_sets : dict  {cluster_name: set_of_genes}
    """
    clusters = sorted(gene_sets.keys())
    n_sets   = len(clusters)

    if n_sets < 2:
        print(f"  '{title}': only 1 cluster — skipping")
        return

    # ── 1. Build intersection counts ──────────────────────────────────────────
    all_genes = set().union(*gene_sets.values())

    membership = {}          # gene → frozenset of clusters
    for gene in all_genes:
        membership[gene] = frozenset(c for c in clusters if gene in gene_sets[c])

    counts = Counter(membership.values())
    counts = {k: v for k, v in counts.items() if v >= min_subset_size}

    if not counts:
        print(f"  '{title}': no intersections ≥ {min_subset_size} — skipping")
        return

    # Sort by count descending, then limit bars
    sorted_inters = sorted(counts.items(), key=lambda x: -x[1])[:max_n_bars]
    inter_sets    = [s for s, _ in sorted_inters]
    inter_counts  = [c for _, c in sorted_inters]
    n_bars        = len(sorted_inters)

    set_sizes = {c: len(gene_sets[c]) for c in clusters}

    # ── 2. Layout ─────────────────────────────────────────────────────────────
    #  Col 0 : set-size bars (horizontal)  |  Col 1 : dot matrix + inter bars
    #  Row 0 : intersection bar chart
    #  Row 1 : dot matrix

    left_w   = max(1.8, n_sets * 0.22)   # width of set-size panel
    right_w  = max(5,   n_bars * 0.55)   # width of intersection panel
    top_h    = 3.0                        # height of bar chart
    bot_h    = n_sets * 0.55             # height of dot matrix

    fig = plt.figure(figsize=(left_w + right_w, top_h + bot_h))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[left_w, right_w],
        height_ratios=[top_h, bot_h],
        hspace=0.05, wspace=0.05,
    )

    ax_bar  = fig.add_subplot(gs[0, 1])   # top-right  : intersection counts
    ax_dot  = fig.add_subplot(gs[1, 1])   # bot-right  : dot matrix
    ax_set  = fig.add_subplot(gs[1, 0])   # bot-left   : set sizes
    ax_empty = fig.add_subplot(gs[0, 0])  # top-left   : empty
    ax_empty.axis('off')

    x_pos = np.arange(n_bars)
    y_pos = np.arange(n_sets)

    # ── 3. Intersection bar chart (top-right) ─────────────────────────────────
    ax_bar.bar(x_pos, inter_counts, color=bar_color, width=0.5, zorder=3)
    for xi, cnt in zip(x_pos, inter_counts):
        ax_bar.text(xi, cnt + max(inter_counts) * 0.01, str(cnt),
                    ha='center', va='bottom', fontsize=8)
    ax_bar.set_xlim(-0.5, n_bars - 0.5)
    ax_bar.set_ylabel("Intersection size", fontsize=10)
    ax_bar.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax_bar.tick_params(bottom=False, labelbottom=False)
    ax_bar.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)

    # ── 4. Dot matrix (bot-right) ─────────────────────────────────────────────
    # Grey background dots
    for yi in y_pos:
        ax_dot.scatter(x_pos, [yi] * n_bars,
                       color='#cccccc', s=80, zorder=2)

    # Filled dots + vertical connector lines for each intersection
    for xi, inter in enumerate(inter_sets):
        active_y = [clusters.index(c) for c in clusters if c in inter]
        if len(active_y) > 1:
            ax_dot.plot([xi, xi], [min(active_y), max(active_y)],
                        color=dot_color, lw=2.5, zorder=3)
        ax_dot.scatter([xi] * len(active_y), active_y,
                       color=dot_color, s=100, zorder=4)

    ax_dot.set_xlim(-0.5, n_bars - 0.5)
    ax_dot.set_ylim(-0.5, n_sets - 0.5)
    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels(clusters, fontsize=9)
    ax_dot.tick_params(bottom=False, labelbottom=False)
    ax_dot.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax_dot.yaxis.set_tick_params(length=0)

    # Alternating row shading for readability
    for yi in y_pos:
        if yi % 2 == 0:
            ax_dot.axhspan(yi - 0.5, yi + 0.5, color='#f5f5f5', zorder=1)

    # ── 5. Set-size bars (bot-left) ───────────────────────────────────────────
    sizes = [set_sizes[c] for c in clusters]
    ax_set.barh(y_pos, sizes, color=set_bar_color, height=0.5)
    ax_set.set_yticks(y_pos)
    ax_set.set_yticklabels([])
    ax_set.set_ylim(-0.5, n_sets - 0.5)
    ax_set.invert_xaxis()
    ax_set.set_xlabel("Set size", fontsize=10)
    ax_set.spines[['left', 'top']].set_visible(False)
    ax_set.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax_set.set_axisbelow(True)

    for yi in y_pos:
        if yi % 2 == 0:
            ax_set.axhspan(yi - 0.5, yi + 0.5, color='#f5f5f5', zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close()