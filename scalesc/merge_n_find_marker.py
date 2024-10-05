import scanpy as sc
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from trim_merge_marker import *


dict_cluster_col = {'/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/TB_harmony.h5ad':'TB.leiden_res.1', # small ()
                   '/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/OL_harmony.h5ad':'OL.leiden_res.1', # medium (566546, 36341)
                   '/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/MS_sn_atlas_brainNspcord_20240524_harmony.h5ad':'broad_celltype'} # large

fname = '/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/TB_harmony.h5ad'
# fname = '/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/OL_harmony.h5ad'
# fname = '/edgehpc/dept/compbio/projects/scaleSC/data/marker_merge/MS_sn_atlas_brainNspcord_20240524_harmony.h5ad'
adata = sc.read_h5ad(fname)
# adata = adata[:1000].copy()
print(adata.shape)
cluster_col = dict_cluster_col[fname]

cluster_col_after_merge = adata_cluster_merge(adata, cluster_col)
sc.pl.umap(adata, color=[cluster_col, cluster_col_after_merge], save=f'_{cluster_col}_before.vs.after_merge.png')


dict_acc, dict_markers = find_markers(adata, cluster_col_after_merge)
clusters = list(dict_markers.keys())
pd.DataFrame({'cluster':clusters, 'marker_genes':[dict_markers[_] for _ in clusters]}).to_csv(f'marker_{cluster_col_after_merge}.csv', index=False)