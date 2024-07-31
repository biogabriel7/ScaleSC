import scanpy as sc
import gc
import pandas as pd
import numpy as np

import logging
import sys

import time
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("scaleSC")
ConsoleOutputHandler = logging.StreamHandler()
logger.addHandler(ConsoleOutputHandler)

steps_t = []
steps = []

def log_time(step=None):
    steps.append(step)
    steps_t.append(time.time())
    if len(steps_t)>1:
        logger.warning(f"STEP {step}: \t time cost \t {steps_t[-1] - steps_t[-2]}")
    return

data_name = '70k_human_lung'
# data_name = '1.3M_mouse_brain'

out_dir = '/edgehpc/dept/compbio/projects/scaleSC/src/log_time/'

logger.warning(f'{data_name}')

log_time(step='start')

# load data
dict_adata_file = {'70k_human_lung':'/edgehpc/dept/compbio/projects/scaleSC/data/70k_human_lung/krasnow_hlca_10x.sparse.h5ad',
                   '1.3M_mouse_brain':'/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/1M_brain_cells_10X.sparse.h5ad',
                  '2.5M_human_brain_Linnarsson':'/edgehpc/dept/compbio/projects/scaleSC/data/2.5M_human_brain_Linnarsson/Linnarsson_neuron.h5ad',
                  '1.7M_human_brain_ROSMAP':'/edgehpc/dept/compbio/projects/scaleSC/data/1.7M_human_brain_ROSMAP/ROSMAP.h5ad'}
adata = sc.read_h5ad(dict_adata_file[data_name])
log_time(step='load_data')

# data to gpu
log_time(step='data_to_gpu')

# qc
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
log_time(step='qc')

# normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
log_time(step='normalize')

# hvg
sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor="seurat_v3")
adata = adata[:, adata.var["highly_variable"]]
log_time(step='hvg')

# pca
sc.pp.pca(adata, n_comps=50)
log_time(step='pca')

# harmony
sc.external.pp.harmony_integrate(adata, key="sampleID", basis='X_pca', adjusted_basis='X_pca_harmony')
log_time(step='harmony')

# neighbors
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50, use_rep='X_pca_harmony')
log_time(step='neighbors')

# umap
sc.tl.umap(adata)
adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
log_time(step='umap')

# leiden cluster
sc.tl.leiden(adata, resolution=0.5)
log_time(step='leiden')

start_time = steps_t[0]
steps_t = [t-start_time for t in steps_t]
steps_tcost = steps_t.copy()
if len(steps_t)>1:
    for i in range(1, len(steps_t)):
        steps_tcost[i] = steps_t[i] - steps_t[i-1]

df_report = pd.DataFrame({'step':steps, 'time':steps_t, 'time_cost':steps_tcost})
df_report = df_report.round(3)
df_report.to_csv(os.path.join(out_dir, f'time.cost__cpu__{data_name}.csv'), index=False)

# adata.write(f'{data_name}_harmony_cpu.h5ad')






