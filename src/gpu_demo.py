import scanpy as sc
import cupy as cp
import gc
import rapids_singlecell as rsc
import pandas as pd
import numpy as np

import logging
import sys
import os

import time
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("scaleSC")
ConsoleOutputHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(ConsoleOutputHandler)

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

data_name = '2.5M_human_brain_Linnarsson'
# data_name = '1.3M_mouse_brain'
# data_name = '70k_human_lung'

# out_dir = '/edgehpc/dept/compbio/projects/scaleSC/src/log_time/'
out_dir = '.'
m_memory = True
pool_alc = False
gpu_after_pca = True

logger.warning(f"gpu__after.pca.{gpu_after_pca}_mm.{m_memory}_pool.alc.{pool_alc}__{data_name}")

rmm.reinitialize(
    managed_memory=m_memory,  # Allows oversubscription
    pool_allocator=pool_alc,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)

steps_t = []
steps = []

def log_time(step=None):
    steps.append(step)
    steps_t.append(time.time())
    if len(steps_t)>1:
        logger.warning(f"STEP {step}: \t time cost \t {steps_t[-1] - steps_t[-2]}")
    return
    
log_time(step='start')

# load data
dict_adata_file = {'70k_human_lung':'/edgehpc/dept/compbio/projects/scaleSC/data/70k_human_lung/krasnow_hlca_10x.sparse.h5ad',
                   '1.3M_mouse_brain':'/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/1M_brain_cells_10X.sparse.h5ad',
                  '2.5M_human_brain_Linnarsson':'/edgehpc/dept/compbio/projects/scaleSC/data/2.5M_human_brain_Linnarsson/Linnarsson_neuron.h5ad',
                  '1.7M_human_brain_ROSMAP':'/edgehpc/dept/compbio/projects/scaleSC/data/1.7M_human_brain_ROSMAP/ROSMAP.h5ad'}
adata = sc.read_h5ad(dict_adata_file[data_name])
log_time(step='load_data')

if gpu_after_pca:
    # qc
    sc.pp.calculate_qc_metrics(adata)
    sc.pp.filter_genes(adata, min_cells=3)
    log_time(step='qc')
    
    # normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    log_time(step='normalize')
    gc.collect() ##################################
    # hvg
    sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor="seurat_v3", subset=True)
    # adata = adata[:, adata.var["highly_variable"]]
    log_time(step='hvg')
    
    gc.collect()
    
    # pca
    sc.pp.pca(adata, n_comps=50)
    # sc.pp.pca(adata)
    log_time(step='pca')
    
    # data to gpu
    rsc.get.anndata_to_GPU(adata)
    log_time(step='data_to_gpu')

else:
    # data to gpu
    rsc.get.anndata_to_GPU(adata)
    log_time(step='data_to_gpu')
    
    # qc
    rsc.pp.flag_gene_family(adata, gene_family_name="MT", gene_family_prefix="mt-")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["MT"])
    rsc.pp.filter_genes(adata, min_count=3)
    log_time(step='qc')
    
    # normalize
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    log_time(step='normalize')
    
    # hvg
    rsc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]]
    log_time(step='hvg')
    
    # pca
    rsc.pp.pca(adata, n_comps=50)
    log_time(step='pca')

#### remove adata.X
#### cell QC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111

# harmony
adata.obsm['X_pca'] = adata.obsm['X_pca'] * -1
rsc.pp.harmony_integrate(adata, key="sampleID", basis='X_pca', adjusted_basis='X_pca_harmony')
log_time(step='harmony')

# neighbors
rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=50, use_rep='X_pca_harmony')
log_time(step='neighbors')

# umap
rsc.tl.umap(adata)
adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
log_time(step='umap')

# leiden cluster
rsc.tl.leiden(adata, resolution=0.6)
log_time(step='leiden')

start_time = steps_t[0]
steps_t = [t-start_time for t in steps_t]
steps_tcost = steps_t.copy()
if len(steps_t)>1:
    for i in range(1, len(steps_t)):
        steps_tcost[i] = steps_t[i] - steps_t[i-1]

df_report = pd.DataFrame({'step':steps, 'time':steps_t, 'time_cost':steps_tcost})
df_report = df_report.round(3)
df_report.to_csv(os.path.join(out_dir, f'time.cost__gpu__after.pca.{gpu_after_pca}_mm.{m_memory}_pool.alc.{pool_alc}__{data_name}.csv'), index=False)

# correct leiden
df_tmp = adata.obs.leiden.value_counts()
old2new = {df_tmp.index[i]:str(i) for i in range(df_tmp.shape[0])}
adata.obs['leiden'] = adata.obs['leiden'].astype(str).apply(lambda x: old2new[x])

# adata.write(f'{data_name}_harmony_gpu.h5ad')