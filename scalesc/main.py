import os
import cupy as cp 
import numpy as np 
import scanpy as sc
import rapids_singlecell as rsc
import util
from kernels import *
from time import time 
import logging

# create logger
logger = logging.getLogger("scaleSC")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def log(msg, level=None, verbose=True):
    if verbose:
        if level == 'warning':
            logger.warning(msg)
        elif level == 'info':
            logger.info(msg)
        elif level == 'error':
            logger.error(msg)
        else:
            logger.debug(msg)
        

def write_to_disk(adata, output_dir, data_name, batch_name=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if batch_name is None:
        out_name = f'{output_dir}/{data_name}.h5ad'
    else:
        out_name = f'{output_dir}/{data_name}.{batch_name}.h5ad'
    adata.write_h5ad(out_name)


def main(data_dir,                  # data dir containing multiple .h5ad files 
         min_genes_per_cell=200,    # minimum number of genes 
         max_genes_per_cell=6000,   # maximum number of genes
         min_cells_per_gene=3,      # minimum number of cells
         n_top_genes=4000,          # number of hvgs 
         n_components=50,           # number of components in PCA
         preload_on_cpu=True,       # if True, pre-load data in RAM
         preload_on_gpu=False,      # if True, pre-load data in GRAM
         gpus=None,                 # a list of GPU ids. [0] as default
         max_cell_batch=100000,     # the maximum number of cells in a single batch. improper setting will result in too large indptr in csr_matrix
         n_init = 10,               # number of init in neighbors
         n_neighbors = 20,          # number of neighbors
         resolution = 0.5,          # resolution in leiden
         random_state = 42,         # random state, not guranteed to be the same
         save_raw_counts = False,   # save raw counts matrix in batches
         save_norm_counts = False,  # save normalized counts matrix in batches
         save_after_each_step = False,  # save .h5ad after each step
         output_dir = 'results',     # output dir
         verbose = True, 
         logger_level = 'info',        
         ):       
    data_name = data_dir.split('/')[-1]

    # ------------------  data prepare  ------------------------
    STEP = 'Data Loading'
    log(f'{STEP}:\tstart')
    start_load = time()
    reader = util.AnnDataBatchReader(data_dir=data_dir, 
                                     preload_on_cpu=preload_on_cpu, 
                                     preload_on_gpu=preload_on_gpu, 
                                     gpus=gpus, 
                                     max_gpu_memory_usage=10*2**30,
                                     max_cell_batch=max_cell_batch) 
    end_load = time()
    # print(f'[Data loading]:\tfinish in {(end_load-start_load):.2f}s')
    log(f'{STEP}:\tfinish in {(end_load-start_load):.2f}s', level='info', verbose=verbose)
    # ----------------------------------------------------------

    # -----------------  qc & filtering  -----------------------
    STEP = 'Filtering'
    log(f'{STEP}:\tstart', level='debug', verbose=verbose)
    start_filter = time()
    cell_names = []
    cells_filter = []
    genes_counts = []
    for i, d in enumerate(reader.batchify(axis='cell')):
        rsc.pp.calculate_qc_metrics(d)
        cells_index = util.filter_cells(d, qc_var='n_genes_by_counts', min_count=min_genes_per_cell, max_count=max_genes_per_cell)
        cells_filter.append(cells_index)
        cell_names += d.obs.index[cells_index].tolist()
        genes_counts.append(d.var.n_cells_by_counts)
    gene_total_counts = np.sum(genes_counts, axis=0)
    genes_filter = gene_total_counts >= min_cells_per_gene
    reader.set_cells_filter(cells_filter)
    reader.set_genes_filter(genes_filter)
    log(f'{STEP}:\thas filtered out {reader.n_cell_origin - reader.n_cell} cells, {reader.n_gene_origin - reader.n_gene} genes', verbose=verbose)
    log(f'{STEP}:\tshape is {reader.shape}', verbose=verbose)
    end_filter = time()
    log(f'{STEP}:\tfinish in {(end_filter-start_filter):.2f}s', level='info', verbose=verbose)
    # -----------------------------------------------------------


    # ------------------  hvg (seurat_v3) -----------------------
    STEP = 'HVG'
    log(f'{STEP}:\tstart', verbose=verbose)
    start_hvg = time()
    from skmisc.loess import loess
    N = reader.shape[0]
    M = reader.shape[1]
    _sum_x = cp.zeros([M], dtype=cp.float64)
    _sum_x_sq = cp.zeros([M], dtype=cp.float64)
    for d in reader.batchify(axis='cell'):
        X_batch = d.X
        x_sum, x_sq_sum = util.get_mean_var(X_batch, axis=0)
        _sum_x += x_sum
        _sum_x_sq += x_sq_sum
    mean = _sum_x / N
    var = (_sum_x_sq / N - mean**2) * N / (N-1)  
    estimate_var = cp.zeros(M, dtype=cp.float64)
    x = cp.log10(mean[var > 0])
    y = cp.log10(var[var > 0])
    model = loess(x.get(), y.get(), span=0.3, degree=2) # fix span and degree here
    model.fit()
    estimate_var[var > 0] = model.outputs.fitted_values
    std = cp.sqrt(10**estimate_var)  # TODO: problematic, double check later!!!
    clip_val = std * cp.sqrt(N) + mean
    squared_batch_counts_sum = cp.zeros(clip_val.shape, dtype=cp.float64)
    batch_counts_sum = cp.zeros(clip_val.shape, dtype=cp.float64)
    for d in reader.batchify(axis='cell'):
        # batch_counts = cp.sparse.csr_matrix(d.X, dtype=cp.float32)
        batch_counts = d.X
        x_sq = cp.zeros_like(squared_batch_counts_sum, dtype=cp.float64)
        x = cp.zeros_like(batch_counts_sum, dtype=cp.float64)
        seurat_v3_elementwise_kernel(batch_counts.data, batch_counts.indices, clip_val, x_sq, x)
        squared_batch_counts_sum += x_sq
        batch_counts_sum += x 
    """
        ** is not correct here
        z = (x-m) / s
        var(z) = E[z^2] - E[z]^2
        E[z^2] = E[x^2 - 2xm + m^2] / s^2
        E[z] = E[x-m] / s
        x is the truncated value x by \sqrt N. m is the mean before trunction, s is the estimated std
        E[z]^2 is supposed to be close to 0.
    """   
    e_z_sq = (1 / ((N - 1) * cp.square(std))) *\
                    (N*cp.square(mean) + squared_batch_counts_sum - 2*batch_counts_sum*mean)
    e_sq_z = (1 / cp.square(std) / (N-1)**2) *\
                    cp.square((squared_batch_counts_sum - N*mean))
    print('ezsq', e_z_sq)
    print('esqz', e_sq_z)
    norm_gene_var = e_z_sq         
    ranked_norm_gene_vars = cp.argsort(cp.argsort(-norm_gene_var))
    genes_hvg_filter = (ranked_norm_gene_vars < n_top_genes).get() 
    reader.set_genes_filter(genes_hvg_filter, update=False) # do not update data, since normalization needs to be performed on all genes after filtering.
    log(f'{STEP}:\t{reader.shape[1]} genes are selected', verbose=verbose)
    end_hvg = time()
    log(f'{STEP}:\tfinish in {(end_hvg - start_hvg):.2f}s', level='info', verbose=verbose)
    # --------------------------------------------------------------


    # ---------------------- Norm & PCA ----------------------------
    """
        (X-m).T @ (X-m) = X.T @ X - m.T @ X - X.T @ m + m.T @ m  
    """
    STEP = 'Norm & PCA'
    log(f'{STEP}:\tstart', verbose=verbose)
    start_norm_pca = time.time()
    cov = cp.zeros((n_top_genes, n_top_genes), dtype=cp.float64)
    s = cp.zeros((1, n_top_genes), dtype=cp.float64)
    for i, d in enumerate(reader.batchify(axis='cell')):  # the first loop is used to calculate mean and X.TX
        if save_raw_counts:
            write_to_disk(d, output_dir=f'{output_dir}/raw_counts', data_name=data_name, batch_name=f'batch_{i}')
        rsc.pp.normalize_total(d, target_sum=1e4)
        rsc.pp.log1p(d) 
        if save_norm_counts:
            write_to_disk(d, output_dir=f'{output_dir}/norm_counts', data_name=data_name, batch_name=f'batch_{i}')
        d = d[:, genes_hvg_filter].copy()   # use all genes to normalize instead of hvgs
        X = d.X.toarray() 
        cov += cp.dot(X.T, X)  
        s += X.sum(axis=0, dtype=cp.float64) 
    m = s / N
    cov_norm = cov - cp.dot(m.T, s) - cp.dot(s.T, m) + cp.dot(m.T, m)*N
    eigvecs = cp.linalg.eigh(cov_norm)[1][:, :-n_components-1:-1] # eig values is acsending, eigvecs[:, i] corresponds to the i-th eigvec
    eigvecs = util.svd_flip(eigvecs)
    X_pca = cp.zeros([N, n_components], dtype=cp.float64)
    start_index = 0
    for d in reader.batchify(axis='cell'):  # the second loop is used to obtain PCA projection
        rsc.pp.normalize_total(d, target_sum=1e4)
        rsc.pp.log1p(d)
        d = d[:, genes_hvg_filter].copy()
        X = d.X.toarray()
        X_pca_batch = (X-m) @ eigvecs
        end_index = min(start_index+X_pca_batch.shape[0], N)
        X_pca[start_index:end_index] = X_pca_batch
        start_index = end_index
    X_pca_cpu = X_pca.get()
    adata = reader.get_anndata_obj()
    adata.obsm['X_pca'] = X_pca_cpu
    end_norm_pca = time()
    log(f'{STEP}:\tfinish in {(end_norm_pca - start_norm_pca):.2f}s', verbose=verbose)
    if save_after_each_step:
        write_to_disk(adata, output_dir=output_dir, data_name=f'{data_name}_after_pca')


    # --------- Harmony & Neighbor & Leiden & UMAP---------------
    STEP = 'Harmony'
    start_harmony = time()
    log(f'{STEP}:\tstart')
    util.harmony(adata, key='sampleID', init_seeds='2-step', n_init=n_init)
    end_harmony = time()
    log(f'{STEP}:\tfinish in {(end_harmony-start_harmony):.2f}s', level='info', verbose=verbose)
    if save_after_each_step:
        write_to_disk(adata, output_dir=output_dir, data_name=f'{data_name}_after_harmony')

    STEP = 'Neighbor'
    start_neighbors = time()
    log(f'{STEP}:\tstart')
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_components, use_rep='X_pca_harmony', algorithm='cagra')
    end_neighbors = time()
    log(f'{STEP}:\tfinish in {(end_neighbors-start_neighbors):.2f}s', level='info', verbose=verbose)
    if save_after_each_step:
        write_to_disk(adata, output_dir=output_dir, data_name=f'{data_name}_after_neighbor')
    

    STEP = 'Leiden'
    start_leiden = time()
    log(f'{STEP}:\tstart')
    rsc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    util.correct_leiden(adata)
    end_leiden = time()
    log(f'{STEP}:\tfinish in {(end_leiden-start_leiden):.2f}s', level='info', verbose=verbose)
    if save_after_each_step:
        write_to_disk(adata, output_dir=output_dir, data_name=f'{data_name}_after_leiden')
    

    STEP = 'UMAP'
    start_umap = time()
    log(f'{STEP}:\tstart')
    rsc.tl.umap(adata, random_state=random_state)
    end_umap = time()
    log(f'{STEP}:\tfinish in {(end_umap-start_umap):.2f}s', level='info', verbose=verbose)
    if save_after_each_step:
        write_to_disk(adata, output_dir=output_dir, data_name=f'{data_name}_after_umap')

    # ---------------------------------------------------------

    # adata.write_h5ad(f'results/{data_name}_processed.h5ad')
    # print(f'save to "results/{data_name}_processed.h5ad"')



if __name__ == '__main__':
    # -------------- enable RMM -----------------
    # import rmm
    # from rmm.allocators.cupy import rmm_cupy_allocator
    # rmm.reinitialize(managed_memory=True)
    # cp.cuda.set_allocator(rmm_cupy_allocator)
    # print('RMM enabled')
    # -------------------------------------------

    # -------------- set GRAM limit --------------
    # mempool = cp.get_default_memory_pool()
    # with cp.cuda.Device(0):
    #     mempool.set_limit(size=1024**3)  # 1 GiB, but it doesn't work
    # with cp.cuda.Device(0):
    #     mempool.set_limit(size=1024**3) 
    # ---------------------------------------------

    # --------------- Test -------------------------
    # main('../data_dir/2.5M_new')
    # main('../data_dir/13M_fake_4', preload_on_cpu=True, preload_on_gpu=False)
    main('../data_dir/70k_human_lung', preload_on_cpu=True, preload_on_gpu=True, save_norm_counts=True, save_raw_counts=True)
    # main('../data_dir/1.3M_mouse_brain', preload_on_cpu=True, preload_on_gpu=True)
    



















