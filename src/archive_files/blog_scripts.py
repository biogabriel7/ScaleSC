# https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/

import scanpy as sc
import rapids_singlecell as rsc
adata = sc.read("PATH TO DATASET")
cudata = rsc.cunnData.cunnData(adata=adata) 

# Basic QC rapids-singlecell
rsc.pp.flag_gene_family(cudata,gene_family_name="MT", gene_family_prefix="mt-")
rsc.pp.calculate_qc_metrics(cudata,qc_vars=["MT"])
cudata = cudata[cudata.obs["n_genes_by_counts"] > 500]
cudata = cudata[cudata.obs["pct_counts_MT"] < 20]
rsc.pp.filter_genes(cudata,min_count=3)

# log normalization and highly variable gene selection
cudata.layers["counts"] = cudata.X.copy()
rsc.pp.normalize_total(cudata,target_sum=1e4)
rsc.pp.log1p(cudata)
rsc.pp.highly_variable_genes(cudata,n_top_genes=5000,flavor="seurat_v3",layer = "counts")
cudata = cudata[:,cudata.var["highly_variable"]==True]

# Regression, scaling and PCA
rsc.pp.regress_out(cudata,keys=["total_counts", "pct_counts_MT"])
rsc.pp.scale(cudata,max_value=10)
rsc.pp.pca(cudata, n_comps = 100)

sc.pl.pca_variance_ratio(cudata, log=True,n_pcs=100)

adata_preprocessed = cudata.to_AnnData()

