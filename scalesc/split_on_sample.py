import scanpy as sc
import h5py as h5
import sys 
import os 


DATA_DIR = './'
data_name = sys.argv[1]
dict_adata_file = {
                    '70k_human_lung':'/edgehpc/dept/compbio/projects/scaleSC/data/70k_human_lung/krasnow_hlca_10x.sparse.h5ad', # 1G
                   '1.3M_mouse_brain':'/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/1M_brain_cells_10X.sparse.h5ad', #30G
                   '1.7M_human_brain_ROSMAP':'/edgehpc/dept/compbio/projects/scaleSC/data/1.7M_human_brain_ROSMAP/ROSMAP.h5ad', # 69.05G
                  '2.5M_human_brain_Linnarsson':'/edgehpc/dept/compbio/projects/scaleSC/data/2.5M_human_brain_Linnarsson/Linnarsson_neuron.h5ad', # 127G
                    '2.5M_new': '/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/2M_cells_extended.from1.3M.h5ad',
                    '13M_fake': '/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/13M_cells_extended.from1.3M.h5ad',
                    '13M_fake_2': '/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/fake_data_copy/13M_brain_gene.unique_fake_seed0-9.h5ad',
                    '13M_fake_3': '/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/fake_data_copy/13M_brain_gene.unique_fake_seed0-9.h5ad',
                    '13M_fake_4': '/edgehpc/dept/compbio/projects/scaleSC/data/1.3M_mouse_brain/fake_data_only_clean_clusters/13M_brain_gene.unique_fake_seed0-49.h5ad'
                  } 


def split_by_sample(adata):
    sample_ids = adata.obs['sampleID'].unique().tolist()
    for sample_id in sample_ids:
        sample = adata[adata.obs['sampleID'] == sample_id, :]
        if not os.path.exists(f'{DATA_DIR}/{data_name}'):
            os.makedirs(f'{DATA_DIR}/{data_name}')
        sample.write_h5ad(f'{DATA_DIR}/{data_name}/{data_name}_{sample_id}.h5ad')



if __name__ == '__main__':
    if data_name not in dict_adata_file:
        raise Exception(f"dataset should be in {','.join(list(dict_adata_file.keys()))}")
    adata = sc.read_h5ad(dict_adata_file[data_name])
    data_size = sys.getsizeof(adata) / 1024 / 1024/ 1024 
    print(f'{data_name}: {adata.shape}, {data_size:.4f} G')
    res = split_by_sample(adata)
    print('done')
    