# scaleSC (update 10/29/24)

#### scaleSC requires a **high-end GPU** and a matching **CUDA** version to support GPU-accelerated computing. 
---
Requirements:
>   - [**RAPIDS**](https://rapids.ai/) from Nvidia
>   - [**rapids-singlecell**](https://rapids-singlecell.readthedocs.io/en/latest/index.html), an alternative of *scanpy* that employs GPU for acceleration. 
>   - [**Conda**](https://docs.conda.io/projects/conda/en/latest/index.html), version >=22.11 is strongly encoruaged, because *conda-libmamba-solver* is set as default, which 
significant speeds up solving dependencies.  
>   - [**pip**](), a python package installer.

Environment Setup:
1. Install [**RAPIDS**](https://rapids.ai/) through Conda, \
    `conda create -n scalesc -c rapidsai -c conda-forge -c nvidia
    rapids=24.10 python=3.10 'cuda-version>=11.4,<=11.8'` \
    Users have flexibility to install it according to their systems by using this [online selector](https://docs.rapids.ai/install/?_gl=1*1em94gj*_ga*OTg5MDQyNDkyLjE3MjM0OTAyNjk.*_ga_RKXFW6CM42*MTczMDIxNzIzOS4yLjAuMTczMDIxNzIzOS42MC4wLjA.#selector).

2. Activate conda env, \
    `conda activate scalesc`
3. Install [**rapids-singlecell**](https://rapids-singlecell.readthedocs.io/en/latest/index.html) using pip, \
    `pip install rapids-singlecell` 

4. Install scaleSC, \
    - pull scaleSC from github \
        `git clone https://github.com/interactivereport/scaleSC.git`  \
    - enter the folder and install scaleSC \
        `cd scaleSC` \
        `pip install .`
5. check env:
    - `python -c "import scalesc; print(scalesc.__version__)"` == 0.1.0
    - `python -c "import cupy; print(cupy.__version__)"` >= 13.3.0
    - `python -c "import cuml; print(cuml.__version__)"` >= 24.10
    - `python -c "import cupy; print(cupy.cuda.is_available())"` = True
    -  `python -c "import xgboost; print(xgboost.__version__)` >= 2.1.1, optionally for marker annotation.

Tutorial:
    see ipynb notebook in github.


    