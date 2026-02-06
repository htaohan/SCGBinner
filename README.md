# SCGBinner
Metagenomic analysis tool
## Install SCGBinner
Install the dependecies of SCGBinner
```
conda create -n SCGBinner
conda activate SCGBinner
mamba install biopython numpy=1.19 scipy igraph leidenalg joblib pandas=1.4 scikit-learn pyyaml tensorboard tqdm hnswlib atomicwrites bedtools
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
Install SCGBinner
```
git clone https://github.com/htaohan/SCGBinner.git
cd SCGBinner
pip install .
```
