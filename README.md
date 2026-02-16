# SCGBinner
Metagenomic binning method
## Install SCGBinner
Install the dependecies of SCGBinner
```
conda create -n SCGBinner python=3.9
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
## Run SCGBinner
```
conda activate SCGBinner
########################## Run SCGBinner using single-coverage ##########################
scgbinner -a contig_file.fa \
-o output_path \
-b S1.sorted.bam \
-t 16

########################## Run SCGBinner using multi-coverage ##########################
scgbinner -a contig_file.fa \
-o output_path \
-b "S1.sorted.bam S2.sorted.bam" \
-t 16

Alternatively, using wildcard expansion:
scgbinner -a contig_file.fa \
-o output_path \
-b "*.sorted.bam" \
-t 16
```
