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
SCGBinner is recommended to be run in a GPU environment.
```
conda activate SCGBinner
########################## Run SCGBinner using single-coverage ##########################
scgbinner -a contig_file.fa -o output_path -b S1.sorted.bam -t 16

########################## Run SCGBinner using multi-coverage ##########################
scgbinner -a contig_file.fa -o output_path -b "S1.sorted.bam S2.sorted.bam" -t 16

Alternatively, using wildcard expansion:
scgbinner -a contig_file.fa -o output_path -b "*.sorted.bam" -t 16
```
## Output
The MAGs are output to the scgbinner_res/SCGBINNER_result directory.
## Options
```
Options:
  -a STR          metagenomic assembly file
  -o STR          output directory
  -b STR          bam files
  -t INT          number of threads (default=16)
  -n INT          number of views for contrastive multiple-view learning (default=6)
  -l FLOAT        temperature in loss function (default=0.07 for assemblies with an N50 > 10000, default=0.15 for others)
  -e INT          embedding size for comebined network (default=2048)
  -c INT          embedding size for coverage network (default=2048)
  -p INT          standard batch size (default=1024)
```
## References
[1] Wang Z, You R, Han H, et al. Effective binning of metagenomic contigs using contrastive multi-view representation learning[J]. Nature Communications, 2024, 15(1): 585.
[2] Pan S, Zhao X M, Coelho L P. SemiBin2: self-supervised contrastive learning leads to better MAGs for short-and long-read sequencing[J]. Bioinformatics, 2023, 39(Supplement_1): i21-i29.
[3] Liu C C, Dong S S, Chen J B, et al. MetaDecoder: a novel method for clustering metagenomic contigs[J]. Microbiome, 2022, 10(1): 46.
[4] Han H, Wang Z, Zhu S. Benchmarking metagenomic binning tools on real datasets across sequencing platforms and binning modes[J]. Nature Communications, 2025, 16(1): 2865.
