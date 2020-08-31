# GraphSCC
Accurately Clustering Single-cell RNA-seq data by Capturing
Structural Relations between Cells through Graph Convolutional
Network


# Dataset
The datasets we used in this study can be available at
https://hemberg-lab.github.io/scRNA.seq.datasets/


#Preprocessing 
 For the simulated datasets, we normalized
them using transcripts per million (TPM) method [41] and then
scaled the value of each gene to [0, 1]. For real datasets, we
employed the procedure suggested by Seurat3.0 to normalize
and select top 2000 highly variable genes for scRNA-seq data,
then to scale the value of each gene to [0,1]. Note that for real
datasets normalized by FPKM, we first converted them to TPM.

# Usage
```
python GraphSCC.py --name [goolam|baron_mouse]
```

