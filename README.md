# scMGCA
Single-cell RNA sequencing (scRNA-seq) provides high-throughput gene expression information to explore cellular heterogeneity at the individual cell level. A major challenge in characterizing high-throughput gene expression data arises from the curse of dimensionality, and the prevalence of dropout events. To address these concerns, we developed a single-cell clustering method (scMGCA) based on a graph-embedding autoencoder that simultaneously learns cell–cell topology representation and cluster assignments. In scMGCA, we propose a graph convolutional autoencoder to preserve the topological information of cells from the embedded space in multinomial distribution, and employs the positive pointwise mutual information (PPMI) matrix for cell graph augmentation. Experiments show that scMGCA is accurate and effective for cell segregation and superior to other state-of-the-art models across multiple platforms. In addition, we perform genomic interpretation on the key compressed transcriptomic space of the graph-embedding autoencoder to demonstrate the underlying gene regulation mechanism. In a pancreatic ductal adenocarcinoma (PDAC) dataset, with 8921 individual pancreatic cells from primary PDAC tumors and control pancreases, scMGCA successfully provided annotations on the specific cell types and revealed differential gene expression levels across multiple tumor-associated and cell signalling pathways in PDAC progression through single-cell trajectory and gene set enrichment analysis.



![image](https://user-images.githubusercontent.com/65069252/158551648-e833c769-4396-4e26-957c-f38996c8c68d.png)




## Installation

### pip

```
$ pip install -r requirements
$ pip install scMGCA
```


## Usage

You can run the scTAG from the command line:

```
$ from scMGCA.run import train
$ train(data,highly_genes=500,pretrain_epochs=1000,maxiter=300)
```



## Example

```
$ from scMGCA.run import train
$ data = './dataset/Quake_10x_Limb_Muscle/data.h5'
$ train(data,highly_genes=500,pretrain_epochs=1000,maxiter=300)
```

The result will give NMI and ARI



## Arguments

|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
|      data       | A h5 file. Contains a matrix of scRNA-seq expression values,true labels, and other information. By default, genes are assumed to be represented by columns and samples are assumed to be represented by rows. |
|  highly genes   | Number of genes selected                                     |
| pretrain epochs | Number of pretrain epochs                                    |
|     maxiter     | Number of training epochs                                    |

