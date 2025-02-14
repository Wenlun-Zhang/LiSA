# LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection

## 1. Overview

* `./module`: Includes models of LiSA, GraphCopy, NIA baseline, and related surrogate models.
* `./run`: Includes experiment logs of results reported in this paper.
* `./script`: Includes scripts to reproduce results in this paper.
* `./tools`: Some common functions used in this project.

## 2. Environments

```
python==3.11.5
torch==2.2.1
torch-geometric==2.5.0
numpy==1.24.3
scipy==1.11.1
scikit-learn==1.3.0
```

## 3. LiSA

### Abstract

[Paper](https://arxiv.org/abs/2502.09271)

Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

### Play with LiSA

You can try LiSA in different settings by modifying parameters in main code and run:

```
python3 module/lisa.py
```

### Reproduce the Results

The hyperparameters and random seeds are set exactly the same as to reproduce the results in this paper. Please run:

```
python3 script/lisa_cora.py
```

## Datasets

The experiments are validated on four public real-word datasets: Cora, Amazon-Photo, Amazon-Computers, PubMed, and FacebookPagePage, which can be downloaded through torch-geometric API.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{LiSA,
  title={LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection},
  author={Zhang, Wenlun and Dai, Enyan and Yoshioka, Kentaro},
  journal={arXiv:2502.09271},
  year={2025}
}
```
