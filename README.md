<h1 align="center"><b>awesome-graph-OOD</b></h1>
<p align="center">
    <a href="https://github.com/THUMNLab/awesome-graph-ood/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs"></a>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="awesome"></a>
    <a href="https://graph.ood-generalization.com/"><img src="https://img.shields.io/badge/-Website-grey?logo=svelte&logoColor=white" alt="Website"></a>
    <img src="https://img.shields.io/github/stars/THUMNLab/awesome-graph-ood?color=yellow&label=Star" alt="Stars" >
    <img src="https://img.shields.io/github/forks/THUMNLab/awesome-graph-ood?color=blue&label=Fork" alt="Forks" >
</p>

This repository contains the paper list of **Graph Out-of-Distribution (OOD) Generalization**. The existing literature can be summarized into three categories from conceptually different perspectives, i.e., *data*, *model*, and *learning strategy*, based on their positions in the graph machine learning pipeline. 
For more details, please refer to our survey paper: [Out-Of-Distribution Generalization on Graphs: A Survey.](https://arxiv.org/pdf/2202.07987.pdf) 

We will try our best to make this paper list updated. If you notice some related papers missing, do not hesitate to contact us via pull requests at our repo.

# Papers

## Data

#### Graph Data Augmentation

- [ICML 2022] G-Mixup: Graph Data Augmentation for Graph Classification [[paper]](https://arxiv.org/pdf/2202.07179.pdf)
- [KDD 2022] Graph Rationalization with Environment-based Augmentations [[paper]](https://arxiv.org/pdf/2206.02886.pdf)
- [NeurIPS 2021] Metropolis-Hastings Data Augmentation for Graph Neural Networks [[paper]](https://proceedings.neurips.cc/paper/2021/file/9e7ba617ad9e69b39bd0c29335b79629-Paper.pdf)
- [AAAI 2021] Data Augmentation for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2006.06830.pdf)
- [CVPR 2022] Robust Optimization as Data Augmentation for Large-scale Graphs [[paper]](https://arxiv.org/pdf/2010.09891.pdf)
- [NeurIPS 2020] Graph Random Neural Network for Semi-Supervised Learning on Graphs [[paper]](https://arxiv.org/pdf/2005.11079.pdf)
- [ICLR 2020] DropEdge: Towards Deep Graph Convolutional Networks on Node Classification [[paper]](https://arxiv.org/pdf/1907.10903.pdf)

## Model

#### Disentanglement-based Graph Models

- [TKDE 2022] Disentangled Graph Contrastive Learning With Independence Promotion [[paper]](https://haoyang.li/assets/pdf/2022_TKDE_IDGCL.pdf)
- [NeurIPS 2022] Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure [[paper]](https://arxiv.org/pdf/2209.14107.pdf)
- [NeurIPS 2021] Disentangled Contrastive Learning on Graphs [[paper]](https://openreview.net/pdf?id=C_L0Xw_Qf8M)
- [AAAI 2020] Independence Promoted Graph Disentangled Networks [[paper]](https://arxiv.org/pdf/1911.11430.pdf)
- [NeurIPS 2020] Factorizable graph convolutional networks [[paper]](https://arxiv.org/pdf/2010.05421.pdf)
- [KDD 2020] Interpretable deep graph generation with node-edge co-disentanglement [[paper]](https://arxiv.org/pdf/2006.05385.pdf)
- [ICML 2019] Disentangled Graph Convolutional Networks [[paper]](http://proceedings.mlr.press/v97/ma19a/ma19a.pdf)
- [ICANN 2018] GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders [[paper]](https://arxiv.org/pdf/1802.03480.pdf)
- [NeurIPS Workshop 2016] Variational Graph Auto-Encoders [[paper]](https://arxiv.org/pdf/1611.07308.pdf)

#### Causality-based Graph Models

- [TKDE 2022] OOD-GNN: Out-of-Distribution Generalized Graph Neural Network [[paper]](https://arxiv.org/abs/2112.03806) 
- [NeurIPS 2022] OOD Link Prediction Generalization Capabilities of Message-Passing GNNs in Larger Test Graphs [[paper]](https://arxiv.org/pdf/2205.15117.pdf)
- [ICML 2022] Learning from Counterfactual Links for Link Prediction
 [[paper]](https://arxiv.org/abs/2106.02172)
- [KDD 2022] Causal Attention for Interpretable and Generalizable Graph Classification [[paper]](https://arxiv.org/pdf/2112.15089.pdf)
- [arXiv 2022] Deconfounding to Explanation Evaluation in Graph Neural Networks [[paper]](https://arxiv.org/pdf/2201.08802.pdf)
- [TNNLS 2022] Debiased Graph Neural Networks with Agnostic Label Selection Bias [[paper]](https://arxiv.org/abs/2201.07708)
- [arXiv 2021] Generalizing Graph Neural Networks on Out-Of-Distribution Graphs [[paper]](https://arxiv.org/abs/2111.10657) 
- [ICML 2021] Generative Causal Explanations for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2104.06643.pdf)
- [ICML 2021] Size-Invariant Graph Representations for Graph Classification Extrapolations [[paper]](https://arxiv.org/abs/2103.05045)

## Learning Strategy

#### Graph Invariant Learning

- [NeurIPS 2022] Learning Invariant Graph Representations for Out-of-Distribution Generalization [[paper]](https://haoyang.li/assets/pdf/2022_NeurIPS_GIL.pdf)
- [NeurIPS 2022] Dynamic Graph Neural Networks Under Spatio-Temporal Distribution Shift [[paper]](https://haoyang.li/assets/pdf/2022_NeurIPS_DIDA.pdf)
- [NeurIPS 2022] Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs [[paper]](https://arxiv.org/pdf/2202.05441.pdf)
- [ICML 2022] Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism [[paper]](https://arxiv.org/pdf/2201.12987.pdf)
- [arXiv 2022] Finding Diverse and Predictable Subgraphs for Graph Domain Generalization [[paper]](https://arxiv.org/pdf/2206.09345.pdf)
- [ICLR 2022] Towards Distribution Shift of Node-Level Prediction on Graphs: An Invariance Perspective [[paper]](https://openreview.net/pdf?id=FQOC5u-1egI) 
- [ICLR 2022] Discovering Invariant Rationales for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2201.12872.pdf)
- [NeurIPS 2021] Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data [[paper]](https://arxiv.org/pdf/2108.01099.pdf)
- [arXiv 2021] Stable Prediction on Graphs with Agnostic Distribution Shift [[paper]](https://arxiv.org/pdf/2110.03865.pdf)

#### Graph Adversarial Training

- [arXiv 2022] Shift-Robust Node Classification via Graph Adversarial Clustering [[paper]](https://arxiv.org/pdf/2203.15802.pdf)
- [arXiv 2021] CAP: Co-Adversarial Perturbation on Weights and Features for Improving Generalization of Graph Neural Networks [[paper]](https://arxiv.org/pdf/2110.14855.pdf) 
- [arXiv 2021] Distributionally Robust Semi-Supervised Learning Over Graphs [[paper]](https://arxiv.org/pdf/2110.10582.pdf) 
- [Openreview 2021] Adversarial Weight Perturbation Improves Generalization in Graph Neural Networks [[paper]](https://openreview.net/pdf?id=hUr6K4D9f7P) 
- [TKDE 2019] Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure [[paper]](https://arxiv.org/pdf/1902.08226.pdf)
- [ICDM 2019] Domain-Adversarial Graph Neural Networks for Text Classification [[paper]](https://shiruipan.github.io/publication/icdm-19-wu/icdm-19-wu.pdf) 

#### Graph Self-supervised Learning

- [arXiv 2022] GraphTTA: Test Time Adaptation on Graph Neural Networks [[paper]](https://arxiv.org/pdf/2208.09126.pdf)
- [WWW 2022] Confidence May Cheat: Self-Training on Graph Neural Networks under Distribution Shift [[paper]](https://arxiv.org/pdf/2201.11349.pdf)
- [arXiv 2021] Graph Self-Supervised Learning: A Survey [[paper]](https://arxiv.org/pdf/2103.00111.pdf)
- [ICML 2021] From Local Structures to Size Generalization in Graph Neural Networks [[paper]](https://arxiv.org/pdf/2010.08853.pdf) 
- [NeurIPS 2020] Graph Contrastive Learning with Augmentations [[paper]](https://arxiv.org/pdf/2010.13902.pdf)
- [ICLR 2020] Strategies for Pre-training Graph Neural Networks [[paper]](https://arxiv.org/pdf/1905.12265.pdf)
- [KDD 2020] GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training [[paper]](https://arxiv.org/pdf/2006.09963.pdf)

## Theory

- [NeurIPS 2021] Subgroup Generalization and Fairness of Graph Neural Networks [[paper]](https://arxiv.org/pdf/2106.15535.pdf)
- [NeurIPS 2021] Learning Theory Can (Sometimes) Explain Generalisation in Graph Neural Networks [[paper]](https://arxiv.org/pdf/2112.03968.pdf)
- [ICLR 2021] How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks [[paper]](https://arxiv.org/abs/2009.11848)
- [ICLR 2021] A pac-bayesian approach to generalization bounds for graph neural networks [[paper]](https://arxiv.org/pdf/2012.07690.pdf)
- [arXiv 2021] Generalization bounds for graph convolutional neural networks via Rademacher complexity [[paper]](https://arxiv.org/pdf/2102.10234.pdf)
- [ICML 2021] Graph Convolution for Semi-Supervised Classification Improved Linear Separability and Out-of-Distribution Generalization [[paper]](https://arxiv.org/pdf/2102.06966.pdf)
- [ICML 2020 WorkShop] From Graph Low-Rank Global Attention to 2-FWL Approximation [[paper]](https://grlplus.github.io/papers/92.pdf)
- [ICML 2020] Generalization and Representational Limits of Graph Neural Networks [[paper]](https://arxiv.org/pdf/2002.06157.pdf)
- [NeurIPS 2019] Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels  [[paper]](https://arxiv.org/pdf/1905.13192.pdf)
- [KDD 2019] Stability and Generalization of Graph Convolutional Neural Networks [[paper]](https://arxiv.org/pdf/1905.01004.pdf)
- [Neural Networks] The Vapnik–Chervonenkis dimension of graph and recursive neural networks [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0893608018302363)

## Other Related Papers

#### GNN Architecture

- [ICML 2022] Graph Neural Architecture Search Under Distribution Shifts [[paper]](https://proceedings.mlr.press/v162/qin22b/qin22b.pdf)
- [arXiv 2021] Learning to Pool in Graph Neural Networks for Extrapolation [[paper]](https://arxiv.org/pdf/2106.06210.pdf)
- [ICLR 2020] What Can Neural Networks Reason About? [[paper]](https://arxiv.org/pdf/1905.13211.pdf)
- [ICLR 2020] Neural Execution of Graph Algorithms [[paper]](https://arxiv.org/pdf/1910.10593.pdf)
- [NeurIPS 2019] Understanding Attention and Generalization in Graph Neural Networks [[paper]](https://arxiv.org/pdf/1905.02850.pdf)
- [arXiv 2020] Customized Graph Neural Networks [[paper]](https://arxiv.org/pdf/2005.12386.pdf)

#### Dynamic Environment

- [NeurIPS 2022] Association Graph Learning for Multi-Task Classification with Category Shifts
- [arXiv 2021] Online Adversarial Distillation for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2112.13966.pdf)
- [IJCNN 2021] Lifelong Learning of Graph Neural Networks for Open-World Node Classification [[paper]](https://arxiv.org/pdf/2006.14422.pdf)

#### Domain Knowledge

- [AAAI 2022] How Does Knowledge Graph Embedding Extrapolate to Unseen Data: a Semantic Evidence View [[paper]](https://arxiv.org/pdf/2109.11800.pdf)
- [NeurIPS 2021 Workshop] Reliable Graph Neural Networks for Drug Discovery Under Distributional Shift [[paper]](https://arxiv.org/pdf/2111.12951.pdf)
- [ICML 2020 workshop] Evaluating Logical Generalization in Graph Neural Networks [[paper]](https://arxiv.org/pdf/2003.06560.pdf)

#### Dataset

- [NeurIPS 2022] GOOD: A Graph Out-of-Distribution Benchmark [[paper]](https://openreview.net/pdf?id=8hHg-zs_p-h)
- [NeurIPS 2021 Workshop] A Closer Look at Distribution Shifts and Out-of-Distribution Generalization on Graphs [[paper]](https://openreview.net/pdf?id=XvgPGWazqRH)
- [arXiv 2022] DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations [[paper]](https://arxiv.org/pdf/2201.09637.pdf)


# Cite

Please consider citing our [survey paper](https://arxiv.org/abs/2202.07987) if you find this repository helpful:
```
@article{li2022ood,
  title={Out-of-distribution generalization on graphs: A survey},
  author={Li, Haoyang and Wang, Xin and Zhang, Ziwei and Zhu, Wenwu},
  journal={arXiv preprint arXiv:2202.07987},
  year={2022}
}
```
