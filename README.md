# ICLR 2024 The Need for Speed: Pruning Transformers with One Recipe [[Project Page]](http://www.samirkhaki.com/optin-transformer-pruning/) [[OpenReview]](https://openreview.net/forum?id=MVmT6uQ3cQ)

## News
- [2024/04/08] Initial Codebase for Language Experiments has been released; More coming soon!
- [2024/01/16] OPTIN Transformer Pruning is accepted at ICLR 2024

## Abstract

We introduce the **O**ne-shot **P**runing **T**echnique for **I**nterchangeable **N**etworks (OPTIN) framework as a tool to increase the efficiency of pre-trained transformer architectures, across many domains, without requiring re-training. Recent works have explored improving transformer efficiency, however often incur computation- ally expensive re-training procedures or depend on architecture-specific character- istics, thus impeding practical wide-scale adoption across multiple modalities. To address these shortcomings, the OPTIN framework leverages intermediate feature distillation, capturing the long-range dependencies of model parameters (coined trajectory), to produce state-of-the-art results on natural language, image classifica- tion, transfer learning, and semantic segmentation tasks. Our motivation stems from the need for a generalizable model compression framework that scales well across different transformer architectures and applications. Given a FLOP constraint, the OPTIN framework will compress the network while maintaining competitive accuracy performance and improved throughput. Particularly, we show a ≤ 2% accuracy degradation from NLP baselines and a 0.5% improvement from state- of-the-art methods on image classification at competitive FLOPs reductions. We further demonstrate the generalization of tasks and architecture with comparative performance on Mask2Former for semantic segmentation and cnn-style networks. OPTIN presents one of the first one-shot efficient frameworks for compressing transformer architectures that generalizes well across _multiple class domains_, in particular: natural language and image-related tasks, without _re-training_.


## TODO
- [x] Initial Language-based Code Release
- [ ] BERT Model Saved Parameter Rankings
- [ ] Initial Vision-based Code Release
- [ ] Vision Model Saved Parameter Rankings
- [ ] Cleaning/Finalizing OPTIN Code Release

## Getting Started

```bash
Create virtual env with conda/pyvenv -- python >= 3.11
pip install -r requirements.txt
```

## Running Instructions

```bash
python main.py --config path/to/config.yaml
```

## Relevant Functions
For any of the applications, the core-pruning loss functions are implemented in [loss_components.py](\prune\loss_components.py). These functions are wrapped in the _head pruning_ and _neuron pruning_ functions under ./prune/*. Sample config.yaml files have been provided to reproduce our results. Ablative components discussed in the main paper can mostly be mostly tested by modifying the specs in these config.yaml files.

## Language Experimental Results:

| Data | Performance | Saved Paramter Rankings |
|----------------------|:----------:|:----------:|
| MNLI | 81.90 | [link]( ) |
| QQP | 90.06 | [link]( ) |
| QNLI | 88.49 | [link]( ) |
| SST | 92.24 | [link]( ) |
| STS-B | 87.25 | [link]( ) |
| MRPC | 85.13 | [link]( ) |
<p align="center">
<b> Table1: Summary of All main language experiments with BERT using the OPTIN Transformer Prunnig Scheme.</b> FLOP compression ratios are described in the main paper, pre-saved parameter rankng will be made available soon. 
</p>


## Citation
```
@inproceedings{khaki2024the,
    title={The Need for Speed: Pruning Transformers with One Recipe},
    author={Samir Khaki and Konstantinos N Plataniotis},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=MVmT6uQ3cQ}
}
```