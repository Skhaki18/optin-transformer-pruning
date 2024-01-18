# ICLR 2024 The Need for Speed: Pruning Transformers with One Recipe [[Project Page]](http://www.samirkhaki.com/optin-transformer-pruning/) [[OpenReview]](https://openreview.net/forum?id=MVmT6uQ3cQ)

## News

**Code will be released soon! **

## Abstract

We introduce the **O**ne-shot **P**runing **T**echnique for **I**nterchangeable **N**etworks (OPTIN) framework as a tool to increase the efficiency of pre-trained transformer architectures, across many domains, without requiring re-training. Recent works have explored improving transformer efficiency, however often incur computation- ally expensive re-training procedures or depend on architecture-specific character- istics, thus impeding practical wide-scale adoption across multiple modalities. To address these shortcomings, the OPTIN framework leverages intermediate feature distillation, capturing the long-range dependencies of model parameters (coined trajectory), to produce state-of-the-art results on natural language, image classifica- tion, transfer learning, and semantic segmentation tasks. Our motivation stems from the need for a generalizable model compression framework that scales well across different transformer architectures and applications. Given a FLOP constraint, the OPTIN framework will compress the network while maintaining competitive accuracy performance and improved throughput. Particularly, we show a ≤ 2% accuracy degradation from NLP baselines and a 0.5% improvement from state- of-the-art methods on image classification at competitive FLOPs reductions. We further demonstrate the generalization of tasks and architecture with comparative performance on Mask2Former for semantic segmentation and cnn-style networks. OPTIN presents one of the first one-shot efficient frameworks for compressing transformer architectures that generalizes well across _multiple class domains_, in particular: natural language and image-related tasks, without _re-training_.