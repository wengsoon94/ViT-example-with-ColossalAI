## Overview

This repository is reproduce the ViT example in https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/vit 

@@@@@@@@@@@@@@@@@@@@

You will finetune a a ViT-base model on this dataset, with more than 8000 images of bean leaves. This dataset is for image classification task and there are 3 labels: ['angular_leaf_spot', 'bean_rust', 'healthy'].

Objective: The goal of this assignment is to gain hands-on experience with Colossal-AI, a powerful tool for distributed training of large-scale AI models. You will reproduce one example from the Colossal-AI example repository and run the experiment in your own environment (single GPU environment like Colab is ok).

Instructions:

Visit the Colossal-AI examples directory at https://github.com/hpcaitech/ColossalAI/tree/main/examplesLinks to an external site. and select one specific example to reproduce. For instance, you could choose to fine-tune GPT2 using the code from https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.pyLinks to an external site..
Adapt the code to run in your environment by modifying parallel size, batch size, model hyperparameters, or any other necessary parts. Ensure that the code runs successfully.
Create a public repository on GitHub containing the example you experimented with. The repository name should include the model information and "Colossal-AI" (e.g., "GPT2 fine-tuning with ColossalAI").
In the repository's README.md file, provide the following information:
The model used in the experiment
The dataset employed
Parallel settings (if any)
Instructions on how to run your code
Experiment results, presented in a table or figure
Include a requirements.txt file in the repository that lists all the dependencies needed to run your code.
Ensure that the repository contains all the code you used and the experiment logs.
Submit a zip file named "StuID.zip" (replace "StuID" with your actual student ID, e.g., "A0123456J.zip") to Canvas under "Assignments -> Assignment6". Note that you should use your student ID, not your NUSNET ID. The zip file should include your Colossal-AI example repository. Also, provide the link to your GitHub repository in the submission text box.
Grading Scheme:

Creating a public repository on GitHub with the experimented example: 1 point
Appropriate repository name: 1 point
Complete information in README.md: 2 points
Including a requirements.txt file with dependencies: 1 point
Including all code and experiment logs in the repository: 4 points
Submitting the zip file with the correct naming format and providing the GitHub repository link: 1 point


Vision Transformer is a class of Transformer model tailored for computer vision tasks. It was first proposed in paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and achieved SOTA results on various tasks at that time.

In our example, we are using pretrained weights of ViT loaded from HuggingFace.
We adapt the ViT training code to ColossalAI by leveraging [Boosting API](https://colossalai.org/docs/basics/booster_api) loaded with a chosen plugin, where each plugin corresponds to a specific kind of training strategy. This example supports plugins including TorchDDPPlugin (DDP), LowLevelZeroPlugin (Zero1/Zero2), GeminiPlugin (Gemini) and HybridParallelPlugin (any combination of tensor/pipeline/data parallel).

## Run Demo

By running the following script:
```bash
bash run_demo.sh
```
You will finetune a a [ViT-base](https://huggingface.co/google/vit-base-patch16-224) model on this [dataset](https://huggingface.co/datasets/beans), with more than 8000 images of bean leaves. This dataset is for image classification task and there are 3 labels: ['angular_leaf_spot', 'bean_rust', 'healthy'].

The script can be modified if you want to try another set of hyperparameters or change to another ViT model with different size.

The demo code refers to this [blog](https://huggingface.co/blog/fine-tune-vit).



## Run Benchmark

You can run benchmark for ViT model by running the following script:
```bash
bash run_benchmark.sh
```
The script will test performance (throughput & peak memory usage) for each combination of hyperparameters. You can also play with this script to configure your own set of hyperparameters for testing.
