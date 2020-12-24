## General Info
This repository includes source code for re-implementation of the paper titled "Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning" by Huang et. al. Research paper can be found at https://arxiv.org/pdf/1909.00277.pdf, and the original code can be found at https://github.com/wilburOne/cosmosqa

This implementation uses transformers library and leverages Dataloader and Dataset classes of Pytorch to make it a bit more readable and remove a lot of boilerplate code.

## Requirements
Python3, Pytorch, transformers, tqdm

## Setup
To run this project install the requirements and make sure you have at least 14GB GPU:
```
python -m cosmosqa.driver
```

## Todo
1. Use command line arguments to change training parameters e.g batch size and epochs
2. Create a central interface to make the change of base models easy. For e.g changing the base model to DistilBert from Bert should be easy.
3. Remove print logs