# TF Binding Sites Model Repository

Welcome to the TF (Transcription Factor) Binding Sites Model repository. This repository contains resources, scripts, and guidelines for both pretraining and fine-tuning a model for predicting TF binding sites.

## Overview

In this repository, you'll find a structured process to:

1. **Pretrain the Model**: This involves preprocessing raw BAM and narrowPeak files and then training the model using the provided scripts. 
2. **Fine-tune the Model**: After pretraining, the model can be fine-tuned using a specific dataset from the UCSF server to refine its predictions.

## Directory Structure
```
.
├── pretraining
│ ├── preprocessing script
│ └── training script
├── finetuning
│ ├── preprocessing script
│ └── training script
```

## Quick Start

1. **Pretrain Your Model**: For a step-by-step guide on pretraining your model, please refer to the [Pretraining Guide](./pretraining/README.md).
2. **Fine-tune Your Model**: For instructions on fine-tuning, head over to the [Fine-tuning Guide](./finetuning/README.md).

## Additional Resources

- **AWS S3 Buckets**:
  - Pretraining Data: `s3://tf-binding-sites/pretraining`
  - Fine-tuning Data: `s3://tf-binding-sites/finetuning`

- **UCSF Server Dataset Directory**: `/data1/datasets_1/human_cistrome/SRA/will_metadata/`

## To-Do

There are several enhancement and optimization tasks pending. Check the to-do list in the respective [Pretraining](./pretraining/README.md) and [Fine-tuning](./finetuning/README.md) guides for detailed information.

