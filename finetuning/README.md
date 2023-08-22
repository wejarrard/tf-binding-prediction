# Fine-tuning Your Model Guide

Follow the steps below to fine-tune your model:

1. **Accessing UCSF Server**:
   - Use `SSH` to access `UCSF`.

2. **Navigate to Dataset Directory**:
   - Change directory to `/data1/datasets_1/human_cistrome/SRA/will_metadata/`.

3. **Run the Scripts**:
   - Execute `atac_chip_r.py`.
   - Execute `chip_chip_r.py`.

4. **Upload Results to S3 Bucket**:
   - Upload the resulting `.bam` files (produced by `atac`) and `.bed` files (produced by both `atac` and `chip`) to the S3 bucket: `s3://tf-binding-sites/finetuning/`.

5. **Using AWS SageMaker**:
   - Go to AWS SageMaker.
   - Click on the `Studio Lab` tab.
   - Ensure you're in the `us-west-2` region.
   - Access the environment with the username `willjarrard` (Note: Ensure you have the right permissions to do this).
   
6. **Model Preprocessing and Training**:
   - Execute the `preprocessing` script.
   - Once preprocessing is completed, run the `training` script.

7. **Access the Fine-tuned Model**:
   - The fine-tuned model will be available in the same S3 bucket at: `s3://tf-binding-sites/finetuning/models`.

## 📖 Additional Reading

For a comprehensive guide on the various data processing techniques we've explored, please refer to our dedicated documentation: [**data-processing-notes**](./data-processing-notes).

## 📝 **To-Do List**

🔲 **Model Parameters**: Create a centralized location to manage and monitor the model's tested parameters.  
🔲 **Testing Script Update**: Modify `testing/testing.ipynb` to account for CHIP peaks absent in ATAC peaks as negative predictions. (Note: This was previously done but got lost).

## ✅ **Completed Tasks**

🔳 **Documentation**: Developed the README file and structured this task list.
