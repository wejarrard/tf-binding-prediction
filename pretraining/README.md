# Pretraining Your Model Guide

Follow these steps to pretrain your model:

1. **Upload Files**:
   - BAM files → `s3://tf-binding-sites/pretraining/bam/`
   - Corresponding narrowPeak file → `s3://tf-binding-sites/pretraining/atac/`

2. **AWS SageMaker**:
   - Access [AWS SageMaker](https://console.aws.amazon.com/sagemaker/).
   - Click on the `Studio Lab` tab.

3. **Region & User Settings**:
   - Ensure your region is set to `us-west-2`.
   - Open the environment under the user `willjarrard`.
     > **Note**: Ensure you have the correct permissions to access this environment.

4. **Pretraining Steps**:
   - Navigate to the `pretraining` folder.
   - Execute the preprocessing script. (This might take a few hours.)
   - After preprocessing, run the training script.
    > **Note**: Training script is currently being rewritten to work in SageMaker, will not work currently.

5. **Output**:
   - Once training is complete, your outputs can be found in `s3://tf-binding-sites/pretraining`.

## 📝 **To-Do List**

🔲 **SageMaker Setup**: Initialize the training script on SageMaker.  
🔲 **Enhance Model Capabilities**: Determine a method for the model to actually utilize reads during prediction. Brainstorm with David.

## ✅ **Completed Tasks**

🔳 **Documentation**: Crafted the README file and organized this to-do list.  
🔳 **Model Parameters**: Establish a central location to store and track various parameters tested on the model.  
🔳 **Preprocessing Automation**: Ensure preprocessing runs in the background. Consider using a notebook job. Should be complete, double check running preprocessing script.  
🔳 **Redo Scripts**: Implement command line arguments and utilize feather files.  