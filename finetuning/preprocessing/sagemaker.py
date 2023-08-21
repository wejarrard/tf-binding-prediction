import os
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor

def run_sagemaker_processing():
    role = get_execution_role()

    processor = ScriptProcessor(base_job_name='finetuning-processor',
                                image_uri='016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-prediction:latest',
                                command=['python3'],
                                instance_type='ml.m5.xlarge',
                                instance_count=1,
                                role=role,
                                sagemaker_session=sagemaker.Session(),
                                volume_size_in_gb=300)
    
    base_dir = '/app'
    dirs = os.listdir(base_dir)
    
    for dir in dirs:
        pileup_dir = os.path.join(base_dir, dir)
        if os.path.isdir(pileup_dir):
            output_file = os.path.basename(pileup_dir)


            # Figure out what pricessinginput and output is
            processor.run(code='s3://mybucket/myfolder/processing_script.py',
                        inputs=[
                            ProcessingInput(source='s3://your-bucket/path-to-input-data1',
                                            destination='/opt/ml/processing/input/data1'),
                            ProcessingInput(source='s3://your-bucket/path-to-input-data2',
                                            destination='/opt/ml/processing/input/data2'),
                            ProcessingInput(source='s3://your-bucket/path-to-input-data3',
                                            destination='/opt/ml/processing/input/data3')
                        ],
                        outputs=[ProcessingOutput(source='/opt/ml/processing/output',
                                                    destination='s3://mybucket/myfolder/output')],
                        arguments=['--chipseq-bed-path', 's3://your-bucket/your-path-to-chip-data',
                                    '--atacseq-bed-path', 's3://your-bucket/your-path-to-atac-data',
                                    '--atacseq-bam-path', 's3://your-bucket/your-path-to-atac-bam-data',
                                    '--genome-path', 's3://your-bucket/your-path-to-genome-data',
                                    '--output-path', '/opt/ml/processing/output'])

if __name__ == "__main__":
    run_sagemaker_processing()
