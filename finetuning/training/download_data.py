import boto3
import os

s3 = boto3.resource('s3')
bucket = s3.Bucket('tf-binding-sites')

local_dir = "/Users/wejarrard/projects/atacToChip/finetuning/preprocessing/output/non_intersecting/"
s3_dir = "finetuning/output/non_intersecting/"

counter = 0
for obj in bucket.objects.filter(Prefix=s3_dir):
    if counter < 100:
        filename = os.path.basename(obj.key)  # Only get the filename, not the whole S3 path
        local_file_path = os.path.join(local_dir, filename)  # Construct local file path
        bucket.download_file(obj.key, local_file_path)
        counter += 1
    else:
        break
