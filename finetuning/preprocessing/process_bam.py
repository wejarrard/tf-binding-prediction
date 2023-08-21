import os
import pysam
import subprocess

# Number of cores
ncores = 4

def process_bam(bam_file, ref_file, pileup_dir):
    bam_name = os.path.splitext(os.path.basename(bam_file))[0]
    print("------------------------------------------------------------------")

    # Print bam name
    print(f"Processing {bam_name}")

    # Create sorted BAM file if not exists
    sorted_bam_file = os.path.splitext(bam_file)[0] + "_sorted.bam"
    if not os.path.isfile(sorted_bam_file):
        print("Sorting BAM file")
        pysam.sort("-@", str(ncores), bam_file, "-o", sorted_bam_file)
    else:
        print("Sorted BAM file already exists")

    # Check if sorted BAM index exists
    bam_index_exists = os.path.isfile(sorted_bam_file + ".bai")

    # Check if pileup directory exists
    pileup_dir_path = os.path.join(pileup_dir, bam_name)
    pileup_exists = os.path.isdir(pileup_dir_path)

    # Create BAM index if it doesn't exist
    if not bam_index_exists:
        print("Creating BAM index")
        pysam.index("-@", str(ncores), sorted_bam_file)
    else:
        print("BAM index already exists")

    # Create pileup files if they don't exist
    if not pileup_exists:
        print("Creating pileup files")

        # For each chromosome, create a pileup file
        os.makedirs(pileup_dir_path, exist_ok=True)

        chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]
        for chrom in chromosomes:
            pileup_file_path = os.path.join(pileup_dir_path, f"chr{chrom}.pileup")

            try:
                # Use samtools mpileup to create pileup
                command = f"samtools mpileup -f {ref_file} -r chr{chrom} {sorted_bam_file} > {pileup_file_path}"
                subprocess.run(command, shell=True, check=True)

            except Exception as e:
                print(f"Error processing chromosome {chrom}: {e}")
    else:
        print("Pileup files already exist")

    # Return the pileup directory path
    return pileup_dir_path


if __name__ == "__main__":
    pileup_dir = process_bam("./data/SRX10476719.bowtie.sorted.nodup.bam", "./data/genome.fa", "./data/pileups")
    print(f"Pileup files stored in: {pileup_dir}")
