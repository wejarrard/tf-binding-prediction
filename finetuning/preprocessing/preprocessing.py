from pybedtools import BedTool
import pandas as pd
from tqdm import tqdm
import os
import argparse
import subprocess
import pysam

# Number of cores
ncores = 4

def split_data(df, train_ratio=1):
    # We removed directory creation here because it's handled in the main function
    train = df.sample(frac=train_ratio, random_state=0)
    test = df.drop(train.index)
    return train, test

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

def is_valid_chromosome(x):
    if x in {"chrX", "chrY"} or x[3:].isdigit():
        return True
    return False

def make_chromosome_int(x):
    if x == "chrX":
        return 23
    elif x == "chrY":
        return 24
    else:
        return int(x[3:])
    
    
def get_pileup_data(df, pileup_df):
    start_end_tuples = df[['start', 'end']].to_records(index=False)
    return pd.concat([pileup_df[(pileup_df['pos'] >= start) & (pileup_df['pos'] <= end)] for start, end in start_end_tuples])


def get_total_reads(bam_file):
    # Count the total number of reads in the BAM file.
    total_reads = int(subprocess.run(["samtools", "view", "-c", bam_file], stdout=subprocess.PIPE).stdout.decode("utf-8").strip())
    
    # Normalize total reads to millions.
    total_reads_millions = total_reads / 1_000_000
    print(f"Total reads per million in: {total_reads_millions}")
    if total_reads_millions == 0:
        print(f"Skipping {bam_file} because total reads is 0")
        return
    return total_reads_millions


def process_data(chipseq_bed_path, atacseq_bed_path, blacklist_bed_path, pileup_dir, output_path, total_reads):
    blacklist_bed = BedTool(blacklist_bed_path)

    chipseq_bed_original = BedTool(chipseq_bed_path)
    # df_chipseq = pd.read_table(chipseq_bed_original.fn, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'], index_col=False)
    # cutoff_chipseq = df_chipseq['pValue'].quantile(0.90)
    # df_chipseq_filtered = df_chipseq[df_chipseq['pValue'] >= cutoff_chipseq]
    # chipseq_bed_filtered = BedTool.from_dataframe(df_chipseq_filtered)

    # Subtract the blacklisted regions from the chipseq_bed
    chipseq_bed_filtered = chipseq_bed_original.subtract(blacklist_bed)

    atacseq_bed = BedTool(atacseq_bed_path)
    
    # Subtract the blacklisted regions from the atacseq_bed
    atacseq_bed = atacseq_bed.subtract(blacklist_bed)

    intersecting_peaks = atacseq_bed.intersect(chipseq_bed_filtered, u=True)
    chipseq_only_peaks = chipseq_bed_filtered.intersect(atacseq_bed, v=True)
    atacseq_only_peaks = atacseq_bed.intersect(chipseq_bed_filtered, v=True)

    # Subtract the original chipseq_bed from atacseq_only_peaks
    atacseq_only_peaks = atacseq_only_peaks.subtract(chipseq_bed_original)


    df_intersecting = pd.read_table(intersecting_peaks.fn, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'], index_col=False)
    df_intersecting = df_intersecting[df_intersecting['chrom'].apply(is_valid_chromosome)]
    df_intersecting['chrom'] = df_intersecting['chrom'].apply(make_chromosome_int)
    df_intersecting = df_intersecting.sort_values(by=['chrom', 'start'])

    df_chipseq_only = pd.read_table(chipseq_only_peaks.fn, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'], index_col=False)
    df_chipseq_only = df_chipseq_only[df_chipseq_only['chrom'].apply(is_valid_chromosome)]
    df_chipseq_only['chrom'] = df_chipseq_only['chrom'].apply(make_chromosome_int)
    # if len(df_intersecting) < len(df_chipseq_only):
    #     df_chipseq_only = df_chipseq_only.sample(n=len(df_intersecting))  # Match the number of intersecting peaks
    df_chipseq_only = df_chipseq_only.sort_values(by=['chrom', 'start'])

    df_atacseq_only = pd.read_table(atacseq_only_peaks.fn, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'], index_col=False)
    df_atacseq_only = df_atacseq_only[df_atacseq_only['chrom'].apply(is_valid_chromosome)]
    df_atacseq_only['chrom'] = df_atacseq_only['chrom'].apply(make_chromosome_int)
    # if len(df_intersecting) < len(df_atacseq_only):
    #     df_atacseq_only = df_atacseq_only.sample(n=len(df_intersecting))  # Match the number of intersecting peaks
    df_atacseq_only = df_atacseq_only.sort_values(by=['chrom', 'start'])


    pileup_files_dict = {}

    # make output directory if it doesn't exist already, if it does do nothing
    os.makedirs(output_path, exist_ok=True)

    sample_name = os.path.basename(pileup_dir)

    for filename in os.listdir(pileup_dir):
        if filename.endswith('.pileup'):
            chrom = filename.split('.')[0]
            if is_valid_chromosome(chrom):
                int_chrom = make_chromosome_int(chrom)
                filepath = os.path.join(pileup_dir, filename)
                pileup_files_dict[int_chrom] = filepath

    # make output directories if they don't exist already, if they do nothing
    os.makedirs(os.path.join(output_path, 'train', 'intersecting'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train', 'chipseq_only'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train', 'atacseq_only'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'intersecting'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'chipseq_only'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'atacseq_only'), exist_ok=True)

    for main_df, category in [(df_intersecting, 'intersecting'),
                            (df_chipseq_only, 'chipseq_only'), 
                            (df_atacseq_only, 'atacseq_only')]:

        for chrom in main_df['chrom'].unique():
            print(f"Parsing chromosome {chrom}")
            chrom_df = main_df[main_df['chrom'] == chrom]

            # Split your data here
            train_df, test_df = split_data(chrom_df)

            for split_df, split in [(train_df, "train"), (test_df, "test")]:
                pileup_file = pileup_files_dict[chrom]

                pileup_df = pd.read_csv(pileup_file, sep='\t', header=None, usecols=[0, 1, 2, 3], names=['chrom', 'pos', 'base', 'reads'])
                
                # Normalize the 'reads' column by the total number of reads (in millions).
                pileup_df['reads'] = pileup_df['reads'] / total_reads

                pileup_df = pileup_df[pileup_df['chrom'].apply(is_valid_chromosome)]
                pileup_df['chrom'] = pileup_df['chrom'].apply(make_chromosome_int)

                max_position = pileup_df['pos'].max()
                split_df['start'] = split_df['start'].apply(lambda x: max(0, x - 2500))
                split_df['end'] = split_df['end'].apply(lambda x: min(x + 2500, max_position))

                for i, row in tqdm(split_df.iterrows(), total=split_df.shape[0]):
                    pileup_data = get_pileup_data(pd.DataFrame([row]), pileup_df)
                    feather_df = pd.DataFrame(pileup_data, columns=["chrom", "pos", "base", "reads"])
                    try:
                        feather_df.reset_index(drop=True).to_feather(f"{output_path}/{split}/{category}/{sample_name}_{chrom}_{i}.feather")
                    except Exception as e:
                        print(f"Error writing .feather file: {e}")
                        pass




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--chipseq-bed-path', type=str)
    parser.add_argument('--atacseq-bed-path', type=str)
    parser.add_argument('--atacseq-bam-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--genome-path', type=str)
    parser.add_argument('--blacklist-bed-path', type=str)
    args = parser.parse_args()

    print(f"Checking file paths:")
    for path in [args.chipseq_bed_path, args.atacseq_bed_path, args.atacseq_bam_path, args.output_path, args.genome_path, args.blacklist_bed_path]:
        if os.path.exists(path):
            print(f"File exists: {path}")
        else:
            print(f"File does not exist: {path}")

    pileup_dir = process_bam(
        bam_file=args.atacseq_bam_path,
        ref_file=args.genome_path,
        pileup_dir="./pileups")
    
    total_reads = get_total_reads(args.atacseq_bam_path)

    process_data(
        chipseq_bed_path=args.chipseq_bed_path,
        atacseq_bed_path=args.atacseq_bed_path,
        blacklist_bed_path=args.blacklist_bed_path,
        pileup_dir=pileup_dir,
        output_path=args.output_path, 
        total_reads=total_reads
    )


if __name__ == "__main__":
    main()

    
# python preprocessing.py \
# --chipseq-bed-path ./data/chip_SRX5437818.filtered.narrowPeak \
# --atacseq-bed-path ./data/atac_SRX5437818.filtered.narrowPeak \
# --atacseq-bam-path ./data/SRX5437818.bowtie.sorted.nodup.bam \
# --blacklist-bed-path ./data/GRCh38_unified_blacklist_2020_05_05.bed \
# --output-path ./output \
# --genome-path ./data/genome.fa