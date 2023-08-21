#!/bin/bash

# Set default values
chipseq_file="/app/data/SRX10030477.bed.filtered.narrowPeak"
outdir="/app/tf-footprinting-output"
atac_peaks="/app/data/SRX7023024.filtered.narrowPeak"
genome="/app/data/genome.fa"
bam="/app/data/SRX7023024.bowtie.sorted.nodup.bam"
motifs="/app/data/JASPAR2018_CORE_vertebrates_non-redundant.meme"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --chipseq_file) chipseq_file="$2"; shift ;;
        --outdir) outdir="$2"; shift ;;
        --atac_peaks) atac_peaks="$2"; shift ;;
        --genome) genome="$2"; shift ;;
        --bam) bam="$2"; shift ;;
        --motifs) motifs="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
# Create a temporary directory for intermediate files
temp_dir=$(mktemp -d -t tobias-XXXXXX)

# ATACorrect command
TOBIAS ATACorrect --bam $bam --genome $genome --peaks $atac_peaks --outdir $outdir --cores 4

# FootprintScores command
TOBIAS FootprintScores --signal ${outdir}/$(basename $bam .bam)_corrected.bw --regions $atac_peaks --output ${outdir}/Footprints_$(basename $bam .bam).bw --cores 4

# BINDetect command
TOBIAS BINDetect --motifs $motifs --signals ${outdir}/Footprints_$(basename $bam .bam).bw  --genome $genome --peaks $atac_peaks --outdir ${outdir}/BINDetect_output_$(basename $bam .bam) --cond_names $(basename $bam .bam) --cores 4

# Define file names
fullsite_predictions="${outdir}/BINDetect_output_$(basename $bam .bam)/AR_FullSite/beds/AR_FullSite_$(basename $bam .bam)_bound.bed"
halfsite_predictions="${outdir}/BINDetect_output_$(basename $bam .bam)/AR_HalfSite/beds/AR_HalfSite_$(basename $bam .bam)_bound.bed"
extendedsite_predictions="${outdir}/BINDetect_output_$(basename $bam .bam)/AR_ExtendedSite/beds/AR_ExtendedSite_$(basename $bam .bam)_bound.bed"

combined_predictions="${temp_dir}/combined_bound_predictions.bed"

# Combine the prediction files into a single file
cat $fullsite_predictions $halfsite_predictions $extendedsite_predictions | sort -k1,1 -k2,2n > $combined_predictions

mkdir tf-footprinting

# Define output file for markdown format
OUTPUT_FILE="tf-footprinting/results.md"

# Function to calculate precision, recall, and F1 score
calculate_metrics() {
    local prediction_file="$1"
    local prefix="$2"

    # Calculate overlaps (TP)
    bedtools intersect -a $chipseq_file -b $prediction_file -u > "${prefix}_TP.bed"
    TP=$(wc -l < "${prefix}_TP.bed")

    # Calculate non-overlapping predictions (FP)
    bedtools intersect -a $prediction_file -b $chipseq_file -v > "${prefix}_FP.bed"
    FP=$(wc -l < "${prefix}_FP.bed")

    # Calculate non-overlapping chipseq peaks (FN)
    bedtools intersect -a $chipseq_file -b $prediction_file -v > "${prefix}_FN.bed"
    FN=$(wc -l < "${prefix}_FN.bed")

    # Calculate precision, recall and F1 score
    precision=$(awk "BEGIN {print $TP / ($TP + $FP)}")
    recall=$(awk "BEGIN {print $TP / ($TP + $FN)}")
    f1_score=$(awk "BEGIN {print 2 * $precision * $recall / ($precision + $recall)}")

    # Append results to markdown file in table format
    echo -e "| ${prefix} | $precision | $recall | $f1_score |" >> $OUTPUT_FILE
}

# Initialize the markdown file with table headers
echo -e "| Metric | Precision | Recall | F1 Score |" > $OUTPUT_FILE
echo -e "|--------|-----------|--------|----------|" >> $OUTPUT_FILE

# Calculate metrics for each prediction file
calculate_metrics $fullsite_predictions "FullSite"
calculate_metrics $halfsite_predictions "HalfSite"
calculate_metrics $extendedsite_predictions "ExtendedSite"
calculate_metrics $combined_predictions "Combined"

# Remove temporary directory
rm -rf $temp_dir
