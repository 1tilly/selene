#!/bin/bash

# Chromatin profiles download. ENCODE/Roadmap Epigenomics .bed files used
# in DeepSEA (Zhou & Troyanskaya, 2015).
# wget https://zenodo.org/record/2214970/files/chromatin_profiles.tar.gz
# tar -xzvf chromatin_profiles.tar.gz
module load samtools

~/.conda/envs/phdeep/bin/python process_chromatin_profiles.py ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/deepsea__919_features_.txt \
                                     ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/ENCODE_DNase/ \
                                     ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/ENCODE_TF/ \
                                     ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/Roadmap_Epigenomics/ \
                                     ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/

sort -k1V -k2n -k3n ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/deepsea_full_unsorted.bed > ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/sorted_deepsea_data.bed

bgzip -c ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/sorted_deepsea_data.bed > ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/sorted_deepsea_data.bed.gz

tabix -p bed ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/sorted_deepsea_data.bed.gz

sort -o ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/distinct_features.txt ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/distinct_features.txt

~/.conda/envs/phdeep/bin/python create_TF_intervals_file.py ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/distinct_features.txt \
                                   ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/sorted_deepsea_data.bed \
                                   ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/TF_intervals_unmerged.txt

bedtools merge -i ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/TF_intervals_unmerged.txt > ~/Projects/DeepLearning/DeepSea_data/chromatin_profiles/TF_intervals.txt
