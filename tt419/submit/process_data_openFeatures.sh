#!/bin/bash

# Chromatin profiles download. ENCODE/Roadmap Epigenomics .bed files used
# in DeepSEA (Zhou & Troyanskaya, 2015).
#wget https://zenodo.org/record/2214970/files/chromatin_profiles.tar.gz
#tar -xzvf chromatin_profiles.tar.gz

/home/tt419/anaconda2/envs/Selene/bin/python process_chromatin_profiles_openFeatures.py /home/tt419/Projects/DeepLearning/PhDeep/data/ENCODE_ftp/label_names.txt \
                                     /home/tt419/Projects/DeepLearning/PhDeep/data/ENCODE_ftp/DNAse/ \
                                     /home/tt419/Projects/DeepLearning/PhDeep/data/ENCODE_ftp/TFChip/ \
                                     /home/tt419/Projects/DeepLearning/PhDeep/data/Selene/chromatin_profiles/Roadmap_Epigenomics/ \
                                     /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/

sort -k1V -k2n -k3n /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/selene_fullFeatures_unsorted.bed > /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/sorted_selene_fullFeatures.bed

bgzip -c /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/sorted_selene_fullFeatures.bed > /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/sorted_selene_fullFeatures.bed.gz

tabix -p bed /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/sorted_selene_fullFeatures.bed.gz

sort -o /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/distinct_features.txt /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/distinct_features.txt

/home/tt419/anaconda2/envs/Selene/bin/python create_TF_intervals_file_openFeatures.py /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/distinct_features.txt \
                                   /rds-d5/project/who1000/rds-who1000-wgs10k/user/tt419/Selene_data/selene_ftp_output/sorted_selene_fullFeatures.bed \
                                   TF_intervals_unmerged.txt

bedtools merge -i TF_intervals_unmerged.txt > TF_intervals.txt
