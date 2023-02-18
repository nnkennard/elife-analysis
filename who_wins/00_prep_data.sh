
# Download data

# AMPERE
wget https://xinyuhua.github.io/resources/naacl2019/naacl19_dataset.zip
unzip naacl19_dataset.zip
mkdir -p data/raw/ampere
mv dataset/iclr_anno_final/* data/raw/ampere/
rm naacl19_dataset.zip
rm -r dataset/

# DISAPERE
wget https://github.com/nnkennard/DISAPERE/raw/main/DISAPERE.zip
unzip DISAPERE.zip
mkdir -p data/raw/disapere
mv DISAPERE/final_dataset/* data/raw/disapere/
rm -r DISAPERE*

# ReviewAdvisor (has to be scp-ed)
#mkdir -p data/raw/revadv
# scp -i ~/.ssh/id_nnk_courbet -r ~/Downloads/dataset-2/aspect_data nnayak_umass_edu@unity.rc.umass.edu:/work/nnayak_umass_edu/elife-analysis/who_wins/data/raw/revadv 
unzip data/raw/revadv/review_with_aspect.jsonl.zip
mv review_with_aspect.jsonl data/raw/revadv/


# Convert data format
