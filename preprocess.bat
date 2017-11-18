@echo off
set datasetname=platechars
echo "Generating lst"
python preprocess/im2rec.py %datasetname%/%datasetname% %datasetname% --recursive=True --list=True
echo "Generating rec"
python preprocess/im2rec.py %datasetname%/%datasetname% %datasetname%
echo "Generating synsetwords"
python preprocess/generatesynsetwords.py --datadir=%datasetname%