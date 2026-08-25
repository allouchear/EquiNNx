# Build OMAT24 dataset for EquiNNx 

## Get dataset from (huggingface)[https://huggingface.co/datasets/facebook/OMAT24]
Run xwget script

## Uncompress all .tar.gz file
Run  xuntar script

## Compte average E0s for all atoms using all db_* files in all directorys
 Run xcomputeE0s script

##  Convert all db_* files in all directories in .h5 format
 -  create script to convert all db_* files in all directories in .h5 format
 Run ./xcreateRunAllBuildH5
 This script create the xrunAll script to convert all db_* files in h5 format

 - Use submitGnuParallel to run xrunAll in parallel.
 Lynx cluster : submitGnuParallel nodename 40 xrunAll 
 IN2P3 cluster : submitGnuParallel  long 40 xrunAll

## Merge all h5 files in one
Run xMerge script

## Build batched h5 file for EquiNNx
Run xbuildBatching script

## Run train
Run xtrainBatching script
