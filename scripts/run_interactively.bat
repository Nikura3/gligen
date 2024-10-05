@echo off

title Uploading script
echo Uploading to the cluster

REM Variables
set user=dvezzaro
set source_dir=../*
set destination_dir=/home/dvezzaro/hf_gligen/

echo - local source folder:%source_dir%
echo - remote destination folder:%destination_dir%

echo Uploading ...
scp -r -J %user%@labta.math.unipd.it %source_dir% %user%@labsrv8.math.unipd.it:%destination_dir%
echo Done

echo Connecting to ssh...
ssh -tt -J %user%@labta.math.unipd.it %user%@labsrv8.math.unipd.it "srun --job-name=GLIGEN_Color --chdir=/home/dvezzaro/hf_gligen --partition=allgroups --time=0-04:00:00 --mem=12G --gres=gpu --cpus-per-task=4 --pty bash -c 'source /conf/shared-software/anaconda/etc/profile.d/conda.sh && conda activate hf_gligen && python -W ignore /home/dvezzaro/hf_gligen/run_gligen.py && conda deactivate'"