#!/bin/sh

#$ -N hparams_LRCN
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=64G
#$ -pe gpu-titanx 1
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s1509050@ed.ac.uk
#$ -m beas
#$ -P lel_hcrc_cstr_students

# Initialise the environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate diss

# Run the program
python check.py ../spec_data/UTTS/data
