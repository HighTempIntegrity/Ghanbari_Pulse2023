#BSUB -J p7t2mm
#BSUB -n 12
#BSUB -W 24:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

abaqus job=run_G00L8mm_HXTdep_7t2mm input=1_input cpus=12 scratch=$TMPDIR