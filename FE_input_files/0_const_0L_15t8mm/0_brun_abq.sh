#BSUB -J p15t8mm
#BSUB -n 12
#BSUB -W 72:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

abaqus job=run_15t8mm input=1_input cpus=12 scratch=$TMPDIR