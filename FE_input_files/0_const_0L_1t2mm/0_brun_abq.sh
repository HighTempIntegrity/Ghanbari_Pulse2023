#BSUB -J p1t2mm
#BSUB -n 8
#BSUB -W 24:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

abaqus job=run_1t2mm input=1_input cpus=8 scratch=$TMPDIR