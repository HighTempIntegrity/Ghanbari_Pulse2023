#BSUB -J 01L_8t1mm_M
#BSUB -n 12
#BSUB -W 24:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

abaqus job=run_01L_8t1mm_M input=1_input cpus=12 scratch=$TMPDIR