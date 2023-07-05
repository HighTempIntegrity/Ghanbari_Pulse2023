#BSUB -J pLH
#BSUB -n 8
#BSUB -W 4:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

abaqus job=run_LH input=3_R0000_input cpus=8 scratch=$TMPDIR