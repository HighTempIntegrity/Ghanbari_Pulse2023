#BSUB -J exp[1-4]
#BSUB -n 12
#BSUB -W 120:00
#BSUB -N
#BSUB -R 'rusage[mem=2048,scratch=2000]'
#BSUB -R 'select[model=XeonGold_5118]'

python 3_pycode_exp$LSB_JOBINDEX.py