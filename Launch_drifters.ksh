#!/bin/ksh
#SBATCH -J DRIFTERS_GLO12-SMOC
#SBATCH -A cmems
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -e Drifters_SMOC_2024Q1-Ksmag_10-job%j.e
#SBATCH -o Drifters_SMOC_2024Q1-Ksmag_10-job%j.o
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#CONFIG="GLO12V4_TEST5"
#CONFIG="GLO12V4"
#CONFIG="GLO12_TEST5"
#CONFIG="GLO12"
#CONFIG="GLOBCURR"
#CONFIG="GLO12"
CONFIG="SMOC"
DATE1="20240104"
DATE2="20241228"
DATE2="20240331"
typemod="pgn"
DIR_BIN="/home/ext/mr/smer/regnierc/anaconda3/envs/lagrangian/bin/"
${DIR_BIN}python Lagrangian_tool.py --date1 ${DATE1} --date2 ${DATE2} --lead HDCT --sys ${CONFIG}
