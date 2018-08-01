#!/bin/bash -l
#SBATCH -o brtnfld.o%j
#SBATCH -t 00:15:00
#SBATCH --ntasks-per-node=32 # Cori
#SBATCH -C haswell 
##SBATCH --ntasks-per-node=24 # Edison
###   1  2  4   8   16  32   64   128  
###   32 64 128 256 512 1024 2048 4096
###   1  2  4   8  16  32   64  128  256   512  1024
#### 24 48 96 192 384 768 1536 3072 6144 12288 24576
#SBATCH -N 16 
##SBATCH -n 16
##SBATCH -p regular
#SBATCH -p debug
cd $SCRATCH
WRKDIR=$SCRATCH/$SLURM_JOB_ID
mkdir $WRKDIR
tsk=$SLURM_NTASKS
lfs setstripe -c 12 -S 16m $WRKDIR
cd $WRKDIR
cmdw="GenericIOBenchmarkWrite"
cmdr="GenericIOBenchmarkRead"
cleanup=yes
cp $SLURM_SUBMIT_DIR/$cmdw .
cp $SLURM_SUBMIT_DIR/$cmdr .
#cp $SLURM_SUBMIT_DIR/$script .
#for i in `seq 1 10`; do
export GENERICIO_USE_HDF=1
#export H5FD_mpio_Debug="rw"
#export GENERICIO_USE_MPIIO=1
for i in `seq 1 10`; do
#srun -n $tsk ./$cmdw out.gio 67108864 2
srun -n $tsk ./$cmdw out.gio 1073741824 2
rm -f out.gio*
done
#cp sum_* $SLURM_SUBMIT_DIR/
#cp file* $SLURM_SUBMIT_DIR/
ls -hl out.gio*
echo $PWD
cd $SLURM_SUBMIT_DIR
if [ -n "$cleanup" ]; then
  rm -r -f $WRKDIR
fi
