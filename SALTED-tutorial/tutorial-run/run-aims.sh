#!/bin/bash
### Description: run FHI-aims fully converged calculations for test dataset

### Prepare vars
# AIMS=/Users/rossi/Codes/FHIaims/build_2025/aims.250507.serial.x
AIMS=/home/zekunlou/Projects/FHI_aims/FHIaims_250320/bin/aims.250320.scalapack.mpi.gnu_debian_worklaptop.date_250601.x
ROOTDIR=$(realpath .)
# DATADIR=${ROOTDIR}/pred_data
DATADIR=${ROOTDIR}/pred_data_zl_test
cd ${ROOTDIR}

### Prepare calculation dirs and files
n=$(ls ${DATADIR}/geoms | grep -c 'in')  # number of geometries
for (( i=1; i<=$n; i++ )); do
	mkdir -p $DATADIR/$i
	cp ${ROOTDIR}/control.in ${DATADIR}/$i/control.in
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
done

### Run FHI-aims calculations
export OMP_NUM_THREADS=1
ulimit -s unlimited
for (( i=1; i<=$n; i++ )); do
	echo "$(date) - Running prediction for geometry $i"
	cd ${DATADIR}/$i
	mpirun -np 1 $AIMS < /dev/null > aims.out && mv ri_restart_coeffs.out ri_restart_coeffs_df.out &
	wait
	cd ${DATADIR}
done
