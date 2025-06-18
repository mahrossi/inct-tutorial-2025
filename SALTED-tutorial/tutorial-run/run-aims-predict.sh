#!/bin/bash
### Description: run FHI-aims prediction calculations for test dataset

### Prepare vars
# AIMS=...  # already in ~/.bashrc
ROOTDIR=$(realpath .)
DATADIR=${ROOTDIR}/pred_data
cd ${ROOTDIR}

### Prepare calculation dirs and files
n=$(ls ${DATADIR}/geoms | grep -c 'in')  # number of geometries
for (( i=1; i<=$n; i++ )); do
	mkdir -p ${DATADIR}/$i
	cp ${ROOTDIR}/control_read.in ${DATADIR}/$i/control.in
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
done

### Run FHI-aims calculations
export OMP_NUM_THREADS=1
ulimit -s unlimited
for (( i=1; i<=$n; i++ )); do
	echo "$(date) - Running prediction for geometry $i"
	cd ${DATADIR}/$i
	cp ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	mpirun -np 1 $AIMS < /dev/null > aims_predict.out && mv rho_rebuilt_ri.out rho_ml.out &
	wait
	rm ri_restart_coeffs.out  # remove the restart coeffs file to avoid confusion
	cd ${DATADIR}
done
