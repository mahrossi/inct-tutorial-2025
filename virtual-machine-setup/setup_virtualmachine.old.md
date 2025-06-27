# README

This guide walks you through creating and configuring a VirtualBox VM for building and testing FHI-aims on Debian/Lubuntu hosts. It covers both creating a VM from scratch and importing from an existing OVA appliance.

> TODO: add conda environment yaml files for each tutorial, and add instructions for installing conda environments.

## Install Virtual Box

VirtualBox 7.x installed via your distribution package manager or from the [download page](https://www.virtualbox.org/wiki/Downloads).
I chose `Debian-based Linux distributions` on [this page](https://www.virtualbox.org/wiki/Linux_Downloads) and it works well for MPSD linux laptops (Debian 12).

## Setup Virtual Machine from Scratch

If you have got the OVA file, you can skip this section.

The Image is `Lubuntu 22.04 Jammy Jellyfish` from [osboxes](https://www.osboxes.org/lubuntu/), and I chose `vdi` format.

### 1.Load Image

Enter Virtual Box.

Option `New`:

```text
Name=FHIaims_VM
Folder=/home/zekunlou/Projects/virtualbox/fhi-aims_250504_2
ISO Image=<not selected>
Type=Linux
Subtype=Lubuntu
Version=Lubuntu 22.04 LTS (Jammy Jellyfish) (64-bit)
```

Click `Next`.

```text
Memory size=4096 MB
Processor(s)=4
```

Click `Next`.

```text
Use an existing virtual hard disk file
Location=/home/zekunlou/Projects/virtualbox/fhi-aims_250504_2/64bit/Lubuntu 22.04 (64bit).vdi
(decompressed from osboxes 7z file)
```

Then a clean virtual machine appliance is created.

### 2.Setup FHI-aims

Start the VM. Login with `username=osboxes`, `password=osboxes.org`.

Follow section `Add Shared Folder` below to share files between host and guest.

Install everything
```bash
sudo apt-get update
# Purge all LibreOffice components
sudo apt-get purge --auto-remove libreoffice-* -y
# Remove Thunderbird
sudo apt-get purge --auto-remove thunderbird -y
# Remove a few other desktop-only apps (adjust as you like)
sudo apt-get purge --auto-remove \
    shotwell aisleriot gnome-sudoku gnome-mahjongg \
    Lubuntu-report popularity-contest apport -y
# Finally, clean up residuals
sudo apt-get autoremove --purge
sudo apt-get clean
# Install useful packages
sudo apt-get install -y build-essential gfortran gcc g++ cmake ninja-build pkg-config mpi-default-bin mpi-default-dev libscalapack-mpi-dev libblas-dev liblapack-dev git wget curl vim tmux tree less python3 python3-pip  # around 2 min
```

> Optional: install VS Code and other applications if you would like.

File `initial_cache.vm_gnu.cmake`:
```
set(CMAKE_Fortran_COMPILER mpifort CACHE STRING "" FORCE)
set(CMAKE_Fortran_FLAGS "-O2 -fallow-argument-mismatch -ffree-line-length-none" CACHE STRING "" FORCE)
set(Fortran_MIN_FLAGS "-O0 -ffree-line-length-none" CACHE STRING "" FORCE)
set(CMAKE_C_COMPILER mpicc CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "-O2" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER mpicxx CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-O2" CACHE STRING "" FORCE)
set(LIB_PATHS "/usr/lib/x86_64-linux-gnu" CACHE STRING "" FORCE)   # <-- this line is important
set(LIBS "libblas.so liblapack.so libscalapack-openmpi.so" CACHE STRING "" FORCE)
set(USE_MPI ON CACHE BOOL "" FORCE)
set(USE_SCALAPACK ON CACHE BOOL "" FORCE)
set(USE_LIBXC ON CACHE BOOL "" FORCE)
set(USE_HDF5 OFF CACHE BOOL "" FORCE)
set(USE_RLSY ON CACHE BOOL "" FORCE)
```

File `build.sh`:
```bash
BUILD_DIR=build_vm_gnu
INITIAL_CACHE_FNAME=initial_cache.vm_gnu.cmake

if [ -d "${BUILD_DIR}" ]; then
	rm -r ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -C ../${INITIAL_CACHE_FNAME} ..
make -j 2
```

Then run the test case `H2O-relaxation` and `testcase_water_RI`.

Move FHI-aims executable to `/home/osboxes/aims.250425.scalapack.mpi.x`

### 3.Install `conda`

You can find the [download link to `Linux/x86_64`](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) in the [Download page for miniforge](https://conda-forge.org/download/).
Just follow the default steps to install it.

### 4.Zero and Defragment Disk

> **TODO:** if image size is too large, we can try out other ways, or seek for help from MPSD SSU team.

Suggested by LLM
```bash
sudo dd if=/dev/zero of=zerofile bs=1M; sudo rm zerofile`
```

Also see [this page](https://superuser.com/questions/529149/how-to-compact-virtualboxs-vdi-file-size) and [this page](https://docs.oracle.com/en/virtualization/virtualbox/6.0/user/vboxmanage-modifymedium.html).


### 5.Export OVA File

Before Export, remove FHI-aims code and `.git` folder, also clean `apt` cache and `conda` cache.
```bash
sudo apt-get clean
conda clean --all
```

Then in Virtual Box, select the VM, then `File` -> `Export Appliance...`.
The exported `ova` file name is `FHIaims_VM.ova`.

### 6.Compress OVA File

First `tar -xzf FHIaims_VM.ova` and we get
- `FHIaims_VM.ovf`: descriptor
- `FHIaims_VM-disk001.vmdk`: disk image, largest, to be compressed
- `FHIaims_VM.mf`: manifest with checksums

```bash
VBoxManage clonehd FHIaims_VM-disk001.vmdk FHIaims_VM-disk001.vdi --format VDI
# this takes about 2 min, vmdk 11G -> vdi 20G, we shall check other ways to compress the disk image...
```

## Load Virtual Machine from Existing Appliance

### 1.Load Existing OVA File

Start Virtual Box, select `Tools`, then select `Import`.
In page `Appliance to import` blanck `File`, select `/path/to/image.ova`, then click `Next`.
In page `Appliance settings`, you can change `Machine Base Folder` to your perferred location (e.g. the same folder as the `ova` file). Click `Finish`.
Wait ~ 1 min for importing the appliance, you will see a new folder `<VM_name>`.
Boot the virtual machine. User name is `osboxes`, password is `<password>`.

### 2.Add Shared Folder

[Document](https://docs.oracle.com/en/virtualization/virtualbox/6.0/user/sharedfolders.html)

Create a folder locally called `shared_folder` for file sharing between host and guest (the virtual machine).
In the guest window, open `Devices` -> `Shared Folders` -> `Shared Folder Settings...`.
Click the button of a folder and a plus sign.
For `Folder path`, select the local folder `shared_folder`.
For `Mount point`, input `/home/osboxes/shared_folder`.
Check `Auto-mount`. Then click `OK`.
Reboot the virtual machine to make the changes take effect if you don't see the folder.
It is also available at `/media/sf_shared_folder`, and this is also the folder we will use to share files between host and guest.

Then you can transfer files between host and guest, and remember `chown` and `chmod`.

```bash
sudo chown -R osboxes:osboxes <target_file_or_folder>
chmod -R 777 <target_file_or_folder>
```

### 3.Prepare Conda Environments

`miniforge` is installed. YML files for environments setup is suggested.
See [Create an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) and [Exporting the environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file).

