# VirtualBox Setup Guide for Atomistic Simulations and FHI-aims

This comprehensive guide walks you through creating and configuring a VirtualBox virtual machine (VM) for building, testing, and running FHI-aims calculations and atomistic simulation tutorials. The guide is designed for complete beginners who have never used VirtualBox before.

## Overview

You have two options for setting up your VM:
1. **Create from scratch** - Build everything yourself (recommended for learning)
2. **Import existing appliance** - Use a pre-configured OVA file (faster setup)

Both approaches are covered in this guide.

## Prerequisites and Initial Setup

### 1. Install VirtualBox

Download and install VirtualBox 7.x from the [official download page](https://www.virtualbox.org/wiki/Downloads):

- For **Linux users**: Choose "Debian-based Linux distributions" from the [Linux Downloads page](https://www.virtualbox.org/wiki/Linux_Downloads) (works well for MPSD Linux laptops running Debian 12)
- For **Windows/Mac users**: Download the appropriate installer from the main download page

### 2. Download LLubuntu Image

Download Lubuntu 22.04 Jammy Jellyfish from [osboxes](https://www.osboxes.org/lubuntu/):
- Look for "Lubuntu 22.04 Jammy Jellyfish"
- Download the **VDI format** file
- You'll get a `64bit.7z` compressed file
- Extract it to get the `.vdi` file in the `64bit/` folder

**Important**: Make a backup copy of the original VDI file, as it might be overwritten during VM operations.

## Method 1: Creating VM from Scratch

### Step 1: Create New Virtual Machine

1. Open Oracle VirtualBox
2. Click **"New"** to create a new VM
3. Fill in the following settings:

**Name and Operating System:**
- Name: `SAbIA-Virtual-Machine` (or `FHIaims_VM`)
- Folder: Choose your preferred location (e.g., `/home/username/VirtualBox VMs/`)
- ISO Image: Leave empty (`<not selected>`)
- Type: `Linux`
- Subtype: `Lubuntu`
- Version: `Lubuntu 22.04 LTS (Jammy Jellyfish) (64-bit)`

4. Click **"Next"**

**Hardware Configuration:**
- Memory size: `4096 MB` (4 GB RAM)
- Processor(s): `4` (adjust based on your system)

5. Click **"Next"**

**Hard Disk:**
- Select **"Use an existing virtual hard disk file"**
- Click the folder icon and navigate to your extracted `.vdi` file
- Select the `Lubuntu 22.04 (64bit).vdi` file

6. Click **"Finish"**

### Step 2: Enable Virtualization (If Needed)

When you first start the VM, you might encounter an error message:
```
AMD-V is disabled in the BIOS (or by the host OS) (VERR_SVM_DISABLED)
```

**To fix this:**
1. Restart your computer and enter BIOS/UEFI settings (usually by pressing F2, F12, or Del during boot)
2. Look for virtualization settings (AMD-V, Intel VT-x, or similar)
3. Enable virtualization support
4. Save and exit BIOS

**Note**: The exact procedure varies by manufacturer. Search online for your specific laptop model + "enable virtualization" for detailed instructions.

### Step 3: First Boot and Login

1. Right-click on your VM in the VirtualBox manager
2. Select **"Start"**
3. Wait for Lubuntu to boot
4. Login with:
   - **Username**: `osboxes`
   - **Password**: `osboxes.org`

**Useful Keyboard Shortcuts:**
- **Host+F**: Exit full-screen mode (Host key = Right Ctrl)
- **Host+A**: Auto-adjust window size
- **Host+C**: Scale mode toggle

## Setting Up Shared Folders

Shared folders allow you to easily transfer files between your host computer and the VM.

### Step 1: Create Local Shared Folder
Create a folder on your host computer called `shared_folder` (or any name you prefer).

### Step 2: Configure Shared Folder in VM
1. In the VM window, go to **Devices** → **Shared Folders** → **Shared Folder Settings...**
2. Click the **folder with plus sign** button
3. Configure:
   - **Folder Path**: Select your local `shared_folder`
   - **Folder Name**: Leave default or customize
   - **Mount Point**: `/home/osboxes/shared_folder`
   - **Check**: Auto-mount
   - **Check**: Make Permanent
4. Click **"OK"**
5. **Reboot the VM** to apply changes

### Step 3: Access Shared Files
After reboot, you can access shared files at:
- `/home/osboxes/shared_folder` (preferred mount point)
- `/media/sf_shared_folder` (alternative location)

**Set proper permissions** for transferred files:
```bash
sudo chown -R osboxes:osboxes <target_file_or_folder>
chmod -R 755 <target_file_or_folder>  # Use 755 instead of 777 for security
```

## Installing Software and Dependencies

### Step 1: System Updates and Cleanup

```bash
# Update package lists
sudo apt-get update

# Remove unnecessary desktop applications to save space
sudo apt-get purge --auto-remove libreoffice-* -y
sudo apt-get purge --auto-remove thunderbird -y
sudo apt-get purge --auto-remove \
    shotwell aisleriot gnome-sudoku gnome-mahjongg \
    Lubuntu-report popularity-contest apport -y

# Clean up residuals
sudo apt-get autoremove --purge
sudo apt-get clean
```

### Step 2: Install Development Tools

```bash
# Install essential development packages (~2 minutes)
sudo apt-get install -y \
    build-essential gfortran gcc g++ \
    cmake ninja-build pkg-config \
    mpi-default-bin mpi-default-dev \
    libscalapack-mpi-dev libblas-dev liblapack-dev \
    git wget curl vim tmux tree less \
    python3 python3-pip
```

### Step 3: Install Conda (Miniforge)

1. Download Miniforge for Linux:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

2. Install with default settings:
```bash
bash Miniforge3-Linux-x86_64.sh
```

3. Follow the installation prompts (accept defaults)
4. Restart your terminal or run `source ~/.bashrc`

## Setting Up FHI-aims

### Step 1: Create Build Configuration

Create a file called `initial_cache.vm_gnu.cmake`:

```cmake
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

### Step 2: Create Build Script

Create a file called `build.sh`:

```bash
#!/bin/bash
BUILD_DIR=build_vm_gnu
INITIAL_CACHE_FNAME=initial_cache.vm_gnu.cmake

if [ -d "${BUILD_DIR}" ]; then
    rm -r ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -C ../${INITIAL_CACHE_FNAME} ..
make -j 4
```

### Step 3: Build FHI-aims

```bash
chmod +x build.sh
./build.sh
```
Then you should find the FHI-aims executable in the `build_vm_gnu` directory.

### Step 4: Test Installation

Run test cases:
- `H2O-relaxation`
- `testcase_water_RI`

Move the FHI-aims executable to a convenient location:
```bash
cp build_vm_gnu/aims.*.x /home/osboxes/aims.scalapack.mpi.x
```

## Method 2: Using Pre-configured OVA Appliance

If you have access to a pre-configured OVA file, follow these steps:

### Step 1: Import OVA File

1. Start VirtualBox
2. Go to **Tools** → **Import**
3. In page **Appliance to Import**, select your `.ova` file, then click **Next**
4. In **Appliance settings**:
   - Change **Machine Base Folder** to your preferred location, e.g., the same folder as the OVA file
   - Adjust other settings as needed
5. Click **"Import"** and wait (~1 minute)

### Step 2: Configure Imported VM

1. The imported VM will appear in your VirtualBox manager
2. Follow the **Shared Folders** setup from the previous section
3. Boot the VM and login with provided credentials
    - Username: `osboxes.org`
    - Password: `osboxes.org`

## Optimizing and Maintaining Your VM

### Disk Space Management

**Clean system caches regularly:**
```bash
sudo apt-get clean
conda clean --all
```

**Zero free space before exporting** (optional, for smaller OVA files):
```bash
sudo dd if=/dev/zero of=/home/osboxes/zerofile bs=1M
sudo rm /home/osboxes/zerofile
```

### Exporting Your Configured VM

To share your configured VM:

1. Clean up temporary files and caches
2. Remove sensitive data
3. In VirtualBox: **File** → **Export Appliance...**
4. Choose your VM and export as OVA format
5. Optionally compress the resulting OVA file

### Performance Tips

- **Allocate sufficient RAM**: At least 4GB, more if running large calculations
- **Enable hardware acceleration**: Ensure virtualization is enabled in BIOS
- **Use SSD storage**: Store VM files on SSD for better performance
- **Adjust processor cores**: Allocate 2-4 cores depending on your system

## Troubleshooting Common Issues

### VM Won't Start
- Check if virtualization is enabled in BIOS
- Verify sufficient RAM is available
- Ensure VDI file path is correct

### Slow Performance
- Increase allocated RAM and CPU cores
- Enable 3D acceleration in VM settings
- Close unnecessary applications on host

### Shared Folders Not Working
- Install VirtualBox Guest Additions
- Reboot VM after configuring shared folders
- Check folder permissions

### Build Errors
- Verify all dependencies are installed
- Check compiler paths in CMake configuration
- Ensure MPI libraries are properly linked

## Next Steps

Once your VM is configured:

1. **Run tutorial exercises** using the prepared environments
2. **Practice with sample calculations** to verify everything works
3. **Backup your configured VM** for future use
4. **Share the OVA file** with colleagues for consistent tutorial environments

## Additional Resources

- [VirtualBox Documentation](https://docs.oracle.com/en/virtualization/virtualbox/)
- [FHI-aims Documentation](https://fhi-aims-club.gitlab.io/tutorials/)

