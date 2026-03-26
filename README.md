# COMS_E6998

High Performance Machine Learning
Prof. Kaoutar El Maghraoui
Columbia University
Spring 2026

# Insomnia

## Ed Post by Instructor

https://edstem.org/us/courses/95085/discussion/7715326

Dear class,

You should have received an email with instructions on how to access the Insomnia Research Computing cluster. If you did not receive this email, please let us know by replying to this thread so we can follow up.

For this course, you may use any of the following GPU resources:

- The Insomnia cluster (Columbia Research Computing),
- Google Cloud Platform (GCP) VMs with GPUs,
- or other GPU resources you may already have access to (e.g., institutional clusters, cloud credits, or personal GPU systems),

as long as your setup complies with course guidelines.

**This post provides initial instructions for accessing the Insomnia cluster**, along with links to official documentation and an overview of the GPUs available for coursework and projects.

**Extra credit opportunity:**

If you discover useful tips, best practices, or reusable scripts related to using GPUs on Insomnia (e.g., Slurm job templates, debugging tips, performance optimizations), you are encouraged to share them with the class by posting in this thread. High-quality contributions will receive extra credit toward participation or homework.

This is a great way to help your peers while getting credit for practical HPC experience.

Please read this carefully before logging in for the first time.

---

### 1. Official Documentation (Start Here)

You should bookmark the following two pages. They are the authoritative references for Insomnia usage, policies, and system details.

- **Insomnia – Technical Information**

    https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145136/Insomnia+-+Technical+Information

- **Insomnia – HPC Cluster User Documentation**

    https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145124/Insomnia+HPC+Cluster+User+Documentation


These pages cover:

- Login and authentication
- Job scheduling (Slurm)
- Storage policies
- Software modules
- GPU queues
- Best practices

### 2. Logging In (SSH + Duo)

You will access Insomnia via SSH using your **UNI**.

```bash
ssh <UNI>@insomnia.rcs.columbia.edu
```

You will then see a **Duo two-factor authentication prompt**, for example:

```bash
Duo two-factor login for UNI
Enter a passcode or select:
1. Duo Push
2. Phone call
3. SMS passcodes

Passcode or option (1-3):
```

- Complete the Duo step
- Enter your UNI password when prompted

If you have issues with Duo, see:

[https://cuit.columbia.edu/cuit/duo](http://null/)

### 3. Storage Quotas (Important)

Your account is assigned:

- **50 GB block quota**
- **150,000 file quota**

Check your usage regularly:

```bash
# Check file (inode) usage
df -i /insomnia001/home/<UNI>

# Check storage (block) usage
df -h /insomnia001/home/<UNI>
```

### 4. Create Your Scratch Working Directory

All active work should be done in **scratch storage** (not home).

```bash
mkdir /insomnia001/depts/edu/users/<UNI>
cd /insomnia001/depts/edu/users/<UNI>
```

Replace `<UNI>` with your Columbia UNI.

### 5. IMPORTANT: Storage Is Scratch-Only (Not Backed Up)

- Insomnia storage **is not backed up**
- Files may be deleted without recovery
- Do **not** store the only copy of important work on Insomnia

Use **Globus** to move important files to:

- Google Drive
- Box
- Dropbox
- Your local machine

**Globus endpoint:** `Insomnia-CUIT`

Globus: https://www.globus.org/

### 6. Software & Modules

Insomnia uses a **module system** to manage software.

Common commands:

```bash
module avail
module load <module_name>
module list
```

Always load required modules inside your job scripts.

### 7. GPU Overview (What's Available)

Insomnia provides **shared GPU resources** intended for research and coursework.

Depending on queue availability, you may encounter:

- **NVIDIA V100 GPUs**
- **NVIDIA A100 GPUs**
- Multi-GPU nodes (varies by partition)

Key points:

- GPUs are accessed via **Slurm job submissions**, not directly on login nodes
- GPU access is time-shared
- Always request only the resources you need
- Long-running jobs should be planned carefully

GPU and partition details are listed here:

https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145136/Insomnia+-+Technical+Information

### 8. Training Resources

Research Computing offers short training videos covering:

- Linux command line basics
- HPC workflows
- Slurm job submission

RCS Training Videos:

[https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/38537158/RCS+Training+Videos](http://null/)

### 9. Maintenance Windows (Plan Ahead)

Insomnia has **quarterly maintenance windows**:

- Second week of **March**
- Second week of **June**
- Second week of **September**
- Second week of **December**

During these periods:

- Jobs may be interrupted
- The cluster may be unavailable

Always plan long jobs accordingly.

### 10. Getting Help

If you encounter issues:

- Email Research Computing Support:

    [**hpc-support@columbia.edu**](http://null/)


### Final Notes

- Use Insomnia responsibly — it is a **shared academic resource**
- Clean up unused files
- Avoid running heavy workloads on login nodes
- Read the documentation before submitting large jobs

## Common Mistakes (Please Read)

These are very common issues for first-time HPC users:

- Running heavy computations on login nodes (login nodes are for editing and job submission only).
- Forgetting to request GPUs in Slurm — your code will otherwise run on CPU.
- Exceeding storage or file quotas, causing jobs to fail.
- Storing important data only on Insomnia (storage is scratch-only).
- Forgetting to load required modules inside the job script.
- Submitting large jobs without testing on a small scale first.

Avoiding these will save you (and the TAs) significant time.

## Appendix: Example Slurm GPU Job Script

Below is a **minimal Slurm script** requesting **one GPU**.

Save this as `run_gpu_job.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=example_gpu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

module purge
module load cuda

python train.py
```

Submit and monitor your job:

```bash
sbatch run_gpu_job.sbatch
squeue -u <UNI>
scancel <JOB_ID>
```

---

## Verified Insomnia Configuration (March 25, 2026)

The following was discovered through hands-on testing. The instructor's post above
contains some outdated information.

### Slurm Account & Partitions

The instructor's example uses `--partition=gpu` — **this partition does not exist**.

**Correct settings:**

```bash
--account=edu
--partition=burst    # or short
--qos=burst          # must match partition
```

Check your associations: `sacctmgr show associations user=<UNI> format=Account%20,Partition%20,QOS%30`

### Actual GPU Inventory (from `sinfo`)

The README above lists V100/A100. The actual GPUs on `short`/`burst` partitions are:

| GPU | Per Node | Nodes | Total GPUs |
|-----|----------|-------|------------|
| **H100** | 2 | 2–3 | 4–6 |
| **A6000** | 8 | 13 | 104 |
| **A6000** | 4 | 2 | 8 |
| **L40** | 2 | 3 | 6 |
| **L40S** | 2 | 8 | 16 |

Lab-specific partitions (e.g., `pmg1`, `friesner1`, `morpheus1`) have additional GPUs
but are restricted to their respective groups.

### CUDA Version

- **CUDA 12.9** is installed on compute nodes at `/usr/local/cuda` (symlink to `/usr/local/cuda-12.9/`)
- The `module load cuda/12.3` module is **broken** — it points to `/usr/local/cuda-12.3` which no longer exists (see Ed post #227)
- **Workaround**: Don't use `module load cuda`. Instead, set paths directly in your job script:
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```
- `nvcc` version: 12.9, V12.9.86

### cuDNN

- **NOT system-installed** on Insomnia (as of March 25, 2026)
- The `cuda/12.3` module references a cuDNN path at `/usr/local/lib/python3.9/site-packages/nvidia/cudnn`, but this path does not exist
- Ed post #236 confirms: no system-wide `cudnn.h` on the cluster
- Students who report cuDNN "just working" have it in their own conda/pip environments
- **Workaround**: Install locally via pip and point nvcc to it:
  ```bash
  pip install --user nvidia-cudnn-cu12
  # Find the installed paths
  find ~/.local -name "cudnn.h" 2>/dev/null
  find ~/.local -name "libcudnn*" 2>/dev/null
  # Create the missing symlink (pip only installs libcudnn.so.9, linker needs libcudnn.so)
  cd ~/.local/lib/python3.9/site-packages/nvidia/cudnn/lib/
  ln -s libcudnn.so.9 libcudnn.so
  # Compile with explicit paths
  nvcc c3.cu -o c3 -O3 \
    -I$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/include \
    -L$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/lib \
    -lcudnn
  # At runtime, also set LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  ```

### Login Node vs Compute Nodes

- **Login node**: No GPU, no CUDA toolkit, no `nvidia-smi`. Used for editing, file transfer, and job submission only.
- **Compute nodes** (via `srun`/`sbatch`): Have GPUs, CUDA 12.9, `nvidia-smi`.
- Even **compilation** (`nvcc`) must happen on a compute node.

### Working Slurm Job Template

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

# CUDA setup (don't use module load cuda, it's broken)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Include cuDNN if needed (pip install --user nvidia-cudnn-cu12)
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Your commands here
nvcc -O3 my_program.cu -o my_program
./my_program
```

### Interactive GPU Session

For iterative development (compile, run, edit, repeat):

```bash
srun --account=edu --partition=burst --qos=burst --gres=gpu:1 --time=01:00:00 --pty bash
```

### Scratch Directory

```
/insomnia001/depts/edu/users/wax1/
```

### Python / PyTorch / Triton on Insomnia

The system Python 3.9 has an old PyTorch that requires cuDNN 8. To run Python-based
GPU code (e.g., Triton), install your own stack:

```bash
# From login node
pip install --user torch triton typing_extensions --upgrade
```

The `typing_extensions` upgrade is needed because the system version is too old for
current PyTorch. At runtime on compute nodes, set:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Requesting Specific GPUs

You can request a specific GPU type:

```bash
srun --pty -t 0-00:30 --gres=gpu:A6000:1 -A edu --partition=burst --qos=burst /bin/bash
srun --pty -t 0-00:30 --gres=gpu:h100:1 -A edu --partition=burst --qos=burst /bin/bash
srun --pty -t 0-00:30 --gres=gpu:l40s:1 -A edu --partition=burst --qos=burst /bin/bash
```

Or just `--gres=gpu:1` to get whatever is available fastest.

Note: different nodes may have different software images. CUDA 12.9 is available on
A6000 and H100 nodes via `/usr/local/cuda`. The `module load cuda/12.3` path may or
may not exist depending on the node.

### SSH Multiplexing (avoid repeated Duo 2FA)

Add to `~/.ssh/config` on your local machine:

```
Host insomnia
  HostName insomnia.rcs.columbia.edu
  User <UNI>
  ControlMaster auto
  ControlPath ~/.ssh/sockets/%r@%h-%p
  ControlPersist 4h
```

Then `mkdir -p ~/.ssh/sockets`. First `ssh insomnia` needs Duo; subsequent
`ssh`/`scp` reuse the connection for 4 hours.

### Relevant Ed Posts

- **#192**: Workaround for broken `module load cuda`
- **#209**: Build/run script for HW3 ("super script")
- **#226**: cuDNN `PREFER_FASTEST` deprecated; use `cudnnFindConvolutionForwardAlgorithm()` instead
- **#227**: CUDA permanent fix, explains the 12.3 to 12.9 symlink issue
- **#236**: Confirms no system cuDNN on Insomnia (student workaround: pip install locally)
