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

### 7. GPU Overview (What’s Available)

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