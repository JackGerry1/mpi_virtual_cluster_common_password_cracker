# MPI Parallel Hash Cracker

A **distributed, MPI-based hash cracking tool** written in Python that leverages multiple nodes to crack password hashes using a wordlist.  
Supported hash algorithms include:

- **SHA3-256**
- **SHA3-512**
- **bcrypt**
- **Argon2**

This project is designed to run on a **Linux cluster** (virtual or physical) using **OpenMPI**, making it suitable for learning high-performance computing (HPC), MPI programming, and distributed security tooling.

---

## Features

- Distributed password cracking using **MPI**
- Scales across multiple VMs or cluster nodes
- Secure hash verification for bcrypt and Argon2
- Progress bars per MPI rank using `tqdm`
- Early termination across all nodes once a password is found

---

## How It Works

1. The **master node (rank 0)**:
   - Loads the wordlist
   - Splits it into chunks
   - Distributes chunks to worker nodes

2. Each node:
   - Hashes or verifies passwords in parallel
   - Stops immediately if a match is found

3. Once a password is discovered:
   - The result is broadcast to all nodes
   - MPI aborts remaining processes to ensure accurate timing

---

## Supported Hash Types

| Hash Type  | Notes |
|----------|-------|
| `sha3_256` | Uses Python `hashlib` |
| `sha3_512` | Uses Python `hashlib` |
| `bcrypt` | Uses `bcrypt.checkpw()` |
| `argon2` | Uses `argon2.PasswordHasher` |

---

## Requirements

### Software
- Ubuntu Server (or similar Linux distro)
- Python 3.8+
- OpenMPI
- VirtualBox or VMware (for virtual clusters)

### Python Dependencies
```bash
pip install mpi4py tqdm bcrypt argon2-cffi
```

## System Dependencies
```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

## Usage
1. Prepare Inputs
   - Hash file – contains only the hash:
   - ```bash
     $argon2id$v=19$m=65536,t=3,p=4$...
     ```
   - Wordlist file – one password per line

2. Run with MPI
   - Hash file – contains only the hash:
   - ```bash
     mpirun -np 4 python crack.py --hash-file hash.txt wordlist.txt --hash-type argon2
     ```
   - Where:
   - -np 4 → number of MPI processes / nodes
   - --hash-type → one of:
     - sha3_256
     - sha3_512
     - bcrypt
     - argon2
    
## Example Output
```bash
[*] Starting hash cracking with 4 nodes.
=============================================
[+] FOUND PASSWORD: password123
=============================================
```

## Cluster Setup
A full step-by-step cluster guide is included in the project documentation, covering everything from OS install, including SSH and MPI setup to benchmarking with HPL.: 
