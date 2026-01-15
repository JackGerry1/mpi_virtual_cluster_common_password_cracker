

from mpi4py import MPI
import hashlib
from tqdm import tqdm
import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import argparse


hash_names = ['sha3_256', 'sha3_512', 'bcrypt', 'argon2']

def crack_hash_parallel(hash, wordlist, hash_type=None):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    if hash_type not in hash_names:
        raise ValueError(f'[!] Invalid hash type: {hash_type}, supported are {hash_names}')

    is_bcrypt = hash_type == 'bcrypt'
    is_argon2 = hash_type == 'argon2'
    hash_fn = None if is_bcrypt or is_argon2 else getattr(hashlib, hash_type, None)

    # rank == 0 = master node
    # read wordlist and divide into chunks based on the number of nodes
    if rank == 0:

        with open(wordlist, 'r') as f:
            words = [line.strip() for line in f]
        
        # distribute evenly across nodes
        total_lines = len(words)
        chunk_size = (total_lines + size - 1) // size
        chunks = [words[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)

    password_found = False
    found_password = None

    # handle bcrypt 
    if is_bcrypt:
        for word in tqdm(chunk, desc=f'Rank {rank} processing bcrypt', leave=False):
        
            if bcrypt.checkpw(word.encode(), hash.encode()):
                found_password = word
                password_found = True
                break
                
    # handle argon2 
    elif is_argon2:
        ph = PasswordHasher()

        for word in tqdm(chunk, desc=f'Rank {rank} processing Argon2', leave=False):
            if ph.verify(hash, word):
                found_password = word
                password_found = True
                break

    # Handle sha3-256 and sha3-512 hashes
    else:
        for word in tqdm(chunk, desc=f'Rank {rank} processing', leave=False):
            
            if hash_fn(word.encode()).hexdigest() == hash:
                found_password = word
                password_found = True
                break

    password_found = comm.bcast(password_found, root=0)

    # If password found, abort all processes after displaying a message
    # probably a better way to do this but it was the only way i could figure to stop the program running if one rank found a password
    # without this code if rank 0 found the password in ten seconds rank 1 will still be processing until it reached the end of the chunk
    # resulting in the time taken to run the program being inaccurate. 
    if password_found:
        print("=============================================")
        print(f"[+] FOUND PASSWORD: {str(found_password)}" )
        print("=============================================")
        comm.Abort()

    all_results = comm.gather(found_password, root=0)

    # Master node: Check results and terminate early if a password is found
    if rank == 0:
        for result in all_results:
            if result:
                return result
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Crack a hash using a wordlist with MPI.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--hash-file', help='Path to a file containing the hash to crack.')  

    parser.add_argument('wordlist', help='The path to the wordlist.')

    parser.add_argument('--hash-type', help='The hash type to use.')

    args = parser.parse_args()

    try:
        with open(args.hash_file, 'r') as f:
            hash_to_crack = f.read().strip()  
    
    except Exception as e:
        raise ValueError(f"[!] Unable to read hash file: {e}")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  

    if rank == 0:
        print(f"[*] Starting hash cracking with {comm.Get_size()} nodes.")

    crack_hash_parallel(hash_to_crack, args.wordlist, args.hash_type)
    
    if rank == 0:
        print("[-] Password not found in wordlist.")
