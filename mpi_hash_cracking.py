
# import libraries required for hashing and MPI
from mpi4py import MPI
import hashlib
from tqdm import tqdm
import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import argparse

# List of supported hash types
hash_names = ['sha3_256', 'sha3_512', 'bcrypt', 'argon2']

def crack_hash_parallel(hash, wordlist, hash_type=None):
    # initialise MPI rank and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Validate the hash type
    if hash_type not in hash_names:
        raise ValueError(f'[!] Invalid hash type: {hash_type}, supported are {hash_names}')

    # Handle bcrypt and Argon2 separately
    is_bcrypt = hash_type == 'bcrypt'
    is_argon2 = hash_type == 'argon2'
    hash_fn = None if is_bcrypt or is_argon2 else getattr(hashlib, hash_type, None)

    # rank == 0 = master node
    # read wordlist and divide into chunks based on the number of nodes
    if rank == 0:
        # open file and read every word excluding spaces
        with open(wordlist, 'r') as f:
            words = [line.strip() for line in f]
        
        # obtain total line number and distribute evenly across nodes
        total_lines = len(words)
        chunk_size = (total_lines + size - 1) // size
        chunks = [words[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    # Scatter wordlist chunks to all nodes
    chunk = comm.scatter(chunks, root=0)

    # Shared variable to indicate if password was found
    password_found = False
    found_password = None

    # Handle bcrypt hashes
    if is_bcrypt:
        for word in tqdm(chunk, desc=f'Rank {rank} processing bcrypt', leave=False):
            # stop processing if password is already found
            if password_found:
                break
            # Check if the current passwoird matches the bcrypt hash, if it does store it and set password_found to true. 
            if bcrypt.checkpw(word.encode(), hash.encode()):
                found_password = word
                password_found = True
                break

    # Handle Argon2 hashes
    elif is_argon2:
        # Create an instance of the Argon2 PasswordHasher
        ph = PasswordHasher()

        # loop through all words in chunk on different ranks 0 = master 1 = slave1 and so on
        for word in tqdm(chunk, desc=f'Rank {rank} processing Argon2', leave=False):
            # stop processing if password is already found
            if password_found:
                break  
            try:
                # Verify if the current word matches the Argon2 hash and store the password and set password_found to true 
                if ph.verify(hash, word):
                    found_password = word
                    password_found = True
                    break

            # Continue to the next word if the verification fails (no match)
            except VerifyMismatchError:
                continue

    # Handle sha3-256 and sha3-512 hashes
    else:
        # loop through all words in chunk on different ranks 0 = master 1 = slave1 and so on
        for word in tqdm(chunk, desc=f'Rank {rank} processing', leave=False):
            
            # stop processing if password is already found
            if password_found:
                break  
            
            # see if hash == hash of current word in wordlist, meaning that it has found the password 
            # set password_found to true and break
            if hash_fn(word.encode()).hexdigest() == hash:
                found_password = word
                password_found = True
                break

    # Broadcast the found password status to all nodes
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

    # Gather the found password results to the master node
    all_results = comm.gather(found_password, root=0)

    # Master node: Check results and terminate early if a password is found
    if rank == 0:
        for result in all_results:
            if result:
                return result
        return None

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description='Crack a hash using a wordlist with MPI.')

    # Create a mutually exclusive group to ensure only one of these options is provided
    # to provide a hash of a password through a .txt file 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--hash-file', help='Path to a file containing the hash to crack.')  

    # Argument for the wordlist file, which is the file containing common passwords.  
    parser.add_argument('wordlist', help='The path to the wordlist.')

    # Specifying the hash type for sha3-256, sha3-512, bcrypt and argon2
    parser.add_argument('--hash-type', help='The hash type to use.')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Read the hash from the file if '--hash-file' is specified
    try:
        # Attempt to open and read the hash file and strip any extra whitespace
        with open(args.hash_file, 'r') as f:
            hash_to_crack = f.read().strip()  
    
    # Raise an error if the file cannot be read
    except Exception as e:
        raise ValueError(f"[!] Unable to read hash file: {e}")

    # Initialise MPI communication and get the rank (ID) of the current MPI process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  

    # Print a startup message from the root node (rank 0 master node)
    if rank == 0:
        print(f"[*] Starting hash cracking with {comm.Get_size()} nodes.")

    # Call the parallel hash cracking function
    crack_hash_parallel(hash_to_crack, args.wordlist, args.hash_type)

    # Return a message if the password is not found
    if rank == 0:
        print("[-] Password not found in wordlist.")
