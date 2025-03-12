import os

num_memories = [1,3]

for nb_mem in num_memories:
    print("nb_mem: ", nb_mem)
    file_out = f"out_nbmem={nb_mem}"
    os.system(f"python train_parallel.py --num_memories {nb_mem} > {file_out}")
