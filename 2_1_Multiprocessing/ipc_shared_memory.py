import multiprocessing as mp
import numpy as np


# BUFFER_SIZE of images
arr = np.zeros(BUFFER_SIZE, H, W, C)
# [0, arr] - 0 is the initial value of counter, ie., and integer value
mem = np.array([0, arr])

# create a shared memory block of the same size as arr
shm = mp.shared_memory.SharedMemory(create=True, size=mem.nbytes)
# assign mem to shared memory
shm.buf = mem

Producer = mp.Process(target=pd_routine, args=(shm.name, mem.shape, mem.dtype))
Consumer = mp.Process(target=cs_routine, args=(shm.name, mem.shape, mem.dtype))

Producer.start()
Consumer.start()

def pd_routine():
    # attach to the existing shared memory block
    shm = mp.shared_memory.SharedMemory(name=shm.name)
    temp = np.ndarray(mem.shape, dtype=mem.dytpe, buffer=shm.buf)
    counter = temp[0]
    buffer = temp[1]

    # finally before shm.unlike() to free and release shared memory block
    shm.close()

def cs_routine():
    shm = mp.shared_memory.SharedMemory(name=shm.name)
    temp = np.ndarray(mem.shape, dtype=mem.dytpe, buffer=shm.buf)
    counter = temp[0]
    buffer = temp[1]

    # finally before shm.unlike() to free and release shared memory block
    shm.close()
