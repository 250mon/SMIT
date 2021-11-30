import multiprocessing as mp
from multiprocessing import shared_memory
import time
import numpy as np
import copy

# BUFFER_SIZE of images
BUFFER_SIZE = 1000
H = 2
W = 2
C = 2

def pd_routine(shm, mshape, mtype, lock):
    # attach to the existing shared memory block
    shm = shared_memory.SharedMemory(name=shm.name)
    # sh_array = np.ndarray(mshape, dtype=np.ubyte, buffer=shm.buf, offset=np.ubyte().itemsize)
    sh_array = np.ndarray(mshape, dtype=mtype, buffer=shm.buf, offset=1)
    for i in range(10000):
        lock.acquire()
        sh_array[shm.buf[0]] = np.ones((H, W, C), dtype=np.int32)
        shm.buf[0] += 1
        lock.release()
        print(f'produced {i}th array!')
        time.sleep(0)
    # finally before shm.unlike() to free and release shared memory block
    shm.close()

def cs_routine(shm, mshape, mtype, lock):
    waiting_count = 0
    count = 0
    shm = shared_memory.SharedMemory(name=shm.name)
    while waiting_count < 100:
        lock.acquire()
        # immutable; copied
        sh_counter = shm.buf[0]
        # mutable; referenced
        # sh_array = np.ndarray(mshape, dtype=np.ubyte, buffer=shm.buf, offset=np.ubyte().itemsize)
        sh_array = np.ndarray(mshape, dtype=mtype, buffer=shm.buf, offset=1)
        # copy contents from shared memory to local variables
        buffer = copy.copy(sh_array)
        shm.buf[0] = 0
        lock.release()
        # handling local variables
        if sh_counter == 0:
            waiting_count += 1
        else:
            count += sh_counter
            for i in range(sh_counter):
                buffer += 1
            print(f'count={count}')
            waiting_count = 0
        time.sleep(0)

    # finally before shm.unlike() to free and release shared memory block
    shm.close()


if __name__ == '__main__':
    # need to create an array on byte basis
    arr = np.zeros((BUFFER_SIZE, H, W, C), dtype=np.int32)
    # [0, arr] - 0 is the initial value of counter, ie., and integer value
    # mem = np.array([0, arr])
    mem = arr

    # create a shared memory block of the same size as arr
    shm = shared_memory.SharedMemory(create=True, size=mem.nbytes+1)
    # assign mem to shared memory
    # shm.buf = mem
    # lock
    lock = mp.Lock()

    # Producer = mp.Process(target=pd_routine, args=(shm.name, mem.shape, mem.dtype))
    # Consumer = mp.Process(target=cs_routine, args=(shm.name, mem.shape, mem.dtype))
    Producer = mp.Process(target=pd_routine, args=(shm, mem.shape, mem.dtype, lock,))
    Consumer = mp.Process(target=cs_routine, args=(shm, mem.shape, mem.dtype, lock,))

    Producer.start()
    Consumer.start()
    shm.unlink()
