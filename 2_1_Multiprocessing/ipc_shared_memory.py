import multiprocessing as mp
from multiprocessing import shared_memory
import time
import numpy as np
import copy

# BUFFER_SIZE of images
BUFFER_SIZE = 10000
H = 2
W = 2
C = 2

# producer
def pd_routine(mname, mshape, mtype, lock):
    shm = shared_memory.SharedMemory(name=mname)
    # attach to the existing shared memory block
    attached_counter = shm.buf
    attached_array = np.ndarray(mshape, dtype=mtype, buffer=shm.buf, offset=1)
    # attached array address is different from that of cs_routine
    # print(f"pd... shm addr: {id(attached_array[0])}")

    for i in range(10000):
        lock.acquire()
        attached_array[attached_counter[0]] = np.ones((H,), dtype=np.int32) * i
        attached_counter[0] += 1
        lock.release()
        print(f'produced {i}th array! {attached_array[0]}')
        time.sleep(0.001)
    # finally before shm.unlike() to free and release shared memory block
    shm.close()

# consumer
def cs_routine(mname, mshape, mtype, lock):
    shm = shared_memory.SharedMemory(name=mname)
    # attach to the existing shared memory block
    attached_counter = shm.buf
    attached_array = np.ndarray(mshape, dtype=mtype, buffer=shm.buf, offset=1)
    # attached array address is different from that of pd_routine
    # print(f"\t\tcs... shm addr: {id(attached_array[0])}")

    while True:
        lock.acquire()
        counter = attached_counter[0]
        # print(f"\tconsuming {counter}...")
        # shared mem counter reset
        attached_counter[0] = 0
        lock.release()

        # consuming
        if counter != 0:
            for i in range(counter):
                print(f"\t\tconsuming... {attached_array[i]}")
        time.sleep(0)

    # finally before shm.unlike() to free and release shared memory block
    shm.close()


if __name__ == '__main__':
    # need to create an array on byte basis
    mem = np.zeros((BUFFER_SIZE, H), dtype=np.int32)
    # create a shared memory block of the same size as arr
    # [0, array] - 0 is a counter(int8) which indicates the length of the data
    shm = shared_memory.SharedMemory(create=True, size=mem.nbytes + 1)
    lock = mp.Lock()

    Producer = mp.Process(target=pd_routine, args=(shm.name, mem.shape, mem.dtype, lock,))
    Consumer = mp.Process(target=cs_routine, args=(shm.name, mem.shape, mem.dtype, lock,))

    Producer.start()
    Consumer.start()
    shm.unlink()
