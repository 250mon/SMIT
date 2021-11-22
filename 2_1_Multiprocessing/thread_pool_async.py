import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os, time


def hello(x):
    print("inside hello()")
    print("Process id: ", os.getpid())
    # move to wait queue of OS for 3 seconds (timer event was set to invoke the process)
    time.sleep(3)
    return x * x


if __name__ == "__main__":
    p = ThreadPool(5)
    res = p.map_async(hello, range(3))
    wait_num = 0

    while True:
        # return True when async mapping has been done
        ready = res.ready()
        # print('Ready Return', ready)

        if ready:
            # get computed data
            pool_output = res.get()
            break
        else:
            wait_num += 1
            print('Waiting: ', wait_num)
            # just context switching to let other process(thread) occupy CPU
            time.sleep(0)
    print(pool_output)
