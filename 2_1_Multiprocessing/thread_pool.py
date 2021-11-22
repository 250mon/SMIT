from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
import os, time


def hello(x):
    print("inside hello()")
    print("Process id: ", os.getpid())
    # move to wait queue of OS for 3 seconds (timer event was set to invoke the process)
    time.sleep(3)
    return x * x


if __name__ == "__main__":
    # p = ThreadPool(5)
    p = Pool(5)
    # mapping data in iterable to a thread in Pool, execute concurrently
    # get each result, return the collected results as a list
    pool_output = p.map(hello, range(5))
    print(pool_output)
