import multiprocessing
import threading, time

shared_number = 0
lock = multiprocessing.Lock()

def thread_1(number):
    global shared_number, lock
    print("number = ", number)
    for i in range(number):
        lock.acquire()
        shared_number += 1
        lock.release()


def thread_2(number):
    global shared_number, lock
    print("number = ", number)
    for i in range(number):
        lock.acquire()
        shared_number += 1
        lock.release()


if __name__ == "__main__":
    threads = []
    local_num = 500000

    start_time = time.time()

    t1 = threading.Thread(target=thread_1, args=(local_num,))
    t1.start()
    threads.append(t1)

    t2 = threading.Thread(target=thread_1, args=(local_num,))
    t2.start()
    threads.append(t2)

    for t in threads:
        t.join()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("shared_number= ", shared_number)

