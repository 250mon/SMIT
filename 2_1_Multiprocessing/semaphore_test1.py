import multiprocessing as mp

if __name__ == "__main__":
    s = [mp.Semaphore(i) for i in range(3)]
    bs3 = mp.BoundedSemaphore(3)

    for i in range(3):
        print(f"s{i}: ", s[i])
        s[i].release()
        # s[i].acquire()
        print(f"released s{i}): ", s[i])

    print("released bs: ", bs3.release())

