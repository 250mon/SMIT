import multiprocessing as mp


class MyFancyClass():
    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = mp.current_process().name
        print(f'Doing something fancy in {proc_name} for {self.name}!')


def worker(q):
    obj = q.get()
    obj.do_something()


if __name__ == '__main__':
    queue = mp.Queue()
    p = mp.Process(target=worker, args=(queue,))
    p.start()

    queue.put(MyFancyClass('Fancy Dan'))
    # Wait for the worker to finish
    queue.close()
    # join_thread() can be called only after close()
    queue.join_thread()
    # block until the background thread(managing the Queue) is terminated
    p.join()
