from multiprocessing import Process
import os


def info(title):
    print(title)
    print('module name: ', __name__)
    print('parent process: ', os.getppid())
    print('process id: ', os.getpid())

def f(name):
    info('function f')
    print('hello ', name)


if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    print('waiting for child process is completed\n')

    block_num = 0
    while True:
        # block for 0.001sec
        p.join(0.001)
        # exitcode is None when it is running, 0 when terminated, -N when forced to be quit
        status = p.exitcode
        if status is None:
            block_num += 1
            print(f'waiting still for the child process to be terminated ({block_num})')
        else:
            print(f'child process status: {status}')
            break

    print('child process is terminated and removed from OS')