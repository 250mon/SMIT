def fib(n):
    if n == 2:
        return 2
    elif n > 2:
        npp = fib(n-2)
        np = fib(n-1)
        return np + npp
    else:
        return 1

def fib2(n):
    if n < 2:
        return 1

    seq = [1, 1]
    for i in range(2, n+1):
        seq.append(seq[i-1] + seq[i-2])
    return seq[n]


if __name__ == "__main__":
    print(fib2(40))
    print(fib(40))
