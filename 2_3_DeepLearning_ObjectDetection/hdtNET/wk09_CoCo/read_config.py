from operator import methodcaller


def read_config(file_name):
    with open(file_name, 'r') as fd:
        lines = fd.readlines()
    lines = map(methodcaller('strip'), lines)
    lines = list(map(methodcaller('split', ";"), lines))
    res_dict = {lines[i][0]: lines[i][1] for i in range(len(lines))}
    return res_dict
