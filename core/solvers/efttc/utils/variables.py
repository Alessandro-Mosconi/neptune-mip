import numpy as np

#TODO: check var name 
def init_x(data, x):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                name = f"x[{i}][{f}][{j}]"
                x[(i, f, j)] = {"name": name, "val": 0}

def init_c(data,  c):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            name = "c[{f}][{i}]"
            c[(f, i)] = {"name": name, "val": False}
        

def init_n(data, n):
    for i in range(len(data.nodes)):
        n[i] = {"name": f"n[{i}]", "val": False}

def init_moved_from(data, moved_from):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            moved_from[(f, i)] = {"name": f"moved_from[{f}][{i}]", "val": 0}

def init_moved_to(data, moved_to):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            moved_to[(f, i)] = {"name": f"moved_to[{f}][{i}]", "val": 0}

def init_allocated(data):
    return {"name": "allocated", "val": 0}

def init_deallocated(data):
    return {"name": "deallocated", "val": 0}
