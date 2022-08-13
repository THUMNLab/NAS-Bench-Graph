import pickle
from architecture import all_archs

def light_read(dname):
    f = open("light/{}.bench".format(dname), "rb")
    bench = pickle.load(f)
    f.close()
    return bench

def read(name):
    f = open(name, "rb")
    bench = pickle.load(f)
    f.close()
    return bench

if __name__ == "__main__":
    read()
