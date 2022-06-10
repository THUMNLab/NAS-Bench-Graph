import pickle
from hpo import all_archs

def light_read(dname):
    f = open("light/{}.bench".format(dname), "rb")
    bench = pickle.load(f)
    f.close()
    return bench

def read():
    f = open("light/proteins.bench", "rb")
    bench = pickle.load(f)
    f.close()

    max_p = 0
    best_arch = None
    archs = all_archs()
    for arch in archs:
        hash = arch.hash_arch()
        info = bench.get(hash, None)
        if info: 
            perf = info['perf']
            if perf > max_p:
                max_p = perf 
                best_arch = arch
        else:
            continue
            print(arch.link)
            print(arch.ops)

    print(max_p)
    print(best_arch.link)
    print(best_arch.ops)

    print(len(bench))
    v = list(bench.values())
    v = [i['para'] for i in v]
    print(max(v))
    return
    for key in bench:
        print(key)
        print(bench)

if __name__ == "__main__":
    read()