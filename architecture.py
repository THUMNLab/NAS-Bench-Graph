import random
import json

link_list = [
[0,0,0,0],
[0,0,0,1],
[0,0,1,1],
[0,0,1,2],
[0,0,1,3],
[0,1,1,1],
[0,1,1,2],
[0,1,2,2],
[0,1,2,3]
]

gnn_list = [
    "gat",  # GAT with 2 heads
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  # k-GNN
    "fc",  # fully-connected
    "skip"  # skip connection
]

gnn_list_proteins = [
    "gcn",  # GCN
    "sage",  # sage
    "arma",
    "fc",  # fully-connected
    "skip"  # skip connection
]

class Arch:
    def __init__(self, lk=None, op=None):
        self.link = lk
        self.ops = op

    def random_arch(self):
        self.ops = []
        self.link = random.choice(link_list)
        for i in self.link:
            self.ops.append(random.choice(gnn_list))

    def hash_arch(self, use_proteins = False):
        lk = self.link
        op = self.ops
        if use_proteins:
            gnn_g = {name: i for i, name in enumerate(gnn_list_proteins)}
            b = len(gnn_list_proteins) + 1
        else:
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            b = len(gnn_list) + 1
        if lk == [0,0,0,0]:
            lk_hash = 0
        elif lk == [0,0,0,1]:
            lk_hash = 1
        elif lk == [0,0,1,1]:
            lk_hash = 2
        elif lk == [0,0,1,2]:
            lk_hash = 3
        elif lk == [0,0,1,3]:
            lk_hash = 4
        elif lk == [0,1,1,1]:
            lk_hash = 5
        elif lk == [0,1,1,2]:
            lk_hash = 6
        elif lk == [0,1,2,2]:
            lk_hash = 7
        elif lk == [0,1,2,3]:
            lk_hash = 8

        for i in op:
            lk_hash = lk_hash * b + gnn_g[i]
        return lk_hash

    def regularize(self):
        lk = self.link[:]
        ops = self.ops[:]
        if lk == [0,0,0,2]:
            lk = [0,0,0,1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,0,3]:
            lk = [0,0,0,1]
            ops = [ops[2], ops[0], ops[1], ops[3]]
        elif lk == [0,0,1,0]:
            lk = [0,0,0,1]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        elif lk == [0,0,2,0]:
            lk = [0,0,0,1]
            ops = [ops[1], ops[0], ops[3], ops[2]]
        elif lk == [0,0,2,1]:
            lk = [0,0,1,2]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,2,2]:
            lk = [0,0,1,1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,2,3]:
            lk = [0,0,1,3]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,1,0,0]:
            lk = [0,0,0,1]
            ops = [ops[0], ops[2], ops[3], ops[1]]
        elif lk == [0,1,0,1]:
            lk = [0,0,1,1]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,0,2]:
            lk = [0,0,1,3]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,0,3]:
            lk = [0,0,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,1,0]:
            lk = [0,0,1,1]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0,1,1,3]:
            lk = [0,1,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,2,0]:
            lk = [0,0,1,3]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0,1,2,1]:
            lk = [0,1,1,2]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        return Arch(lk, ops)

    def equalpart_sort(self):
        lk = self.link
        op = self.ops
        ops = op[:]
        def part_sort(ids, ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            opli = [gnn_g[ops[i]] for i in ids]
            opli.sort()
            for posid, opid in zip(ids, opli):
                ops[posid] = gnn_list[opid]
            return ops

        def sort0012(ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            if gnn_g[op[0]] > gnn_g[op[1]] or op[0] == op[1] and gnn_g[op[2]] > gnn_g[op[3]]:
                ops = [ops[1], ops[0], ops[3], ops[2]]
            return ops

        if lk == [0,0,0,0]:
            ids = [0,1,2,3]
        elif lk == [0,0,0,1]:
            ids = [1,2] 
        elif lk == [0,0,1,1]:
            ids = [2,3] 
        elif lk == [0,0,1,2]:
            ids = None
            ops = sort0012(ops)
        elif lk == [0,1,1,1]:
            ids = [1,2,3] 
        elif lk == [0,1,2,2]:
            ids = [2,3] 
        else:
            ids = None

        if ids:
            ops = part_sort(ids, ops)

        self.ops = ops

    def move_skip_op(self):
        link = self.link[:]
        ops = self.ops[:]
        def move_one(k, link, ops):
            ops = [ops[k]] + ops[:k] + ops[k + 1:]
            for i, father in enumerate(link):
                if father == k + 1:
                    link[i] = link[k]
                if father <= k:
                    link[i] = link[i] + 1
            link = [0] + link[:k] + link[k + 1:]
            return link, ops

        def check_dim(k, link, ops):
            # check if a dimension is original dimension
            while k > -1:
                if ops[k] != 'skip':
                    return False
                k = link[k] - 1
            return True

        for i in range(len(link)):
            if ops[i] != 'skip':
                continue
            son = False
            brother = False
            for j, fa in enumerate(link):
                if fa == i + 1:
                    son = True
                elif j != i and fa == link[i]:
                    brother = True
            if son or not brother or check_dim(i, link, ops) and not son:
                link, ops = move_one(i, link, ops)

        if link == [0,1,2,1]:
            link = [0,1,1,2]
            ops = ops[:2] + [ops[3], ops[2]]
        elif link == [0,1,1,3]:
            link = [0,1,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]

        #if link not in link_list:
        #    print(lk, link)
            
        self.link = link
        self.ops = ops

    def valid_hash(self):
        b = self.regularize()
        b.move_skip_op()
        b.equalpart_sort()
        return b.hash_arch()

    def check_isomorph(self):
        link, ops = self.link, self.ops
        linkm = link[:]
        opsm = ops[:]
        self.move_skip_op()
        self.equalpart_sort()
        #print(self.link, self.ops)
        return linkm == self.link and opsm == self.ops

def check_isom():
    lk = [0, 0, 0, 0]
    op = ['skip', 'gcn', 'gcn', 'skip']
    arch = Arch(lk, op)
    print(arch.check_isomorph())

def all_archs():
    # all combination of link_list and ops
    ng = len(gnn_list)
    archs = []
    for i in gnn_list:
        for j in gnn_list:
            for k in gnn_list:
                for l in gnn_list:
                    if i == 'skip' and j == 'skip' and k == 'skip' and l == 'skip':
                        continue
                    for lk in link_list:
                        arch = Arch(lk, [i, j, k, l])
                        if arch.check_isomorph():
                            archs.append(arch)

    print(len(archs))
    return archs

class HP:
    def __repr__(self):
        return json.dumps(self, default=lambda obj: obj.__dict__)

def random_hp():
    randfloat = lambda x, y: x + (y - x) * random.random()
    hp = HP()
    hp.dropout = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    hp.dim = random.choice((64, 128, 256))
    hp.num_cells = 1# random.randint(1,2)
    hp.num_pre = 0#random.randint(0, 1)
    hp.num_pro = random.randint(0, 1)

    #hp.lr = random.choice((0.05, 0.02, 0.01, 0.005, 0.002, 0.001))
    hp.lr = random.choice((0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001))
    hp.wd = random.choice((0, 5e-4))
    #hp.optimizer = 'Adam'
    hp.optimizer = random.choice(('Adam', 'SGD'))
    hp.num_epochs = random.choice((200, 300, 400, 500))
    return hp

anchors = [
    Arch([0, 1, 2, 3], ['skip', 'skip', 'gcn', 'gcn']),
    Arch([0, 1, 2, 3], ['skip', 'skip', 'gat', 'gat']),
    Arch([0, 1, 2, 3], ['skip', 'skip', 'gin', 'gin']),
    Arch([0, 1, 2, 3], ['skip', 'skip', 'graph', 'graph']),
    Arch([0, 1, 2, 3], ['skip', 'skip', 'sage', 'sage']),
    Arch([0, 1, 2, 3], ['skip', 'gcn', 'gcn', 'gcn']),
    Arch([0, 1, 2, 3], ['skip', 'gat', 'gat', 'gat']),
    Arch([0, 1, 2, 3], ['skip', 'gin', 'gin', 'gin']),
    Arch([0, 1, 2, 3], ['skip', 'graph', 'graph', 'graph']),
    Arch([0, 1, 2, 3], ['skip', 'sage', 'sage', 'sage'])
]

if __name__ == '__main__':
    #all_archs()
    check_isom()
