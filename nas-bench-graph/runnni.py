from modulefinder import Module
import torch
import nni
import sys
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from nni.retiarii.evaluator import FunctionalEvaluator
import nni.retiarii.strategy as strategy

from architecture import gnn_list, gnn_list_proteins
from architecture import all_archs, HP, Arch
from readbench import light_read
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
dnames = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']
op_proteins = ['gcn', 'arma', 'cheb', 'fc', 'skip']
#op_proteins = gnn_list

# model
class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.name = lambd

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)

    
def map_nn(l):
    return nn.LayerChoice([StrModule(x) for x in l])

def map_value(l, label):
    return nn.ValueChoice(l, label = label)

@model_wrapper
class Space_rl(nn.Module):
    def __init__(self):
        super().__init__()
        self.lk0 = 0
        self.op0 = map_value(op_proteins, "op0")
        self.lk1 = map_value([0, 1], "lk1")
        self.op1 = map_value(op_proteins, "op1")
        self.lk2 = map_value([0, 1, 2], "lk2")
        self.op2 = map_value(op_proteins, "op2")
        self.lk3 = map_value([0, 1, 2, 3], "lk3")
        self.op3 = map_value(op_proteins, "op3")
        
    def forward(self, bench):
        lks = [getattr(self, "lk" + str(i)) for i in range(4)]
        ops = [getattr(self, "op" + str(i)) for i in range(4)]
        arch = Arch(lks, ops)
        h = arch.valid_hash()
        if h == "88888":
            return 0
        return bench[h]['perf']

@model_wrapper
class Space(nn.Module):
    def __init__(self):
        super().__init__()
        self.links = nn.ModuleList()
        self.ops = nn.ModuleList()
        
        for i in range(4):
            self.links.append(map_nn(range(i + 1)))
            self.ops.append(map_nn(op_proteins))
        
    def forward(self, bench):
        lks = [i.name for i in self.links]
        ops = [i.name for i in self.ops]
        arch = Arch(lks, ops)
        h = arch.valid_hash()
        if h == "88888":
            return 0
        return bench[h]['perf']

    def seeinsed(self):
        for i in self.ops:
            print(i.name)

def evaluate_model(model_cls, dname):
    bench = light_read(dname)
    model = model_cls()
    ans = model(bench)
    nni.report_final_result(ans)
    return ans

def atest(nas, dname, port):
    space = Space_rl()
    evaluator = FunctionalEvaluator(evaluate_model, dname=dname)
    if nas == "random":
        exploration_strategy = strategy.Random(dedup=True)
    elif nas == "ea":
        exploration_strategy = strategy.RegularizedEvolution()
    elif nas == "rl":
        exploration_strategy = strategy.PolicyBasedRL()

    exp = RetiariiExperiment(space, evaluator, [], exploration_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.max_trial_number = 40   # spawn 4 trials at most
    exp_config.trial_concurrency = 10  # will run two trials concurrently
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True
    exp.run(exp_config, port)
    for model_dict in exp.export_top_models(formatter='dict'):
        return model_dict

def main():
    fin = {}
    alg = sys.argv[1]
    dname = sys.argv[2]
    port = int(sys.argv[3])
    ans = atest(alg, dname, port)
    fin[alg + dname] = ans
    fo = open("pickle/{}{}.txt".format(alg, dname), "w")
    for key in fin:
        fo.write(key + "\n") 
        fo.write(str(fin[key]) + "\n")
    fo.close()

main()
