import torch 
from pathlib import Path
import argparse
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.train.reporter import Reporter
if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument( "--modelpaths",type=str,nargs='+')
    parser.add_argument( "--criterions",nargs='+',type=str,default=["valid acc max"],)
    parser.add_argument( "--nbest",type=int,nargs='+',default=[5,])
    args = parser.parse_args()
    criterions = [tuple(crit.split(" ")) for crit in args.criterions]
    for model_dir in args.modelpaths:
        print(model_dir)
        model_dir = Path(model_dir) 
        states = torch.load(model_dir / "checkpoint.pth")
        reporter = Reporter()
        reporter.load_state_dict(states["reporter"])
        average_nbest_models(model_dir,reporter,criterions,args.nbest)
    