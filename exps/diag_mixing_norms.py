import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["diag_mixing_norms"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.iters = 2
args.lr = 0.01
args.number_workers = 0
args.device = "cpu"
args.dataset = "office"
args.seed = 0
args.num_classes = 10
args.size = 64
args.batch = 32
args.wk_iters = 1
args.adcol_epoch = 1
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.domain_keyed_proto = True
args.no_discriminator_fix = False
set_seed(args)

tl, cd, testl = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(tl)

ours(args, tl, testl, cd, peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.1)
