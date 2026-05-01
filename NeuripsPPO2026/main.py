import os
import faulthandler

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(CACHE_DIR, 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', CACHE_DIR)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('KMP_AFFINITY', 'disabled')
faulthandler.enable()

from NeuripsPPO2026 import ExperimentManager
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", type=str, default='reservoir', help="The instance to run")
parser.add_argument("--instance", type=int, default=1, help="The instance to run")
parser.add_argument("--seeds", type=int, default=1, help="how many seeds to train on")
parser.add_argument("--exp_num", type=int, default=None, help="How many policies to train")
parser.add_argument("--seed", type=int, default=1, help="The experiment seed to use")
parser.add_argument("--train_seed", type=int, default=112, help="The first training seed")
parser.add_argument("--eval_seed", type=int, default=42, help="The first evaluation seed")
parser.add_argument("--save_model", action="store_true", help="Save trained model parameters")
parser.add_argument("--length", type=int, default=10, help="How many episode to train on")
parser.add_argument("--average", type=int, default=1, help="How many episode to average test on")
parser.add_argument("--horizon", type=int, default=120, help="How many episode to average test on")

args = parser.parse_args()



def main(domain,instance, seed, length, average, horizon, exp_num, train_seed, eval_seed, save_model):
    instance2run = 'instance_'+str(instance)+'.rddl'
    exp_name = domain + '_' + str(instance) + '_ppo' + str(seed)
    if train_seed is None:
        train_seed = seed
    Manager = ExperimentManager.ExperimentManager(name=exp_name, domain=domain, instance=instance2run, run_from_scratch=False,
                                                  average=average, horizon=horizon, train_episodes_interval=1,
                                                  exp_num=exp_num, train_seed=train_seed, eval_seed=eval_seed,
                                                  save_model=save_model)
    Manager.run_experiment(num_train_episodes=length)


if __name__ == '__main__':
    exp_num = args.seeds if args.exp_num is None else args.exp_num
    main(args.domain, args.instance, args.seed, args.length, args.average, args.horizon,
         exp_num, args.train_seed, args.eval_seed, args.save_model)
