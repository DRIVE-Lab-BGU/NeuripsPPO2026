from stable_baselines3 import *
import pyRDDLGym
from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
from aux import aux
from aux.seeder import FibonacciSeeder
import torch
import numpy as np
import os
import csv
import time


class ExperimentManager():
    EVAL_INTERVAL = 10

    def __init__(self, name='new', domain='reservoir', instance='instance1.rddl', run_from_scratch=True, average=10,
                 net_arch=None, train_episodes_interval=1, horizon=None, exp_num=1, train_seed=112, eval_seed=42,
                 save_model=False):
        self.experiment_name = name
        self.domain=domain
        self.instance = instance
        self.exp_num = exp_num
        self.save_model = save_model
        self.run_from_scratch = run_from_scratch
        self.average = average
        self.net_arch = [12, 12] if net_arch is None else net_arch
        self.train_episodes_interval = train_episodes_interval
        self.trained_episodes = 0
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        instance_number = self.extract_instance_number(instance)
        self.log_dir = os.path.join(self.script_dir, 'logs', self.domain, instance_number)
        self.model_dir = os.path.join(self.script_dir, 'models', self.domain, instance_number)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_model:
            os.makedirs(self.model_dir, exist_ok=True)
        self.log_name = os.path.join(self.log_dir, self.experiment_name+'.csv')
        self.model_name = os.path.join(self.model_dir, self.experiment_name+'.zip')
        self.horizon = horizon
        self.trainSeeder = FibonacciSeeder(start_seed=train_seed)
        self.evalSeeder = FibonacciSeeder(start_seed=eval_seed)

    @staticmethod
    def extract_instance_number(instance):
        name = os.path.splitext(os.path.basename(str(instance)))[0]
        prefix = 'instance_'
        if name.startswith(prefix):
            return name[len(prefix):]
        return name


    def run_experiment(self, num_train_episodes):
        experiment_returns = []
        for policy_index in range(self.exp_num):
            returns = self.run_single_experiement(
                num_train_episodes=num_train_episodes,
                policy_index=policy_index
            )
            experiment_returns.append(returns)
        self.log_results(experiment_returns)

    def run_single_experiement(self, num_train_episodes, policy_index=0):
        train_episodes = num_train_episodes
        train_seed = next(self.trainSeeder)

        # abs_domain_file_location = os.path.join(self.script_dir, 'reservoir', 'domain.rddl')
        # abs_instance_file_location = os.path.join(self.script_dir, 'reservoir', self.instance)
        abs_domain_file_location = os.path.join(self.script_dir, 'problems', self.domain, 'domain.rddl')
        abs_instance_file_location = os.path.join(self.script_dir, 'problems', self.domain, self.instance)

        myTrainEnv = pyRDDLGym.make(domain=abs_domain_file_location, instance=abs_instance_file_location,
                                    base_class=SimplifiedActionRDDLEnv)
        myEvalEnv = pyRDDLGym.make(domain=abs_domain_file_location, instance=abs_instance_file_location,
                                   base_class=SimplifiedActionRDDLEnv)
        horizon = self.horizon
        if horizon is None:
            horizon = myTrainEnv.horizon

        myTrainEnv.reset(seed=train_seed)

        policy_kwargs = {
            "net_arch": self.net_arch,
            "activation_fn": torch.nn.Tanh  # Use ReLU activation
        }

        model_name = self.get_model_name(policy_index)
        try:
            if not self.run_from_scratch and os.path.exists(model_name):
                model = PPO.load(model_name, env=myTrainEnv)
            else:
                model = PPO('MultiInputPolicy', myTrainEnv, policy_kwargs=policy_kwargs,
                            n_steps=self.train_episodes_interval * horizon,
                            batch_size=30, n_epochs=10, verbose=0, seed=train_seed)

            returns = []
            for episode in range(train_episodes):
                model.learn(total_timesteps=self.train_episodes_interval * horizon, reset_num_timesteps=False)
                self.trained_episodes += 1
                iteration = episode + 1
                if iteration % self.EVAL_INTERVAL == 0:
                    agent = StableBaselinesAgent(model, deterministic=True)
                    eval_seeds = [next(self.evalSeeder) for _ in range(self.average)]
                    try:
                        r = self.evaluate(agent, myEvalEnv, seeds=eval_seeds, verbose=0)
                    finally:
                        self.evalSeeder.reset()
                    print('policy: ', policy_index, ' iteration: ', iteration, ' ended with: ', r)
                    returns.append({
                        'iteration': iteration,
                        'mean': r['mean']
                    })
            if self.save_model:
                os.makedirs(self.model_dir, exist_ok=True)
                model.save(model_name)
            return returns
        finally:
            myTrainEnv.close()
            myEvalEnv.close()

    def get_model_name(self, policy_index):
        if self.exp_num == 1:
            return self.model_name
        return os.path.join(
            self.model_dir,
            f'{self.experiment_name}_policy_{policy_index}.zip'
        )

    def evaluate(self, agent, Env, episodes=None, verbose=False, seed=42, seeds=None):
        if seeds is None:
            if episodes is None:
                episodes = 1
            seeds = [seed + eps for eps in range(episodes)]
        returns = []
        for eval_seed in seeds:
            total_reward = 0
            state, _ = Env.reset(seed=eval_seed)
            for step in range(Env.horizon):
                action = agent.sample_action(state)
                next_state, reward, terminated, truncated, _ = Env.step(action)
                done = terminated or truncated
                total_reward += reward
                if verbose:
                    print()
                    print(f'step       = {step}')
                    print(f'state      = {state}')
                    print(f'action     = {action}')
                    print(f'next state = {next_state}')
                    print(f'reward     = {reward}')
                state = next_state
                if done:
                    break
            if verbose:
                print(f'episode ended with reward {total_reward}')
            returns.append(total_reward)
        return {'mean': np.mean(returns), 'std': np.std(returns)}

    def measure_results(self, is_train=True, num_episodes=1, ppo=True):
        episodes = num_episodes

        abs_domain_file_location = os.path.join(self.script_dir, 'reservoir', 'domain.rddl')
        abs_instance_file_location = os.path.join(self.script_dir, 'reservoir', self.instance)

        myEnv = pyRDDLGym.make(domain=abs_domain_file_location, instance=abs_instance_file_location,
                                    base_class=SimplifiedActionRDDLEnv)

        horizon = myEnv.horizon

        policy_kwargs = {
            "net_arch": self.net_arch,
            "activation_fn": torch.nn.Tanh  # Use ReLU activation
        }
        if ppo:
            model = PPO('MultiInputPolicy', myEnv, policy_kwargs=policy_kwargs,
                        n_steps=self.train_episodes_interval * horizon,
                        batch_size=self.train_episodes_interval * horizon, n_epochs=1, verbose=0)
        else:
            model = PPO('MultiInputPolicy', myEnv, policy_kwargs=policy_kwargs,
                        n_steps=self.train_episodes_interval * horizon,
                        batch_size=64, n_epochs=20, verbose=0)
        agent = StableBaselinesAgent(model, deterministic=True)

        start = time.time()
        for episode in range(episodes):
            if is_train:
                model.learn(total_timesteps=self.train_episodes_interval * horizon, reset_num_timesteps=False)
            else:
                r = self.evaluate(agent, myEnv, episodes=1, seed=0, verbose=0)
        end = time.time()
        actual_time = (end-start)/episodes
        print(actual_time)
        myEnv.close()



    def log_results(self, experiment_returns):
        csv_file = self.log_name
        with open(csv_file, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["iteration", "mean_eval_reward", "std_eval_reward"])
            writer.writeheader()

            for row in self.aggregate_results(experiment_returns):
                writer.writerow(row)

    def aggregate_results(self, experiment_returns):
        by_iteration = {}
        for policy_returns in experiment_returns:
            for result in policy_returns:
                by_iteration.setdefault(result['iteration'], []).append(result['mean'])

        rows = []
        for iteration in sorted(by_iteration):
            policy_means = np.asarray(by_iteration[iteration], dtype=np.float64)
            rows.append({
                'iteration': iteration,
                'mean_eval_reward': np.mean(policy_means),
                'std_eval_reward': np.std(policy_means)
            })
        return rows
