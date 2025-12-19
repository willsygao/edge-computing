import random
from typing import Union, List

import gym
import numpy as np
from gym import spaces
from .core import MecWorld, MecAgent


class MultiAgentMecEnv(gym.Env):
    agents: list[MecAgent]
    world: MecWorld
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world: MecWorld, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, discrete_action=True, benchmark=False):

        self.world = world
        self.agents = self.world.agents_list
        self.servers = self.world.servers_list
        self.agent_num = len(self.agents)
        self.n = self.agent_num
        self.server_num = len(self.servers)
        # scenario callbacks
        self.reset_callback = reset_callback  # 传入了一个方法，reset调用函数
        self.reward_callback = reward_callback  # 传入了一个方法，reward调用函数
        self.observation_callback = observation_callback  # 传入了一个获取状态的方法，observation调用函数
        self.info_callback = info_callback  # None
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True

        self.benchmark = benchmark

        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False

        # configure spaces
        self.action_space = []
        self.observation_space = []  # [40, 40,..., 40]
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            offload_action_space = spaces.Discrete(self.server_num + 1)
            total_action_space.append(offload_action_space)
            # power_action_space = spaces.Box(low=0, high=1, shape=(world.dim_power,))
            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    # 否则，创建一个`spaces.Tuple`对象，其中包含所有的动作空间。
                    act_space = tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = len(observation_callback(agent, self.world))  # 15
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)))
            agent.action.offload = np.zeros(self.world.dim_offload)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            random.seed(seed)

    def step(self, action_n):
        reward_n = []
        obs_n = []
        done_n = []
        info_n = []

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.world)

        num_off_in_servers = np.sum(action_n, axis=0)[1:self.world.num_servers + 1]
        for i, server in enumerate(self.servers):
            server.state.num_offload = num_off_in_servers[i]

        self.world.step()

        for i, agent in enumerate(self.agents):
            # if agent.task._state == 0 and sum(agent.action.offload) > 0:
            if agent.task._state == 0 and sum(action_n[i][1:self.world.num_servers + 1]) > 0:
                assert sum(action_n[i][1:self.world.num_servers + 1]) == 1
                offload_s_id = int(np.argmax(agent.action.offload) + 1)
                self.world.edge_cost(agent)
                action_type = 1
                trans_rate = float(agent.state.trans_rate) if agent.state.trans_rate is not None else 0.0
                p_weight = float(agent.state.trans_pow)
            # elif agent.task._state == 0 and sum(agent.action.offload) == 0:
            elif agent.task._state == 0 and sum(action_n[i][1:self.world.num_servers + 1]) == 0:
                offload_s_id = 0
                self.world.local_cost(agent)
                # if not hasattr(self.world, '_ended_agents_ids_step'):
                #     self.world._ended_agents_ids_step = []
                # if agent.id not in self.world._ended_agents_ids_step:
                #     self.world._ended_agents_ids_step.append(agent.id)
                action_type = 0
                trans_rate = 0.0
                p_weight = 1.0
            else:
                if agent.task.offloading_target == 'edge':
                    action_type = 1
                    offload_s_id = int(np.argmax(agent.action.offload) + 1)
                    trans_rate = float(agent.state.trans_rate) if agent.state.trans_rate is not None else 0.0
                    p_weight = float(agent.state.trans_pow)
                else:
                    action_type = 0
                    trans_rate = 0.0
                    p_weight = 1.0
                    offload_s_id = 0
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            trade_lambda = float(self.world.trade_lambda) if hasattr(self.world, 'trade_lambda') and self.world.trade_lambda is not None else 0.0
            objective_cur = float(p_weight) * (float(agent.state.energy_cur) + trade_lambda * float(agent.state.time_cur))
            env_info = [float(agent.state.epi_energy), float(agent.state.time_cur), float(objective_cur), float(action_type), float(offload_s_id), float(trans_rate)]
            info_n.append(env_info)

        if hasattr(self.world, 'compute_utilities_cache'):
            self.world.compute_utilities_cache()

        for agent in self.agents:
            reward_n.append([self._get_reward(agent)])

        # 计算效用并更新可视化（使用最新的能耗/时延与队列状态）
        if hasattr(self.world, 'compute_utilities_and_update_visualizer'):
            self.world.compute_utilities_and_update_visualizer()

        reward = np.sum(reward_n)

        reward_n = [[reward]] * self.n
        share_obs = []

        avail_actions = []
        for agent in self.agents:
            avail_actions.append(agent.avail_action)

        return obs_n, share_obs, reward_n, done_n, info_n, avail_actions

    # obs, share_obs, rews, dones, infos, available_actions
    @staticmethod
    def _set_action(action: list, agent, world: MecWorld):
        agent.action.local = action[0]
        agent.action.offload = action[1:world.num_servers + 1]

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        # 如果`self.info_callback`不是`None`，则调用`self.info_callback`方法，该方法接受两个参数：`agent`和`self.world`。
        # 该方法应返回一个包含有关代理和世界的信息的字典。
        return self.info_callback(agent, self.world)

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        share_obs = []
        self.agents = self.world.agents_list
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        avail_actions = []
        for agent in self.agents:
            avail_actions.append(agent.avail_action)
        return obs_n, share_obs, avail_actions
    # get observation for a particular agent
    # 获取指定智能体对环境的观测
    def _get_obs(self, agent):
        # 用于获取特定智能体的观测值
        if self.observation_callback is None:
            return np.zeros(0)
        obs = self.observation_callback(agent, self.world)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return obs

    # get dones for a particular agent
    def _get_done(self, agent):
        if self.done_callback is None:
            if agent.state.finish == True:
                return True
            else:
                return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    # 获取指定智能体对环境的奖励
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)
