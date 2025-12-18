import numpy as np
from onpolicy.envs.my.core import MecWorld, MecAgent, MecServer
from onpolicy.envs.my.visualize import QueueVisualizer
from onpolicy.envs.my.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self, args):
        # make mec world
        world = MecWorld()
        world.avg_energy = 60.0  # J
        world.max_time = args.episode_length
        world.max_energy = 50  # in J
        world.avg_energy = world.max_energy / world.max_time
        world.trade_lambda = 8
        world.ly_coe = 0.001
        world.noise_in_DBm = -113
        world.bandwidth = 20e6  # 1MHz
        world.p_max = 1  # 1mW

        # add agents
        num_users = args.num_agents
        # agent number
        world.num_users = num_users
        # add edge users
        world.agents = [MecAgent() for i in range(num_users)]  # 初始化环境中所有的agent
        world.dim_offload = len(world.agents)

        for i, agent in enumerate(world.agents):
            agent.name = 'UE %d' % i
            agent.type = 'user'
            agent.id = i
            agent.size = 0.15

        # add mec server
        num_servers = 4
        world.num_servers = num_servers
        world.servers = [MecServer() for i in range(num_servers)]
        mec_server: MecServer
        for i, mec_server in enumerate(world.servers):
            mec_server.name = 'UAV %d' % (i + 1)
            mec_server.type = 'server'
            mec_server.id = i + 1
            mec_server.priority_server.server_id = mec_server.id
            mec_server.size = 0.25
        # make initial conditions
        self.reset_world(world)
        world.visualizer = QueueVisualizer(use_wandb=args.use_wandb)
        if hasattr(world.servers[0], 'priority_server'):
            world.slot_time = float(world.servers[0].priority_server.slot_time)
        assert len(world.servers_list) == num_servers
        return world

    def reset_world(self, world: MecWorld):
        """
        重设基站、智能体位置、通信状态等
        """
        world.time = 0
        for agent in world.agents:
            agent.utility_history = []
            agent.last_utility = None
        # set servers position states
        world.servers_list[0].state.p_pos = [250, 250, 15]
        world.servers_list[1].state.p_pos = [750, 250, 15]
        world.servers_list[2].state.p_pos = [250, 750, 15]
        world.servers_list[3].state.p_pos = [750, 750, 15]

        # random properties for agents
        for i in range(0, world.num_users):
            world.agents[i].color = np.array([0.85, 0.35, 0.35])

        # set random initial states
        agent: MecAgent
        for agent in world.agents:
            # - `(-1, +1, world.dim_p)`：表示生成的随机数范围在-1到+1之间，数组的长度为`world.dim_p`。
            # initial users' position state
            x_choice = np.random.randint(low=0, high=10)
            y_choice = np.random.randint(low=0, high=10)
            if x_choice % 2:
                if y_choice % 2:
                    # x奇数，y奇数，从道路右侧出发
                    agent.state.p_pos[0] = 1000
                    agent.state.p_pos[1] = np.random.uniform(500, 750)
                    agent.state.p_pos[2] = 0
                    agent.state.p_vel = [-10, 0, 0]
                else:
                    # x奇数，y偶数，从道路左侧出发
                    agent.state.p_pos[0] = 0
                    agent.state.p_pos[1] = np.random.uniform(250, 500)
                    agent.state.p_pos[2] = 0
                    agent.state.p_vel = [10, 0, 0]
            else:
                if y_choice % 2:
                    # x偶数，y奇数，从道路上方出发
                    agent.state.p_pos[0] = np.random.uniform(250, 500)
                    agent.state.p_pos[1] = 1000
                    agent.state.p_pos[2] = 0
                    agent.state.p_vel = [0, -10, 0]
                else:
                    # x偶数，y偶数，从道路下方出发
                    agent.state.p_pos[0] = np.random.uniform(500, 750)
                    agent.state.p_pos[1] = 0
                    agent.state.p_pos[2] = 0
                    agent.state.p_vel = [0, 10, 0]

            # initial channel state
            world.update_agent_channel_state(agent)
            # initial task state
            world.update_agent_task_state(agent)
            world.update_conn_state()

    @staticmethod
    def reward(agent: MecAgent, world: MecWorld):
        # n = len(world.agents_list) if hasattr(world, 'agents_list') else 1
        og = float(world._cache_og_total) if hasattr(world, '_cache_og_total') and world._cache_og_total is not None else 0.0
        ended_ids = getattr(world, '_ended_agents_ids_step', []) if hasattr(world, '_ended_agents_ids_step') else []
        fail_pen = float(getattr(world, 'fail_penalty', 1.0)) if agent.id in ended_ids and getattr(agent.task, '_state', None) == 3 else 0.0
        return og - fail_pen

    @staticmethod
    def observation(agent: MecAgent, world: MecWorld):
        i_mb = float(agent.state.task_i_s if agent.state.task_i_s is not None else 0.0) / 1e6
        i_mb = float(np.clip(i_mb, 0.0, 200.0))
        e_gcy = float(agent.state.task_e_s if agent.state.task_e_s is not None else 0.0) / 1e9
        e_gcy = float(np.clip(e_gcy, 0.0, 100.0))
        tau_s = float(agent.state.task_delay_tol if agent.state.task_delay_tol is not None else 0.0)
        tau_norm = float(np.clip(tau_s / 2.0, 0.0, 1.0))
        q_left = float(agent.state.task_q_left if agent.state.task_q_left is not None else 0.0)
        q_left = float(np.clip(q_left, 0.0, 1e6))
        pg_list = []
        if agent.state.power_gain is not None:
            for pg in agent.state.power_gain.values():
                pg_db = 10.0 * np.log10(pg + 1e-12)
                pg_norm = (pg_db + 120.0) / 100.0
                pg_list.append(float(np.clip(pg_norm, 0.0, 1.0)))
        else:
            pg_list = [0.0] * len(world.servers_list)
        obs = np.array([i_mb, e_gcy, tau_norm, q_left] + pg_list, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return obs

    @staticmethod
    def done(world: MecWorld):
        if world.time >= world.max_time:
            return 1
        else:
            return 0

    @staticmethod
    def info(agent, world):
        info = [agent.state.epi_energy, agent.state.time_cur]

        return info
