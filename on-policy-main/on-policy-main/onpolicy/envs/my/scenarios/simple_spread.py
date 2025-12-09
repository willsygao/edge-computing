import numpy as np
from onpolicy.envs.my.core import MecWorld, MecAgent, MecServer
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
        assert len(world.servers_list) == num_servers
        return world

    def reset_world(self, world: MecWorld):
        """
        重设基站、智能体位置、通信状态等
        """
        world.time = 0
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
        r1 = agent.state.energy_cur
        return -r1
        # r2 = agent.mu2 * (agent.state.task_delay_tol - agent.state.time_cur)
        # if (agent.state.task_delay_tol - agent.state.time_cur) >= 0:
        #     return r2 - r1
        # else:
        #     return -agent.mu1

    @staticmethod
    def observation(agent: MecAgent, world: MecWorld):
        # get positions of all entities in this agent's reference frame
        agents_task_input = []
        agents_task_execution = []
        agents_task_tau = []
        agents_task_q_left = []
        agents_power_gain = []

        agents_task_input.append(agent.state.task_i_s)
        agents_task_execution.append(agent.state.task_e_s)
        agents_task_tau.append(agent.state.task_delay_tol)
        agents_task_q_left.append(agent.state.task_q_left)
        agents_power_gain = agents_power_gain + list(agent.state.power_gain.values())
        # for other in world.agents:
        #     if other is agent: continue
        #     agents_task_input.append(other.state.task_i_s)
        #     agents_task_execution.append(other.state.task_e_s)
        #     agents_task_tau.append(other.state.task_delay_tol)
        #     agents_task_q_left.append(other.state.task_q_left)
        #     agents_power_gain = agents_power_gain + list(other.state.power_gain.values())

        return np.concatenate([agents_task_input + agents_task_execution + agents_task_tau + agents_task_q_left
                               + agents_power_gain])

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
