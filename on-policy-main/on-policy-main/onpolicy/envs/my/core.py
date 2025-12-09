from math import log
import numpy as np
import task_queue

class Task:
    def __init__(self, u_name, tsId):
        self._mRIndex = u_name
        self._timeIndex = tsId
        self._state = 0
        self.input_data = np.random.randint(5000, 10000)
        self.exe_data = np.random.uniform(5000, 40000)
        self.delay_tol = 0.008
        self.type = np.random.randint(0, 2)

class EntityState(object):
    p_pos: list
    p_vel: list

    def __init__(self):
        self.p_pos = [0, 0, 0]  # physical position [0]:x [1]:y [2]:z
        self.p_vel = [0, 0, 0]  # physical velocity [0]:x [1]:y [2]:z

class AgentState(EntityState):

    def __init__(self):
        super(AgentState, self).__init__()
        self.task_i_s = None  # input size
        self.task_e_s = None  # execution size
        self.task_delay_tol = None  # task deadline constraint
        self.p_vel_mean = None
        self.p_vel_std = None
        self.power_gain = None  # 对每个BS的信道功率增益
        self.trans_rate = None  # 数据传输速率
        self.trans_pow = 0.25  # w
        self.snr = None
        self.anchor_ser = None
        self.energy_cur = None  # current energy
        self.time_cur = None
        self.num_conn = None
        self.kl = 1.00e-27  # 本地能耗系数
        self.distance_to_server = None  # 距离每个server的距离
        self.offload_pass_t = None
        self.task_q_left = None  # unexecuted task size in queue
        self.vir_q = 0  # virtual queue size

        self.epi_energy = 0
        self.finish = False

class ServerState(EntityState):
    def __init__(self):
        super(ServerState, self).__init__()
        self.num_conn = None
        self.num_offload = None
        self.ke = 1.00e-29  # 服务器能耗系数


class Action(object):
    offload: list

    def __init__(self):
        self.local = None
        self.offload = []  # offload action 0: local, 1-M:edge
        self.v_resource_alloc = 1  # 本地资源分配ui0
        self.b_resource_alloc = 1  # 服务器资源分配uij


class Entity(object):
    def __init__(self):
        self.name = ''  # name
        self.type = ''  # type
        self.id = 0
        self.size = 0.050  # properties in figure
        self.state = EntityState()  # state
        self.color = None  # color
        self.initial_mass = 1.0  # mass

    @property
    def mass(self):
        return self.initial_mass

class MecServer(Entity):
    con_agents: dict
    state: ServerState

    def __init__(self):
        super(MecServer, self).__init__()
        # entity is not movable by default
        self.freq = 10.0e9  # frequency 服务器的最大计算资源
        self.bandwidth = 15e6  # bandwidth
        self.com_range = 400  # communication range
        self.n_propeller = 4  # number of propellers
        self.air_rho = 1.29  # air density
        self.wing_len = 0.20  # wing area in m
        self.movable = False
        self.agent_dict = {}
        self.state = ServerState()
        self.priority_server = task_queue.PriorityQueueServer(self.id, self.freq)

    @property
    def connected_agents(self):
        return self.con_agents


class MecAgent(Entity):
    state: AgentState

    def __init__(self):
        super(MecAgent, self).__init__()
        self.movable = True  # agents are movable by default
        self.u_range = None  # communication range
        self.time = 0  # initial simulation time
        self.energy_avg = .5  # average energy consumption
        self.freq = 2.4e9  # UE frequency
        self.state = AgentState()  # state
        self.action = Action()  # action
        self.action_callback = None
        self.mu1 = 0.5  # reward adjust factor
        self.mu2 = 0.5  # reward adjust factor
        self.server_list = []
        self.avail_action = []
        self.task = Task(self.name, 0)

class MecWorld(object):
    num_users: int
    agents: list[MecAgent]
    servers: list[MecServer]

    def __init__(self):
        self.world_id = None
        self.x_range = 1000  # x range in meters
        self.y_range = 1000  # y range in meters
        self.z_range = 200  # z range in meters

        # list of agents and entities (can change at execution-time!)
        self.agents = []  # 智能体列表
        self.servers = []  # MEC服务器列表
        self.num_users = 0
        self.num_servers = 0

        # offload dimensionality
        self.dim_offload = None
        self.dim_power = 1
        self.bandwidth = None  # bandwidth

        # channel state dimensionality
        self.dim_c = 1  # 信道状态维度
        # position dimensionality
        self.dim_p = 3  # 位置的维度，3维
        # task size dimensionality
        self.dim_task_size = 1  # 任务量维度
        self.dim_task_type = 1  # task type dimensionality
        # simulation timestep
        self.dt = 2  # 模拟的时间步长
        self.time = 0  # simulation time
        self.max_time = 1000  # maximum time for one episode

        # memorability parameter
        self.m = 0.5  # 记忆性参数μ
        self.kappa = 1.00e-28  # 电容常数
        self.local_c = 500  # 本地cpu时钟频率
        self.energy_u_b = 0.001  # in J
        self.p_max = 1  # maximum power, in W
        self.p_min = 0.001  # minimum power, in W
        self.x_local = 5000  # local computing speed
        self.x_edge = 5000  # edge computing speed
        self.max_energy = None
        self.avg_energy = None
        self.trade_lambda = None

        self.noise_in_DBm = -113  # background noise power constant, -113 in dBm
        self.b_noise = 10 ** (self.noise_in_DBm / 10)  # 5.011872336272715e-12 in mW
        self.b_noise_InW = self.b_noise / 1000

        self._MAX_LOCAL_FREQUENCY = 1.5e9  # 本地cpu最大频率, 1.5GHz

        self.local_count = 0  # 本地计数
        self.edge_count = 0  # 边缘计数
        self.ly_coe = 100  # lyapunov coefficient
        self.context_delay_number = 5e-11  # 计算上下文传输延迟的常数

        self.debug = True  # 调试信息

    # return all entities in the world
    # 获取包含所有智能体对象的列表
    @property
    def agents_list(self):
        return self.agents

    @property
    def servers_list(self):
        return self.servers

    @property
    def max_local_frequency(self):
        return self._MAX_LOCAL_FREQUENCY

    # update state of the world
    def step(self):  # 智能体位置、连接状态、更新一个代理（MecAgent）与一组服务器（self.servers_list）之间的信道状态信息、更新一个代理（MecAgent）的任务状态信息、任务队列、虚拟队列
        self.time += 1
        # update agent state
        for agent in self.agents:
            self.update_agent_position_state(agent)
            self.update_conn_state()
            self.update_agent_channel_state(agent)
            self.update_agent_task_state(agent)
        # print("env {} time: {}".format(self.world_id, self.time))

    # update position state for a particular agent
    def update_agent_position_state(self, agent: MecAgent):
        """
        update position using Gauss-Markov Mobility Model
        """
        s = agent.state
        s.p_pos[0] += s.p_vel[0] * self.dt
        s.p_pos[1] += s.p_vel[1] * self.dt
        if (s.p_pos[0] > 1000 or s.p_pos[0] < 0) or (s.p_pos[1] > 1000 or s.p_pos[1] < 0):
            s.finish = True

        # if self.debug:
        #     print(f"智能体{agent.id}的位置是：{s.p_pos[0]},{s.p_pos[1]},{s.p_pos[2]}")


    def update_agent_channel_state(self, agent: MecAgent):
        """
        distance between two points
        agent 对每个 server 的信道功率增益保存在 agent.state.power_gain
        """
        channel_state = {}
        distance = {}
        for s in self.servers_list:
            xd = s.state.p_pos[0] - agent.state.p_pos[0]
            yd = s.state.p_pos[1] - agent.state.p_pos[1]
            zd = s.state.p_pos[2] - agent.state.p_pos[2]
            d = pow(xd ** 2 + yd ** 2 + zd ** 2, 1 / 2)
            distance[s.id] = d
            # 信道状态
            path_loss = float(128 + 37.6 * log(d / 1000, 10))  # 自由空间路径损耗
            power_gain_coe = (10 ** (path_loss / 10)) ** -1  # 路径增益系数
            rayleigh_coe = np.random.rayleigh(scale=1, size=1)[0]  # 瑞利随机过程
            power_gain = power_gain_coe * rayleigh_coe  # 信道功率增益
            channel_state[s.id] = power_gain
        agent.state.power_gain = channel_state
        agent.state.distance_to_server = distance

    def update_agent_task_state(self, agent: MecAgent):
        t = Task(agent.name, self.time)  # 生成任务
        agent.task = t
        agent.state.task_i_s = t.input_data
        agent.state.task_e_s = t.exe_data
        agent.state.task_delay_tol = t.delay_tol
        agent.state.task_type = t.type
        agent.state.task_q_left = 0

    def local_cost(self, agent: MecAgent):
        agent.state.offload_pass_t = 0
        task = agent.task
        t_cost = task.exe_data / (agent.action.v_resource_alloc * agent.freq)
        e_cost = agent.state.kl * task.exe_data * pow((agent.action.v_resource_alloc * agent.freq), 2)
        agent.state.time_cur = t_cost
        agent.state.energy_cur = e_cost
        agent.state.epi_energy += e_cost

        if self.debug:
            print(f"卸载到本地计算：energy_cur为{agent.state.energy_cur},epi_energy为{agent.state.epi_energy},epi_latency为{agent.state.time_cur}")

        return t_cost, e_cost

    def edge_cost(self, agent: MecAgent):

        task = agent.task
        offload_s_id = np.argmax(agent.action.offload) + 1  # 基站id：1234, 卸载下标为0123
        if self.debug:
            print(f"智能体{agent.id}的动作空间为：{agent.action.offload}，卸载的基站id为：{offload_s_id}")
            print(f"智能体{agent.id}可卸载的基站为：{agent.server_list}")

        if offload_s_id not in agent.server_list:
            # if self.debug:
            #     print(f"智能体{agent.id}的卸载目标{offload_s_id}不合法，转为本地卸载")
            agent.mu1 += 0.5
            return self.local_cost(agent)

        context_trans_delay = 0
        if agent.state.offload_pass_t != offload_s_id and agent.state.offload_pass_t != 0:  # 计算上下文转移延迟
            context_trans_delay = task.exe_data * self.context_delay_number
        agent.state.offload_pass_t = offload_s_id
        server: MecServer = self.servers[offload_s_id - 1]
        if server.state.num_conn == 0:
            ratio = self.bandwidth
            exe_t = task.exe_data / (agent.action.b_resource_alloc * server.freq)
        else:
            ratio = self.bandwidth / server.state.num_conn
            exe_t = task.exe_data / (agent.action.b_resource_alloc * server.freq / server.state.num_conn)  # 执行时间

        def compute_trans_rate():
            trans_rate = ratio * log((1 + (agent.action.v_resource_alloc * agent.state.trans_pow *
                                           agent.state.power_gain[offload_s_id] / self.b_noise_InW)), 2)
            return trans_rate

        agent.state.trans_rate = compute_trans_rate()
        trans_t = task.input_data / agent.state.trans_rate  # 传输时间
        trans_e = agent.action.v_resource_alloc * agent.state.trans_pow * trans_t
        # exe_e = server.state.ke * pow((agent.action.b_resource_alloc * server.freq / server.state.num_conn),
        #                               2) * task.exe_data
        t_cost = exe_t + trans_t + context_trans_delay
        e_cost = trans_e
        agent.state.time_cur = t_cost
        agent.state.energy_cur = e_cost
        agent.state.epi_energy += e_cost

        if self.debug:
            print(
                f"卸载到边缘计算：智能体{agent.id}的energy_cur为{agent.state.energy_cur},epi_energy为{agent.state.epi_energy},epi_latency为{agent.state.time_cur}")

        return t_cost, e_cost


    def update_conn_state(self):
        for a in self.agents:
            s: MecServer
            for s in self.servers:
                dis = pow((a.state.p_pos[0] - s.state.p_pos[0]) ** 2 + (a.state.p_pos[1] -
                                                                        s.state.p_pos[1]) ** 2, 1 / 2)
                if dis > s.com_range and s.id in a.server_list:
                    a.server_list.remove(s.id)
                if dis <= s.com_range:
                    if s.id not in a.server_list:
                        a.server_list.append(s.id)
                    s.agent_dict[a.id] = dis
                elif dis > s.com_range and a.id in s.agent_dict.keys():
                    s.agent_dict.pop(a.id)
            a.state.num_conn = len(a.server_list)
        for s in self.servers:
            s.state.num_conn = len(s.agent_dict.keys())
        for a in self.agents:
            a.avail_action = np.zeros(len(self.servers) + 1)
            for i in a.server_list:
                a.avail_action[i] = 1
            a.avail_action[0] = 1