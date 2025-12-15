from math import log
import numpy as np
try:
    from . import task_queue
    from .visualize import QueueVisualizer
except ImportError:
    import task_queue
    from visualize import QueueVisualizer

class Task:
    def __init__(self, u_name, tsId):
        self._mRIndex = u_name
        self._timeIndex = tsId  # 任务到达时间
        self._state = 0  # 0 ：初始/新任务, 1 ：已提交到边缘，执行中, 2 ：完成本地计算, 3 ：任务失败
        self.input_data = int(np.random.randint(400, 1000))  # 单位：KB
        self.exe_data = float(np.random.uniform(1.6e9, 4e9))  # 单位：cycles
        self.delay_tol = float(np.random.uniform(0.1, 5.0))  # 单位：s
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
        self.trans_pow = 0.2  # w
        self.snr = None
        self.anchor_ser = None
        self.energy_cur = None  # current energy
        self.time_cur = None  # 当前时隙任务的时延
        self.num_conn = None
        self.kl = 1.00e-27  # 本地能耗系数
        self.distance_to_server = None  # 距离每个server的距离
        self.offload_pass_t = None
        self.offload_prev_t = 0
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
        self.freq = 20.0e9  # frequency 服务器的最大计算资源
        self.bandwidth = 20e6  # bandwidth
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

    def submit_offload_task(self, agent: 'MecAgent'):
        task_id = f"{agent.id}-{agent.task._timeIndex}"
        comp_req = float(agent.state.task_e_s)
        slot_time = self.priority_server.slot_time
        delay_tol = float(agent.state.task_delay_tol)
        max_slots = int(np.ceil(delay_tol / slot_time)) if slot_time > 0 else int(1)
        t = task_queue.ComputeTask(
            task_id=task_id,
            agent_id=str(agent.id),
            computation_requirement=comp_req,
            max_tolerance_delay=max_slots,
            creation_slot=self.priority_server.current_slot,
            current_comp_resource=0.0
        )
        self.priority_server.add_task(t)


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
        self.server_list = []
        self.avail_action = []
        self.task = Task(self.name, 0)
        self.pending_offload_task_id = None
        self.pending_server_id = None
        self.cur_server_id = None

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
        self.slot_time = 0.05  # 模拟的时间步长
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

        self.noise_in_DBm = -114  # background noise power constant, -114 in dBm
        self.b_noise = 10 ** (self.noise_in_DBm / 10)  # 5.011872336272715e-12 in mW
        self.b_noise_InW = self.b_noise / 1000

        self._MAX_LOCAL_FREQUENCY = 1.5e9  # 本地cpu最大频率, 1.5GHz

        self.local_count = 0  # 本地计数
        self.edge_count = 0  # 边缘计数
        self.ly_coe = 100  # lyapunov coefficient
        self.context_delay_number = 5e-11  # 计算上下文传输延迟的常数

        self.debug = True  # 调试信息
        self.visualizer = QueueVisualizer()
        self.sat_weight = 0.5
        self.server_util_a = 1.0
        self.server_util_b = 1.0
        self.server_util_c = 0.1
        self.server_util_d = 0.1
        self._last_server_status = {}
        self._cache_og_total = None
        self._cache_server_utils = None
        self._cache_agent_utils = None
        self.fail_penalty = 1.0

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
        self._ended_agents_ids_step = []  # 记录在当前时间步里“任务已结束”的智能体 ID（结束包括边缘端执行完成或失败）
        for agent in self.agents:
            self.update_agent_position_state(agent)
            self.update_conn_state()
            self.update_agent_channel_state(agent)
            self.update_agent_task_state(agent)
            self.update_agent_action_state(agent)
        for s in self.servers:
            s.priority_server.process_time_slot()  # 此时没任务
        for agent in self.agents:
            if agent.pending_offload_task_id and agent.pending_server_id:
                s = self.servers[agent.pending_server_id - 1]
                done = False
                # 看这个任务是否计算完毕了
                for t in s.priority_server.completed_tasks:
                    if t.task_id == agent.pending_offload_task_id:
                        agent.task._state = 2
                        trans_t = float(agent.state.time_cur) if agent.state.time_cur is not None else 0.0
                        agent.state.time_cur = trans_t + t.time_total
                        # 计算上下文转移延迟
                        prev_sid = int(agent.state.offload_prev_t) if agent.state.offload_prev_t else 0
                        cur_sid = int(np.argmax(agent.action.offload) + 1)
                        if prev_sid != 0 and cur_sid != prev_sid:
                            agent.state.time_cur = float(agent.state.time_cur) + (float(agent.task.exe_data) * float(self.context_delay_number))

                        agent.state.energy_cur += t.energy_total
                        agent.state.epi_energy = agent.state.energy_cur
                        self._ended_agents_ids_step.append(agent.id)
                        done = True
                        break

                if not done:
                    for t in s.priority_server.failed_tasks:
                        if t.task_id == agent.pending_offload_task_id:
                            agent.task._state = 3
                            max_slots = int(t.max_tolerance_delay)
                            agent.state.time_cur = max_slots * float(s.priority_server.slot_time)
                            self._ended_agents_ids_step.append(agent.id)
                            done = True
                            break
                if done:
                    agent.pending_offload_task_id = None
                    agent.pending_server_id = None
        
    # update position state for a particular agent
    def update_agent_position_state(self, agent: MecAgent):
        """
        update position using Gauss-Markov Mobility Model
        """
        s = agent.state
        s.p_pos[0] += s.p_vel[0] * self.slot_time
        s.p_pos[1] += s.p_vel[1] * self.slot_time
        if (s.p_pos[0] > 1000 or s.p_pos[0] < 0) or (s.p_pos[1] > 1000 or s.p_pos[1] < 0):
            s.finish = True

    def update_agent_action_state(self, agent: MecAgent):
        """
        更新每个agent做出的动作
        """
        if sum(agent.action.offload) > 0:
            agent.cur_server_id = np.argmax(agent.action.offload) + 1
            # for i in range(len(agent.action.offload)):
            #     if agent.action.offload[i] == 1:
            #         agent.pending_offload_task_id = f"{agent.id}-{agent.task._timeIndex}"
            #         agent.pending_server_id = i + 1

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
            if d <= 0.0:
                d = 1e-3
            d = max(d, 0.1)
            distance[s.id] = d
            # 信道状态
            path_loss = float(128 + 37.6 * log(max(d / 1000.0, 1e-6), 10))
            power_gain_coe = (10 ** (path_loss / 10)) ** -1
            rayleigh_coe = np.random.rayleigh(scale=1, size=1)[0]
            power_gain = power_gain_coe * rayleigh_coe
            if not np.isfinite(power_gain):
                power_gain = 1e-6
            power_gain = float(np.clip(power_gain, 1e-12, 1e6))
            channel_state[s.id] = power_gain
        agent.state.power_gain = channel_state
        agent.state.distance_to_server = distance

    def update_agent_task_state(self, agent: MecAgent):
        if agent.task is None or agent.task._state in (2, 3):
            agent.task = Task(agent.name, self.time)
        t = agent.task
        agent.state.task_i_s = t.input_data
        agent.state.task_e_s = t.exe_data
        agent.state.task_delay_tol = float(t.delay_tol)
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
        agent.task._state = 2

        if self.debug:
            print(f"卸载到本地计算：energy_cur为{agent.state.energy_cur},epi_energy为{agent.state.epi_energy},epi_latency为{agent.state.time_cur}")

        return t_cost, e_cost

    def edge_cost(self, agent: MecAgent):

        task = agent.task
        offload_s_id = agent.cur_server_id  # 基站id：1234, 卸载下标为0123

        agent.state.offload_prev_t = agent.state.offload_pass_t if agent.state.offload_pass_t is not None else 0
        agent.state.offload_pass_t = offload_s_id
        server: MecServer = self.servers[offload_s_id - 1]

        ratio = self.bandwidth

        def compute_trans_rate():
            trans_rate = ratio * log((1 + (agent.action.v_resource_alloc * agent.state.trans_pow * agent.state.power_gain[offload_s_id] / self.b_noise_InW)), 2)
            return trans_rate

        agent.state.trans_rate = compute_trans_rate()
        trans_t = task.input_data * 8000 / agent.state.trans_rate if agent.state.trans_rate and agent.state.trans_rate > 0 else float('inf')
        trans_e = agent.action.v_resource_alloc * agent.state.trans_pow * trans_t
        cur_task_id = f"{agent.id}-{agent.task._timeIndex}"
        if agent.pending_offload_task_id != cur_task_id:
            server.submit_offload_task(agent)
            agent.pending_offload_task_id = cur_task_id
            agent.pending_server_id = offload_s_id
            agent.task._state = 1
            t_cost = trans_t
            e_cost = trans_e
            agent.state.time_cur = t_cost
            agent.state.energy_cur = e_cost
            return

    def update_conn_state(self):
        for a in self.agents:
            s: MecServer
            for s in self.servers:
                dis = pow((a.state.p_pos[0] - s.state.p_pos[0]) ** 2 + (a.state.p_pos[1] - s.state.p_pos[1]) ** 2, 1 / 2)
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

    def _get_allocated_resource(self, server: MecServer, task_id: str):
        for p in [task_queue.TaskPriority.HIGH, task_queue.TaskPriority.MEDIUM, task_queue.TaskPriority.LOW]:
            for e in server.priority_server.priority_queues[p]:
                if e['task'].task_id == task_id:
                    return float(e['task'].current_comp_resource)
        for t in server.priority_server.completed_tasks:
            if t.task_id == task_id:
                return float(np.mean(t.alloc_traj)) if t.alloc_traj else 0.0
        for t in server.priority_server.failed_tasks:
            if t.task_id == task_id:
                return float(np.mean(t.alloc_traj)) if t.alloc_traj else 0.0
        return 0.0

    def agent_utility(self, agent: MecAgent):
        if agent.task is None or agent.task._state not in (2, 3):
            return 0.0
        tau = float(agent.state.task_delay_tol) if agent.state.task_delay_tol is not None else 0.0
        t_cur = float(agent.state.time_cur) if agent.state.time_cur is not None else 0.0
        if tau > 0:
            s_norm = max(0.0, min(1.0, 1.0 - (t_cur / tau)))
        else:
            s_norm = 0.0
        if sum(agent.action.offload) > 0:
            offload_s_id = int(np.argmax(agent.action.offload) + 1)
            cur_task_id = f"{agent.id}-{agent.task._timeIndex}"
            server = self.servers[offload_s_id - 1]
            c_norm = self._get_allocated_resource(server, cur_task_id)
        else:
            f_ue = float(agent.freq)
            v_alloc = float(agent.action.v_resource_alloc)
            c_norm = (v_alloc * f_ue) / float(self.max_local_frequency) if self.max_local_frequency > 0 else 0.0
            c_norm = max(0.0, min(1.0, c_norm))
        theta = float(self.sat_weight)
        return theta * s_norm - (1.0 - theta) * c_norm

    def _server_near_deadline_total(self, server: MecServer, status: dict):
        nd = status.get('near_deadline', {})
        return float(nd.get('high', 0) + nd.get('medium', 0) + nd.get('low', 0))

    def server_utility(self, server: MecServer):
        cur = server.priority_server.get_queue_status()
        sid = server.id
        last = self._last_server_status.get(sid, None)
        dC = float(cur.get('completed_tasks', 0)) - (float(last.get('completed_tasks', 0)) if last else 0.0)
        dF = float(cur.get('failed_tasks', 0)) - (float(last.get('failed_tasks', 0)) if last else 0.0)
        Q = float(cur.get('high', 0) + cur.get('medium', 0) + cur.get('low', 0))
        N = self._server_near_deadline_total(server, cur)
        a = float(self.server_util_a)
        b = float(self.server_util_b)
        c = float(self.server_util_c)
        d = float(self.server_util_d)
        util = a * dC - b * dF - c * Q - d * N
        self._last_server_status[sid] = cur
        return util

    def total_objective(self):
        og = 0.0
        server_utils = {s.id: self.server_utility(s) for s in self.servers}
        for agent in self.agents:
            if not hasattr(self, '_ended_agents_ids_step') or agent.id not in getattr(self, '_ended_agents_ids_step'):
                continue
            if sum(agent.action.offload) > 0:
                offload_s_id = int(np.argmax(agent.action.offload) + 1)
                p_weight = float(agent.state.trans_pow)
                u_agent = self.agent_utility(agent)
                u_server = float(server_utils.get(offload_s_id, 0.0))
                og += 1.0 * p_weight * (u_agent + u_server)
            else:
                p_weight = 1.0
                u_agent = self.agent_utility(agent)
                # 本地不计服务器效用
                og += 1.0 * p_weight * (u_agent + 0.0)
        return og, server_utils

    def compute_utilities_and_update_visualizer(self):
        if self._cache_agent_utils is None or self._cache_server_utils is None or self._cache_og_total is None:
            self._cache_agent_utils = [self.agent_utility(a) for a in self.agents]
            self._cache_og_total, self._cache_server_utils = self.total_objective()
        metrics = {
            'agent_utility_mean': float(np.mean(self._cache_agent_utils)) if len(self._cache_agent_utils) > 0 else 0.0,
            'agent_utility_values': self._cache_agent_utils,
            'server_utility': self._cache_server_utils,
            'og_total': float(self._cache_og_total)
        }
        if hasattr(self, 'visualizer') and self.visualizer:
            self.visualizer.update(self.servers, self.time, self.agents, metrics)

    def compute_utilities_cache(self):
        ended_agents = [a for a in self.agents if hasattr(self, '_ended_agents_ids_step') and a.id in getattr(self, '_ended_agents_ids_step')]
        self._cache_agent_utils = [self.agent_utility(a) for a in ended_agents]
        self._cache_og_total, self._cache_server_utils = self.total_objective()
