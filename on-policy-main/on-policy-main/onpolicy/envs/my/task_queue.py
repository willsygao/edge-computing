from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

debug = True

class TaskPriority(Enum):
    """任务优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComputeTask:
    """计算任务数据类"""
    task_id: str
    agent_id: str
    computation_requirement: float  # 总计算需求 (M Cycles)
    max_tolerance_delay: int  # 最大容忍延迟 (时隙数)
    creation_slot: int  # 创建时隙
    current_comp_resource: float  # 当前时隙分配的计算资源量
    status: TaskStatus = TaskStatus.PENDING
    remaining_computation: Optional[float] = None  # 剩余计算量
    completion_slot: Optional[int] = None
    failure_reason: Optional[str] = None
    failure_slot: Optional[int] = None
    alloc_traj: List[float] = None
    energy_total: float = 0.0
    energy_traj: List[float] = None
    time_total: float = 0.0

    def __post_init__(self):
        """初始化后设置剩余计算量"""
        if self.remaining_computation is None:
            self.remaining_computation = self.computation_requirement
        if self.alloc_traj is None:
            self.alloc_traj = []
        if self.energy_traj is None:
            self.energy_traj = []


class TaskPriorityEvaluator:
    """任务优先级评估器"""

    @staticmethod
    def calculate_urgency_score(task: ComputeTask, current_slot: int) -> float:
        """
        计算任务紧急度分数
        基于剩余容忍时隙比例
        """
        time_elapsed = current_slot - task.creation_slot
        time_remaining = task.max_tolerance_delay - time_elapsed

        if time_remaining <= 0:
            # 任务已超时，返回最高紧急度
            return 1.0
        else:
            # 基于剩余时间比例计算紧急度
            urgency_score = 1.0 - (time_remaining / task.max_tolerance_delay)
            return max(0.0, min(1.0, urgency_score))

    @staticmethod
    def calculate_resource_score(task: ComputeTask) -> float:
        """
        计算资源需求分数
        使用剩余计算量作为资源需求指标
        """
        return task.remaining_computation

    @staticmethod
    def determine_priority(task: ComputeTask, current_slot: int) -> TaskPriority:
        """确定任务优先级"""
        urgency_score = TaskPriorityEvaluator.calculate_urgency_score(task, current_slot)

        composite_score = urgency_score

        # 根据综合评分划分优先级
        if composite_score >= 0.7:
            return TaskPriority.HIGH
        elif composite_score >= 0.4:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW


class PriorityQueueServer:
    """多级优先级队列服务器（时间步长版本）"""

    def __init__(self, server_id: int, total_computation_power: float = 10.0e9, energy_coeff: float = 1.0e-29):
        """
        初始化服务器

        Args:
            server_id: 服务器标识
            total_computation_power: 服务器总计算能力 (GHz = e9 CPU周期/s)
        """
        self.server_id = server_id
        self.total_computation_power = total_computation_power  # CPU周期/s
        self.slot_time = 0.05  # 时隙大小 0.05s
        self.energy_coeff = energy_coeff
        self.energy_last_slot = 0.0

        # 多级优先级队列
        self.priority_queues = {
            TaskPriority.HIGH: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.LOW: []
        }

        # 时隙计数器
        self.current_slot = 0

        # 统计信息
        self.completed_tasks: List[ComputeTask] = []
        self.failed_tasks: List[ComputeTask] = []
        self.total_processing_slots: int = 0
        self.queue_weights = {TaskPriority.HIGH: 1.0, TaskPriority.MEDIUM: 0.0, TaskPriority.LOW: 0.0}

    def add_task(self, task: ComputeTask) -> bool:
        """
        添加任务到优先级队列

        Args:
            task: 计算任务

        Returns:
            bool: 任务是否成功加入队列
        """
        # 检查任务是否已经超时
        time_elapsed = self.current_slot - task.creation_slot
        if time_elapsed > task.max_tolerance_delay:
            # 任务已超时，直接标记为失败
            task.status = TaskStatus.FAILED
            task.failure_reason = f"任务添加时已超时，已耗时{time_elapsed}时隙，超过最大容忍延迟{task.max_tolerance_delay}时隙"
            self.failed_tasks.append(task)
            return False

        # 确定任务优先级
        priority = TaskPriorityEvaluator.determine_priority(task, self.current_slot)

        # 创建任务条目
        task_entry = {
            'task': task,
            'agent_id': task.agent_id,
            'arrival_slot': self.current_slot,
            'priority': priority,
            'resource_score': TaskPriorityEvaluator.calculate_resource_score(task)
        }

        # 添加到对应优先级队列
        self.priority_queues[priority].append(task_entry)
        task.status = TaskStatus.PENDING

        return True

    def _get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有需要处理的队列中的任务（按优先级顺序）"""
        all_tasks = []
        for priority in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            all_tasks.extend(self.priority_queues[priority])
        return all_tasks

    def _get_cur_tasks(self) -> List[Dict[str, Any]]:
        """获取当前需要处理的队列中的任务（按优先级顺序）"""
        all_tasks = []

        high_priority_tasks = self.priority_queues[TaskPriority.HIGH]
        if high_priority_tasks:
            all_tasks.extend(high_priority_tasks)
            return all_tasks

        medium_priority_tasks = self.priority_queues[TaskPriority.MEDIUM]
        if medium_priority_tasks:
            all_tasks.extend(medium_priority_tasks)
            return all_tasks

        low_priority_tasks = self.priority_queues[TaskPriority.LOW]
        if low_priority_tasks:
            all_tasks.extend(low_priority_tasks)
            return all_tasks

        return all_tasks

    def _compute_queue_metrics(self) -> Dict[TaskPriority, Dict[str, float]]:
        metrics = {}
        for p in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            q = self.priority_queues[p]
            n = len(q)
            near = 0
            for e in q:
                u = TaskPriorityEvaluator.calculate_urgency_score(e['task'], self.current_slot)
                if u >= 0.8:
                    near += 1
            metrics[p] = {
                'len': float(n),
                'near': float(near)
            }
        return metrics

    def _compute_queue_weights(self) -> Dict[TaskPriority, float]:
        m = self._compute_queue_metrics()
        total_len = sum(m[p]['len'] for p in m)
        total_near = sum(m[p]['near'] for p in m)
        def ratio(x, total):
            return (x / total) if total > 0 else 0.0
        w = {}
        for p in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            r_len = ratio(m[p]['len'], total_len)
            r_near = ratio(m[p]['near'], total_near)
            base = 0.1
            if p == TaskPriority.HIGH:
                score = 0.6 * r_near + 0.4 * r_len
            elif p == TaskPriority.MEDIUM:
                score = 0.3 * r_near + 0.4 * r_len
            else:
                score = 0.1 * r_near + 0.2 * r_len
            w[p] = max(base, score)
        s = sum(w.values())
        for p in w:
            w[p] = w[p] / s if s > 0 else (1.0 if p == TaskPriority.HIGH else 0.0)
        self.queue_weights = w
        return w

    def _allocate_resources(self) -> Dict[str, float]:
        all_tasks = self._get_all_tasks()
        if not all_tasks:
            return {}
        w = self._compute_queue_weights()
        allocated_resources = {}
        for p in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            q = self.priority_queues[p]
            if not q:
                continue
            self._allocate_resources_by_computation(q, w[p])
            for e in q:
                task_id = e['task'].task_id
                allocated_resources[task_id] = e['task'].current_comp_resource
        return allocated_resources

    def _allocate_resources_by_computation(self, tasks: List[Dict[str, Any]], queue_weight: float) -> None:
        total_computation = sum(task['resource_score'] for task in tasks)
        if total_computation > 0:
            for task_entry in tasks:
                computation_ratio = task_entry['resource_score'] / total_computation
                task_entry['task'].current_comp_resource = computation_ratio * queue_weight
                task_entry['task'].alloc_traj.append(task_entry['task'].current_comp_resource)

    def _process_tasks(self, allocated_resources: Dict[str, float], time: float) -> List[ComputeTask]:
        """
        处理任务：根据分配的资源推进任务进度
        """
        completed_tasks = []
        tasks_to_remove = []

        flag = False
        rest_time = 0.0

        cur_tasks = self._get_all_tasks()

        for task_entry in cur_tasks:
            task = task_entry['task']
            task_id = task.task_id

            # 检查任务是否超时
            time_elapsed = self.current_slot - task.creation_slot
            if time_elapsed > task.max_tolerance_delay:
                # 任务超时，标记为失败
                task.status = TaskStatus.FAILED
                task.failure_reason = f"任务处理超时，已耗时{time_elapsed}时隙，超过最大容忍延迟{task.max_tolerance_delay}时隙"
                task.failure_slot = self.current_slot
                completed_tasks.append(task)
                tasks_to_remove.append(task_entry)
                self.failed_tasks.append(task)
                continue

            # 获取分配的资源
            allocated_resource = allocated_resources.get(task_id, 0.0)

            # 当前时隙能算完的任务量
            cur_comp = (time * self.total_computation_power) * allocated_resource
            cycles_processed = min(cur_comp, task.remaining_computation)

            e_inc = self.energy_coeff * cycles_processed * ((allocated_resource * self.total_computation_power) ** 2)

            task.energy_total += e_inc
            if task.energy_traj is None:
                task.energy_traj = []
            task.energy_traj.append(e_inc)
            self.energy_last_slot += e_inc
            if cur_comp > task.remaining_computation:
                flag = True
                rest_time = time - (task.remaining_computation * 8 * 10e6 / (allocated_resource * self.total_computation_power))

                if debug:
                    print(f"在第{self.current_slot}时隙，有剩余时间可以算任务，rest_time = {rest_time}")

            # 先看能不能当前时隙内算完，能算完还要再做一次其它任务的计算
            if cur_comp >= task.remaining_computation:
                task.remaining_computation = 0
                task.status = TaskStatus.COMPLETED
                task.completion_slot = self.current_slot
                task.time_total = self.current_slot - task.creation_slot + (
                            task.remaining_computation * 8 * 10e6 / (allocated_resource * self.total_computation_power))
                completed_tasks.append(task)
                tasks_to_remove.append(task_entry)
                self.completed_tasks.append(task)

                if debug:
                    print(f"在{self.current_slot}时隙，完成了agent：{task.agent_id}的任务：{task.task_id}")

            else:
                # 没算完
                task.remaining_computation -= cur_comp

        # 移除已完成或失败的任务,从优先级队列中移除
        self._remove_tasks_from_queues(tasks_to_remove)

        if flag:
            # 更新任务优先级（因为时间变化）
            self._update_task_priorities()

            # 分配资源
            allocated_resources = self._allocate_resources()

            # 处理任务
            completed_tasks.append(self._process_tasks(allocated_resources, rest_time))
            pass

        return completed_tasks

    def _remove_tasks_from_queues(self, tasks_to_remove: List[Dict[str, Any]]) -> None:
        """从队列中移除指定的任务"""
        for task_entry in tasks_to_remove:
            priority = task_entry['priority']
            queue = self.priority_queues[priority]

            # 找到并移除任务
            for i, task in enumerate(queue):
                if (task['task'].task_id == task_entry['task'].task_id and
                        task['arrival_slot'] == task_entry['arrival_slot']):
                    queue.pop(i)
                    break

    def _update_task_priorities(self) -> None:
        """更新所有任务的优先级（因为时间推移）"""
        # 清空队列，重新添加任务（按新优先级）
        all_tasks_entries = self._get_all_tasks()

        # 清空所有队列
        for priority in self.priority_queues:
            self.priority_queues[priority].clear()

        # 重新计算优先级并添加
        for task_entry in all_tasks_entries:
            task = task_entry['task']
            new_priority = TaskPriorityEvaluator.determine_priority(task, self.current_slot)

            # 更新任务条目
            new_task_entry = {
                'task': task,
                'agent_id': task.agent_id,
                'arrival_slot': task_entry['arrival_slot'],
                'priority': new_priority,
                'resource_score': TaskPriorityEvaluator.calculate_resource_score(task)
            }

            # 添加到新优先级队列
            self.priority_queues[new_priority].append(new_task_entry)

    def process_time_slot(self) -> List[ComputeTask]:
        """
        处理一个时隙
        返回本时隙完成的任务列表
        """
        self.current_slot += 1
        self.total_processing_slots += 1
        self.energy_last_slot = 0.0

        # 更新任务优先级（因为时间变化）
        self._update_task_priorities()

        # 分配资源
        allocated_resources = self._allocate_resources()

        # 处理任务，这里计算一下能耗，最后任务完成时将统计出来的任务移交core
        completed_tasks = self._process_tasks(allocated_resources, self.slot_time)

        return completed_tasks

    def get_queue_status(self) -> Dict[str, Any]:
        status = {
            'high': len(self.priority_queues[TaskPriority.HIGH]),
            'medium': len(self.priority_queues[TaskPriority.MEDIUM]),
            'low': len(self.priority_queues[TaskPriority.LOW]),
            'current_slot': self.current_slot,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_processing_slots': self.total_processing_slots,
            'energy_last_slot': self.energy_last_slot
        }
        w = self.queue_weights
        status['weights'] = {
            'high': w.get(TaskPriority.HIGH, 0.0),
            'medium': w.get(TaskPriority.MEDIUM, 0.0),
            'low': w.get(TaskPriority.LOW, 0.0)
        }
        metrics = self._compute_queue_metrics()
        status['near_deadline'] = {
            'high': int(metrics[TaskPriority.HIGH]['near']),
            'medium': int(metrics[TaskPriority.MEDIUM]['near']),
            'low': int(metrics[TaskPriority.LOW]['near'])
        }
        return status
