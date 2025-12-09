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

    def __post_init__(self):
        """初始化后设置剩余计算量"""
        if self.remaining_computation is None:
            self.remaining_computation = self.computation_requirement


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

    def __init__(self, server_id: int, total_computation_power: float = 10.0e9):
        """
        初始化服务器

        Args:
            server_id: 服务器标识
            total_computation_power: 服务器总计算能力 (GHz = e9 CPU周期/s)
        """
        self.server_id = server_id
        self.total_computation_power = total_computation_power  # CPU周期/s
        self.slot_time = 0.1  # 时隙大小 0.1s

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

    def _allocate_resources(self) -> Dict[str, float]:
        """
        为所有任务分配计算资源
        相同优先级的任务同时处理，按计算量比例分配资源
        """
        cur_tasks = self._get_cur_tasks()

        # 如果当前无任务
        if not cur_tasks:
            return {}

        # 按权重比例分配资源
        allocated_resources = {}
        self._allocate_resources_by_computation(cur_tasks)
        for task_entry in cur_tasks:
            task_id = task_entry['task'].task_id
            allocated_resource = task_entry['task'].current_comp_resource
            allocated_resources[task_id] = allocated_resource

        return allocated_resources

    def _allocate_resources_by_computation(self, tasks: List[Dict[str, Any]]) -> None:
        """
        相同优先级任务按照计算量大小比例分配资源
        计算量越大的任务分配越多资源
        """
        total_computation = sum(task['resource_score'] for task in tasks)

        if total_computation > 0:
            for task_entry in tasks:
                # 按计算量比例分配资源
                computation_ratio = task_entry['resource_score'] / total_computation
                task_entry['task'].current_comp_resource = computation_ratio
                # self.allocated_resources[task_entry['agent_id']] = computation_ratio

    def _process_tasks(self, allocated_resources: Dict[str, float], time: float) -> List[ComputeTask]:
        """
        处理任务：根据分配的资源推进任务进度
        """
        completed_tasks = []
        tasks_to_remove = []

        flag = False
        rest_time = 0.0

        cur_tasks = self._get_cur_tasks()

        for task_entry in cur_tasks:
            task = task_entry['task']
            task_id = task.task_id

            # 检查任务是否超时
            time_elapsed = self.current_slot - task.creation_slot
            if time_elapsed > task.max_tolerance_delay:
                # 任务超时，标记为失败
                task.status = TaskStatus.FAILED
                task.failure_reason = f"任务处理超时，已耗时{time_elapsed}时隙，超过最大容忍延迟{task.max_tolerance_delay}时隙"
                completed_tasks.append(task)
                tasks_to_remove.append(task_entry)
                self.failed_tasks.append(task)
                continue

            # 获取分配的资源
            allocated_resource = allocated_resources.get(task_id, 0.0)

            # 当前时隙能算完的任务量
            cur_comp = (time * self.total_computation_power / (8 * 10e6)) * allocated_resource
            if cur_comp > task.remaining_computation:
                flag = True
                rest_time = time - (task.remaining_computation * 8 * 10e6 / (allocated_resource * self.total_computation_power))
                print(f"在第{self.current_slot}时隙，有剩余时间可以算任务，rest_time = {rest_time}")

            # 先看能不能当前时隙内算完，能算完还要再做一次其它任务的计算
            if cur_comp >= task.remaining_computation:
                task.remaining_computation = 0
                task.status = TaskStatus.COMPLETED
                task.completion_slot = self.current_slot
                completed_tasks.append(task)
                tasks_to_remove.append(task_entry)
                self.completed_tasks.append(task)
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
            completed_tasks = self._process_tasks(allocated_resources, rest_time)
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

        # 更新任务优先级（因为时间变化）
        self._update_task_priorities()

        # 分配资源
        allocated_resources = self._allocate_resources()

        # 处理任务
        completed_tasks = self._process_tasks(allocated_resources, self.slot_time)

        return completed_tasks

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态信息"""
        return {
            'high': len(self.priority_queues[TaskPriority.HIGH]),
            'medium': len(self.priority_queues[TaskPriority.MEDIUM]),
            'low': len(self.priority_queues[TaskPriority.LOW]),
            'current_slot': self.current_slot,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_processing_slots': self.total_processing_slots
        }

    def print_detailed_status(self) -> None:
        """打印详细状态信息"""
        status = self.get_queue_status()
        print(f"=== 服务器 {self.server_id} 状态 (时隙 {self.current_slot}) ===")
        print(f"队列任务数: 高优先级{status['high']}个, 中优先级{status['medium']}个, 低优先级{status['low']}个")
        print(
            f"统计: 完成{status['completed_tasks']}个, 失败{status['failed_tasks']}个, 总处理时隙{status['total_processing_slots']}")

        # 打印每个队列的任务详情
        for priority in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            queue = self.priority_queues[priority]
            if queue:
                print(f"\n{priority.value}优先级队列:")
                for i, task_entry in enumerate(queue):
                    task = task_entry['task']
                    print(f"  任务{i + 1}: {task.task_id}, 剩余计算量: {task.remaining_computation:.1f}KB, "
                          f"状态: {task.status.value}")


# 测试代码
def test_priority_queue_server():
    """测试优先级队列服务器"""
    # 1 时隙 0.01 s
    # 创建服务器
    server = PriorityQueueServer("server_1", total_computation_power=10.0)  # 8 GHz

    print("=== 测试多级优先级队列服务器（时间步长版本） ===\n")

    # 创建测试任务
    tasks = [
        ComputeTask(
            task_id="task_1", agent_id="agent_1",
            computation_requirement=1000.0,  # 1000KB
            max_tolerance_delay=500,  # 500个时隙, 5s
            creation_slot=0,
            current_comp_resource=0.0
        ),
        ComputeTask(
            task_id="task_2", agent_id="agent_2",
            computation_requirement=600.0,  # 600KB
            max_tolerance_delay=400,  # 4s
            creation_slot=0,
            current_comp_resource=0.0
        ),
        ComputeTask(
            task_id="task_3", agent_id="agent_3",
            computation_requirement=400.0,  # 400KB
            max_tolerance_delay=10,  # 0.1s
            creation_slot=0,
            current_comp_resource=0.0
        )
    ]

    # 添加任务到服务器
    print("=== 添加初始任务 ===")
    for task in tasks:
        success = server.add_task(task)
        status = "成功" if success else "失败"
        priority = TaskPriorityEvaluator.determine_priority(task, 0)
        print(
            f"任务 {task.task_id} 添加{status}，优先级: {priority.value}，计算量: {task.computation_requirement}KB")

    # 模拟多个时隙
    total_slots = 18  # 18 * 0.1s
    new_tasks_added = False

    for slot in range(1, total_slots + 1):
        print(f"\n--- 时隙 {slot} 开始 ---")

        # 在第3时隙添加新任务
        if slot == 3 and not new_tasks_added:
            new_task = ComputeTask(
                task_id="task_5", agent_id="agent_5",
                computation_requirement=12.0,
                max_tolerance_delay=4,
                creation_slot=slot,
                current_comp_resource=0.0
            )
            success = server.add_task(new_task)
            if success:
                print(f"时隙 {slot} 添加新任务: {new_task.task_id}")
                new_tasks_added = True

        # 处理当前时隙
        completed_tasks = server.process_time_slot()

        # 输出本时隙结果
        if completed_tasks:
            print(f"时隙 {slot} 完成的任务:")
            for task in completed_tasks:
                status_info = f"状态: {task.status.value}"
                if task.failure_reason:
                    status_info += f", 原因: {task.failure_reason}"
                print(f"  - 任务 {task.task_id}: {status_info}")
        else:
            print(f"时隙 {slot} 没有任务完成")

        # 打印队列状态
        status = server.get_queue_status()
        print(f"队列状态: 高{status['high']}个, 中{status['medium']}个, 低{status['low']}个")

        # 每2个时隙打印详细状态
        if slot % 2 == 0:
            server.print_detailed_status()

    print(f"\n=== 测试结束 ===")
    final_status = server.get_queue_status()
    print(f"总处理时隙: {final_status['total_processing_slots']}")
    print(f"成功完成任务: {final_status['completed_tasks']}个")
    print(f"失败任务: {final_status['failed_tasks']}个")

    # 显示失败任务详情
    if server.failed_tasks:
        print("\n失败任务详情:")
        for task in server.failed_tasks:
            print(f"  任务 {task.task_id}: {task.failure_reason}")


if __name__ == "__main__":
    test_priority_queue_server()
