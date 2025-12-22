import os
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    import wandb
except Exception:
    wandb = None

class QueueVisualizer:
    def __init__(self, out_dir: str = 'visual_out', summary_interval: int = 50, dpi: int = 120, use_wandb: bool = True, heatmap_window: int = 500):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if out_dir is None or out_dir == '':
            resolved_dir = os.path.join(base_dir, 'visual_out')
        else:
            resolved_dir = out_dir if os.path.isabs(out_dir) else os.path.join(base_dir, out_dir)
        self.out_dir = resolved_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self._last_counts: Dict[int, tuple] = {}
        self.summary_interval = summary_interval
        self.dpi = dpi
        self.history: Dict[int, Dict[str, list]] = {}
        self._last_time: int = 0
        self.use_wandb = use_wandb
        self.heatmap_window = heatmap_window

    def update(self, servers: List, time_slot: int, agents: List = None, metrics: Dict = None):
        if not servers:
            return
        for s in servers:
            status = s.priority_server.get_queue_status()
            high = status['high']
            med = status['medium']
            low = status['low']
            completed = status['completed_tasks']
            failed = status['failed_tasks']
            total = high + med + low
            offload = getattr(s.state, 'num_offload', 0)
            if s.id not in self.history:
                self.history[s.id] = {
                    'time': [], 'total': [], 'high': [], 'medium': [], 'low': [], 'completed': [], 'failed': [], 'offload': [], 'server_util': []
                }
            h = self.history[s.id]
            h['time'].append(time_slot)
            h['total'].append(total)
            h['high'].append(high)
            h['medium'].append(med)
            h['low'].append(low)
            h['completed'].append(completed)
            h['failed'].append(failed)
            h['offload'].append(offload)
            if metrics is not None and 'server_utility' in metrics:
                su = metrics['server_utility'].get(s.id, 0.0)
                h['server_util'].append(su)
            else:
                h['server_util'].append(0.0)
        self._last_time = time_slot

        if self.summary_interval and time_slot % self.summary_interval == 0:
            self.render_summary()

        if agents is not None:
            if not hasattr(self, 'agent_stats'):
                self.agent_stats = {}
            for a in agents:
                aid = a.id
                if aid not in self.agent_stats:
                    self.agent_stats[aid] = {'submitted': 0, 'success': 0, 'failed': 0, 'submitted_ids': set(), 'success_ids': set(), 'failed_ids': set()}
                cur_id = f"{a.id}-{getattr(a.task, '_timeIndex', time_slot)}" if getattr(a, 'task', None) is not None else None
                if getattr(a, 'task', None) is not None and getattr(a.task, '_timeIndex', None) == time_slot and cur_id not in self.agent_stats[aid]['submitted_ids']:
                    self.agent_stats[aid]['submitted'] += 1
                    self.agent_stats[aid]['submitted_ids'].add(cur_id)
                if getattr(a, 'task', None) is not None and a.task._state == 2 and cur_id not in self.agent_stats[aid]['success_ids']:
                    self.agent_stats[aid]['success'] += 1
                    self.agent_stats[aid]['success_ids'].add(cur_id)
                if getattr(a, 'task', None) is not None and a.task._state == 3 and cur_id not in self.agent_stats[aid]['failed_ids']:
                    self.agent_stats[aid]['failed'] += 1
                    self.agent_stats[aid]['failed_ids'].add(cur_id)
            
        if metrics is not None:
            if not hasattr(self, 'global_metrics'):
                self.global_metrics = {'time': [], 'agent_utility_mean': [], 'og_total': []}
            self.global_metrics['time'].append(time_slot)
            self.global_metrics['agent_utility_mean'].append(metrics.get('agent_utility_mean', 0.0))
            self.global_metrics['og_total'].append(metrics.get('og_total', 0.0))

    def render_summary(self):
        if not self.history:
            return
        servers = sorted(self.history.keys())

        for sid in servers:
            h = self.history[sid]
            plt.figure(figsize=(10, 6))
            plt.plot(h['time'], h['high'], label='high', color='red')
            plt.plot(h['time'], h['medium'], label='medium', color='orange')
            plt.plot(h['time'], h['low'], label='low', color='green')
            plt.xlabel('Time Slot')
            plt.ylabel('Queue Length')
            plt.title(f'Server {sid} Queue Levels Over Time')
            plt.legend()
            plt.tight_layout()
            per_server_path = os.path.join(self.out_dir, f'server_{sid}_queues_levels.png')
            plt.savefig(per_server_path, dpi=self.dpi)
            plt.close()

        plt.figure(figsize=(10, 6))
        for sid in servers:
            h = self.history[sid]
            plt.plot(h['time'], h['total'], label=f'Server {sid}')
        plt.xlabel('Time Slot')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Time')
        plt.legend()
        plt.tight_layout()
        q_path = os.path.join(self.out_dir, 'queues_over_time.png')
        plt.savefig(q_path, dpi=self.dpi)
        plt.close()

        # Server utility over time
        plt.figure(figsize=(10, 6))
        for sid in servers:
            h = self.history[sid]
            plt.plot(h['time'], h['server_util'], label=f'S{sid}')
        plt.xlabel('Time Slot')
        plt.ylabel('Server Utility')
        plt.title('Server Utility Over Time')
        plt.legend()
        plt.tight_layout()
        su_path = os.path.join(self.out_dir, 'server_utility_over_time.png')
        plt.savefig(su_path, dpi=self.dpi)
        plt.close()

        # Agent utility mean over time
        if hasattr(self, 'global_metrics') and self.global_metrics['time']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.global_metrics['time'], self.global_metrics['agent_utility_mean'], label='Agent Utility Mean')
            plt.xlabel('Time Slot')
            plt.ylabel('Agent Utility (Mean)')
            plt.title('Agent Utility Over Time')
            plt.legend()
            plt.tight_layout()
            au_path = os.path.join(self.out_dir, 'agent_utility_over_time.png')
            plt.savefig(au_path, dpi=self.dpi)
            plt.close()

            # OG total over time
            plt.figure(figsize=(10, 6))
            plt.plot(self.global_metrics['time'], self.global_metrics['og_total'], label='OG Total')
            plt.xlabel('Time Slot')
            plt.ylabel('OG Total')
            plt.title('Total Objective OG(t) Over Time')
            plt.legend()
            plt.tight_layout()
            og_path = os.path.join(self.out_dir, 'og_total_over_time.png')
            plt.savefig(og_path, dpi=self.dpi)
            plt.close()

        plt.figure(figsize=(10, 6))
        for sid in servers:
            h = self.history[sid]
            plt.plot(h['time'], h['completed'], label=f'Completed S{sid}')
            plt.plot(h['time'], h['failed'], label=f'Failed S{sid}', linestyle='--')
        plt.xlabel('Time Slot')
        plt.ylabel('Count')
        plt.title('Completion/Failure Over Time')
        plt.legend()
        plt.tight_layout()
        cf_path = os.path.join(self.out_dir, 'completion_failure_over_time.png')
        plt.savefig(cf_path, dpi=self.dpi)
        plt.close()

        times = None
        offload_matrix = []
        for sid in servers:
            h = self.history[sid]
            times = h['time'] if times is None else times
            offload_matrix.append(h['offload'])
        if offload_matrix:
            min_len = min(len(row) for row in offload_matrix)
            offload_matrix = [row[-min(self.heatmap_window, min_len):] for row in offload_matrix]
            if times is not None:
                times_window = times[-len(offload_matrix[0]):]
            else:
                times_window = list(range(len(offload_matrix[0])))
            data = np.array(offload_matrix)
            plt.figure(figsize=(12, 6))
            plt.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(label='Offload Count')
            plt.yticks(range(len(servers)), [f'S{sid}' for sid in servers])
            plt.xticks(range(len(times_window)), times_window if len(times_window) <= 20 else [])
            if len(times_window) > 20:
                plt.xlabel(f'Time Slots (last {len(times_window)})')
            else:
                plt.xlabel('Time Slot')
            plt.title('Server Load Heatmap (Offload Count)')
            plt.tight_layout()
            hm_path = os.path.join(self.out_dir, 'load_heatmap.png')
            plt.savefig(hm_path, dpi=self.dpi)
            plt.close()

        if self.use_wandb and wandb is not None:
            log_payload = {
                'queues_over_time': wandb.Image(q_path),
                'completion_failure_over_time': wandb.Image(cf_path)
            }
            for sid in servers:
                per_server_path = os.path.join(self.out_dir, f'server_{sid}_queues_levels.png')
                log_payload[f'queues_levels_S{sid}'] = wandb.Image(per_server_path)
            if 'hm_path' in locals():
                log_payload['load_heatmap'] = wandb.Image(hm_path)
            if os.path.exists(''+su_path):
                log_payload['server_utility_over_time'] = wandb.Image(su_path)
            if 'au_path' in locals():
                log_payload['agent_utility_over_time'] = wandb.Image(au_path)
            if 'og_path' in locals():
                log_payload['og_total_over_time'] = wandb.Image(og_path)
            wandb.log(log_payload, step=self._last_time)

        if hasattr(self, 'agent_stats') and self.agent_stats:
            aids = sorted(self.agent_stats.keys())
            success = [self.agent_stats[aid]['success'] for aid in aids]
            failed = [self.agent_stats[aid]['failed'] for aid in aids]
            x = np.arange(len(aids))
            plt.figure(figsize=(12, 6))
            plt.bar(x, success, color='green')
            plt.bar(x, failed, bottom=success, color='red')
            plt.xticks(x, [f'A{aid}' for aid in aids])
            plt.ylabel('Tasks')
            plt.title('Agent Task Status (Success/Failed)')
            plt.tight_layout()
            agent_bar_path = os.path.join(self.out_dir, 'agent_status_bar.png')
            plt.savefig(agent_bar_path, dpi=self.dpi)
            plt.close()
            if self.use_wandb and wandb is not None:
                wandb.log({'agent_status_bar': wandb.Image(agent_bar_path)}, step=self._last_time)

    def compute_fail_stats_over_episodes(self, episode_length: int, window: int = None):
        if not self.history or episode_length <= 0:
            return [], []
        servers = sorted(self.history.keys())
        max_time = 0
        for sid in servers:
            h = self.history[sid]
            if h['time']:
                max_time = max(max_time, int(h['time'][-1]))
        total_episodes = (max_time // episode_length) + 1 if max_time > 0 else 1
        fail_counts = []
        for epi in range(total_episodes):
            t_start = epi * episode_length
            t_end = (epi + 1) * episode_length
            inc_sum = 0
            for sid in servers:
                h = self.history[sid]
                times = h['time']
                fails = h['failed']
                if not times:
                    continue
                start_val = 0
                end_val = 0
                for i in range(len(times) - 1, -1, -1):
                    if times[i] <= t_start:
                        start_val = int(fails[i])
                        break
                for i in range(len(times) - 1, -1, -1):
                    if times[i] < t_end:
                        end_val = int(fails[i])
                        break
                inc = max(0, end_val - start_val)
                inc_sum += inc
            fail_counts.append(int(inc_sum))
        means = []
        if window is None or window <= 1:
            s = 0.0
            for i, v in enumerate(fail_counts):
                s += float(v)
                means.append(s / float(i + 1))
        else:
            for i in range(len(fail_counts)):
                left = max(0, i + 1 - window)
                chunk = fail_counts[left:i + 1]
                if chunk:
                    means.append(float(sum(chunk)) / float(len(chunk)))
                else:
                    means.append(0.0)
        return fail_counts, means

    def render_fail_mean_over_episodes(self, fail_counts: list, fail_means: list):
        if not fail_counts:
            return None
        x = list(range(1, len(fail_counts) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(x, fail_counts, label='Fail Count / Episode', color='red')
        if fail_means:
            plt.plot(x, fail_means, label='Mean Fail (cum or window)', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Failures')
        plt.title('Failures per Episode and Mean')
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.out_dir, 'failure_mean_over_episodes.png')
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        if self.use_wandb and wandb is not None:
            wandb.log({'failure_mean_over_episodes': wandb.Image(path)}, step=self._last_time)
        return path
