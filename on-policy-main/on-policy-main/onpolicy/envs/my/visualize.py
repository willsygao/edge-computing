import os
from typing import List, Dict
from collections import defaultdict
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
        self.episode_counts: Dict[int, List[float]] = defaultdict(list)

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
            
            au_mean = metrics.get('agent_utility_mean', 0.0)
            og_tot = metrics.get('og_total', 0.0)

            self.global_metrics['time'].append(time_slot)
            self.global_metrics['agent_utility_mean'].append(au_mean)
            self.global_metrics['og_total'].append(og_tot)

            if self.use_wandb and wandb is not None:
                log_data = {
                    'agent_utility_over_time': au_mean,
                    'og_total_over_time': og_tot
                }
                if 'server_utility' in metrics:
                    for sid, val in metrics['server_utility'].items():
                        log_data[f'server_utility_over_time/S{sid}'] = val
                
                # Use commit=False to aggregate with other logs in the same step (managed by my_runner.py)
                wandb.log(log_data, commit=False)

    def render_summary(self):
        if not self.history:
            return
        
        log_payload = {}
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
            if self.use_wandb and wandb is not None:
                log_payload[f'queues_levels_S{sid}'] = wandb.Image(plt)
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
        if self.use_wandb and wandb is not None:
            log_payload['queues_over_time'] = wandb.Image(plt)
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
        # if self.use_wandb and wandb is not None:
        #     log_payload['server_utility_over_time'] = wandb.Image(plt)
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
            # if self.use_wandb and wandb is not None:
            #     log_payload['agent_utility_over_time'] = wandb.Image(plt)
            plt.close()

            # OG total over time
            plt.figure(figsize=(10, 6))
            plt.plot(self.global_metrics['time'], self.global_metrics['og_total'], label='OG Total')
            plt.xlabel('Time Slot')
            plt.ylabel('OG Total')
            plt.title('Total Objective OG(t) Over Time')
            plt.legend()
            plt.tight_layout()
            # if self.use_wandb and wandb is not None:
            #     log_payload['og_total_over_time'] = wandb.Image(plt)
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
        if self.use_wandb and wandb is not None:
            log_payload['completion_failure_over_time'] = wandb.Image(plt)
        plt.close()

        # Heatmap: Episodes on X, Servers on Y, Offload Ratio as color
        if self.episode_counts:
            # Prepare matrix: rows for servers, cols for episodes
            sids = sorted(self.episode_counts.keys())
            if sids:
                num_episodes = len(self.episode_counts[sids[0]])
                # Matrix shape: (num_servers, num_episodes)
                data_matrix = np.zeros((len(sids), num_episodes))
                
                for i, sid in enumerate(sids):
                    data_matrix[i, :] = self.episode_counts[sid]
                
                # Normalize columns to get ratios
                col_sums = data_matrix.sum(axis=0)
                # Avoid division by zero
                col_sums[col_sums == 0] = 1.0
                ratio_matrix = data_matrix / col_sums[np.newaxis, :]

                # Apply window if needed
                if num_episodes > self.heatmap_window:
                    plot_data = ratio_matrix[:, -self.heatmap_window:]
                    x_indices = range(num_episodes - self.heatmap_window, num_episodes)
                else:
                    plot_data = ratio_matrix
                    x_indices = range(num_episodes)

                plt.figure(figsize=(12, 6))
                plt.imshow(plot_data, aspect='auto', cmap='viridis', origin='lower')
                plt.colorbar(label='Offload Ratio')
                plt.yticks(range(len(sids)), [f'S{sid}' for sid in sids])
                
                # X-axis labels
                if len(x_indices) > 20:
                     tick_step = max(1, len(x_indices) // 10)
                     tick_locs = np.arange(0, len(x_indices), tick_step)
                     tick_labels = [str(x_indices[i]) for i in tick_locs]
                     plt.xticks(tick_locs, tick_labels)
                     plt.xlabel(f'Episode (last {len(x_indices)})')
                else:
                     plt.xticks(range(len(x_indices)), [str(i) for i in x_indices])
                     plt.xlabel('Episode')

                plt.title('Server Load Distribution per Episode')
                plt.tight_layout()
                if self.use_wandb and wandb is not None:
                    log_payload['load_heatmap'] = wandb.Image(plt)
                plt.close()

                # --- NEW: Overall Load Distribution Bar Chart ---
                total_loads = [sum(self.episode_counts[sid]) for sid in sids]
                grand_total = sum(total_loads)
                if grand_total > 0:
                    percentages = [(x / grand_total) * 100.0 for x in total_loads]
                else:
                    percentages = [0.0] * len(sids)
                
                plt.figure(figsize=(8, 6))
                bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
                if len(sids) > len(bar_colors):
                    # Fallback if more than 4 servers
                    bar_colors = plt.cm.tab10(np.arange(len(sids)))
                
                bars = plt.bar([f'S{sid}' for sid in sids], percentages, color=bar_colors[:len(sids)])
                plt.ylabel('Load Percentage (%)')
                plt.title(f'Overall Server Load Distribution (Total {num_episodes} Episodes)')
                plt.ylim(0, 100) # Percentage scale
                
                # Add text labels on bars
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if self.use_wandb and wandb is not None:
                    log_payload['overall_load_distribution'] = wandb.Image(plt)
                plt.close()
                # ------------------------------------------------

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
            if self.use_wandb and wandb is not None:
                log_payload['agent_status_bar'] = wandb.Image(plt)
            plt.close()

        if self.use_wandb and wandb is not None and log_payload:
            wandb.log(log_payload, commit=False)

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
        if self.use_wandb and wandb is not None:
            wandb.log({'failure_mean_over_episodes': wandb.Image(plt)}, commit=False)
        plt.close()
        return None

    def reset(self):
        # Accumulate stats from the current history before clearing
        if self.history:
            servers = sorted(self.history.keys())
            # Sum num_offload.
            for sid in servers:
                offloads = self.history[sid]['offload'] # List[int] or List[float]
                total_offload = sum(offloads) if offloads else 0.0
                self.episode_counts[sid].append(total_offload)
            
            # Ensure consistency
            if self.episode_counts:
                max_len = max(len(v) for v in self.episode_counts.values())
                for sid in self.episode_counts:
                    while len(self.episode_counts[sid]) < max_len:
                        self.episode_counts[sid].append(0.0)

        self._last_counts = {}
        self.history = {}
        self._last_time = 0
        self.agent_stats = {}
        self.global_metrics = {'time': [], 'agent_utility_mean': [], 'og_total': []}
