import time
import torch


class ProfileLog:
    def __init__(self, name: str):
        self.name = name

        # time
        self.start_time = None
        self.end_time = None
        self.elapsed = None

        # memory (bytes)
        self.mem_before = None
        self.mem_after = None
        self.mem_peak = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.mem_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.mem_after = torch.cuda.memory_allocated()
            self.mem_peak = torch.cuda.max_memory_allocated()

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    def summary(self):
        def to_mb(x):
            return x / 1024 / 1024 if x is not None else None

        return {
            "name": self.name,
            "time_s": self.elapsed,
            "mem_before_MB": to_mb(self.mem_before),
            "mem_after_MB": to_mb(self.mem_after),
            "mem_peak_MB": to_mb(self.mem_peak),
        }


class Logger:
    def __init__(self):
        self.logs = []

    def profile(self, name):
        log = ProfileLog(name)
        self.logs.append(log)
        return log

    def print(self):
        print("\n===== Profiling Result =====")
        for log in self.logs:
            s = log.summary()
            print(f"[{s['name']}]")
            print(f"  time: {s['time_s']:.4f}s")

            if s["mem_before_MB"] is not None:
                print(f"  mem_before: {s['mem_before_MB']:.2f} MB")
                print(f"  mem_after : {s['mem_after_MB']:.2f} MB")
                print(f"  mem_peak  : {s['mem_peak_MB']:.2f} MB")
            print()