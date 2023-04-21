import time
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, memory_size=50, memory_duration=20, max_range=(-1, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.memory_size = memory_size
        self.memory_duration = memory_duration

        self.max_range = max_range

        self.times = []
        self.errors = []

    def setTuning(self, tuning):
        kp, ki, kd = tuning

        self.kp = kp
        self.ki = ki
        self.kd = kd

    def __call__(self, error):
        self.times.append(time.perf_counter_ns() * 1e-9)
        self.errors.append(error)

        if len(self.times) > self.memory_size and self.times[-1] - self.times[0] > self.memory_duration:
            self.times.pop(0)
            self.errors.pop(0)

        if len(self.times) < 2:
            return 0

        dts = np.diff(np.array(self.times))

        d = (self.errors[-1] - self.errors[-2]) / dts[-1]

        i = np.sum(np.array(self.errors)[1:] * dts)

        cor = np.clip(self.kp * error + self.ki * i + self.kd * d, self.max_range[0], self.max_range[1])

        return cor

    def clearHistory(self):
        self.times.clear()
        self.errors.clear()
