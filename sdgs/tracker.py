"""Token usage and GPU power tracking utilities."""
import threading
import time


class TokenTracker:
    """Accumulates token usage from API responses."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

    def update(self, usage):
        """Update from an API response's usage object.

        Args:
            usage: An object or dict with prompt_tokens, completion_tokens, total_tokens.
        """
        if usage is None:
            return
        if hasattr(usage, "prompt_tokens"):
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0
        elif isinstance(usage, dict):
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
        self.request_count += 1

    def report(self):
        """Print token usage summary."""
        print("\n" + "=" * 50)
        print("TOKEN USAGE")
        print("=" * 50)
        print(f"Requests:          {self.request_count}")
        print(f"Prompt tokens:     {self.prompt_tokens:,}")
        print(f"Completion tokens: {self.completion_tokens:,}")
        print(f"Total tokens:      {self.total_tokens:,}")
        if self.request_count > 0:
            print(f"Avg tokens/req:    {self.total_tokens // self.request_count:,}")


class GPUTracker:
    """Tracks GPU power draw using pynvml, computes total kWh.

    Gracefully does nothing if pynvml is not installed or no NVIDIA GPU is available.
    """

    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._samples = []
        self._running = False
        self._thread = None
        self._available = False
        self._nvml = None

        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._available = True
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def start(self):
        """Begin background power sampling."""
        if not self._available:
            return
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and compute totals."""
        if not self._available or not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _sample_loop(self):
        """Background thread that samples GPU power."""
        while self._running:
            try:
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.time(), power_mw / 1000.0))  # Convert to watts
            except Exception:
                pass
            time.sleep(self.sample_interval)

    @property
    def total_kwh(self) -> float:
        """Compute total energy in kWh from samples."""
        if len(self._samples) < 2:
            return 0.0
        total_joules = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            avg_watts = (self._samples[i][1] + self._samples[i - 1][1]) / 2
            total_joules += avg_watts * dt
        return total_joules / 3_600_000  # joules to kWh

    @property
    def avg_power_watts(self) -> float:
        """Average power draw in watts."""
        if not self._samples:
            return 0.0
        return sum(w for _, w in self._samples) / len(self._samples)

    @property
    def duration_seconds(self) -> float:
        """Total sampling duration in seconds."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def report(self):
        """Print GPU power summary."""
        if not self._available:
            return
        if not self._samples:
            return
        print("\n" + "=" * 50)
        print("GPU POWER USAGE")
        print("=" * 50)
        print(f"Duration:          {self.duration_seconds:.1f}s")
        print(f"Avg power:         {self.avg_power_watts:.1f}W")
        print(f"Total energy:      {self.total_kwh:.6f} kWh")
        print(f"Samples collected: {len(self._samples)}")
