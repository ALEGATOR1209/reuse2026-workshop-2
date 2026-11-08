"""Whole-brain leaky integrate-and-fire model of the fruit fly (Shiu et al. 2024).

Same equations and parameters as `code/run_pytorch.py` in
github.com/eonsystemspbc/fly-brain, rewritten to be event-driven: instead of a
full sparse matmul every 0.1 ms step, only the out-going synapses of neurons
that actually spiked are touched. Typically <1% of the brain spikes per step,
so this is much faster on a laptop CPU.

The connectome weights are never changed: the brain is a fixed "reservoir".
"""

import numpy as np
import torch

PARAMS = {
    "tauSyn": 5.0,        # ms, synaptic time constant
    "tDelay": 1.8,        # ms, synaptic delay
    "vRest": -52.0,       # mV
    "vReset": -52.0,      # mV
    "vThreshold": -45.0,  # mV
    "tauMem": 20.0,       # ms, membrane time constant
    "tRefrac": 2.2,       # ms
    "scalePoisson": 250,  # input spike strength (in units of synapses)
    "wScale": 0.275,      # mV per synapse
}


class FlyBrain:
    def __init__(self, pre, post, weight, n_neurons, input_idx, dt=0.1, params=PARAMS):
        self.n = n_neurons
        self.dt = dt
        self.p = params
        # Synapses grouped by presynaptic neuron (CSR), so a spike of neuron j
        # reads the contiguous slice indptr[j]:indptr[j+1].
        order = np.argsort(pre, kind="stable")
        self._order = order
        self.pre_sorted = pre[order]          # synapse i goes pre_sorted[i] -> post[i]
        self.post = torch.as_tensor(post[order])
        self.w_intact = torch.as_tensor(weight[order] * params["wScale"])
        self.w = self.w_intact
        self.indptr = torch.zeros(n_neurons + 1, dtype=torch.int64)
        self.indptr[1:] = torch.as_tensor(np.bincount(pre, minlength=n_neurons)).cumsum(0)

        self.input_idx = torch.as_tensor(input_idx, dtype=torch.int64)
        self.refrac_steps = torch.full((n_neurons,), round(params["tRefrac"] / dt), dtype=torch.int32)
        self.refrac_steps[self.input_idx] = 0  # stimulated neurons, as in fly-brain
        self.delay_steps = int(params["tDelay"] / dt)
        self.input_alive = torch.ones(len(input_idx))

    def set_lesion(self, silenced=None, cut_synapses=None):
        """Knock out neurons and/or cut connections. Call with no arguments to restore.

        silenced: bool (N,) neurons to silence: all their synapses in and out are removed
            and they get no eye input, like optogenetic silencing in fly-brain.
        cut_synapses: bool (n_synapses,) in the original connectivity-file order.
        Returns the number of connections removed.
        """
        keep = np.ones(len(self.pre_sorted), bool)
        if silenced is not None and silenced.any():
            keep &= ~silenced[self.pre_sorted] & ~silenced[self.post.numpy()]
        if cut_synapses is not None:
            keep &= ~cut_synapses[self._order]
        self.w = self.w_intact * torch.as_tensor(keep, dtype=torch.float32) if not keep.all() else self.w_intact
        dead = silenced[self.input_idx.numpy()] if silenced is not None else np.zeros(len(self.input_idx), bool)
        self.input_alive = torch.as_tensor(~dead, dtype=torch.float32)
        return int((~keep).sum())

    def _propagate(self, spikes):
        """Summed synaptic input (B, N) caused by `spikes` (B, N bool)."""
        B = spikes.shape[0]
        out = torch.zeros(B * self.n)
        b, j = spikes.nonzero(as_tuple=True)
        if len(j) == 0:
            return out.view(B, self.n)
        start = self.indptr[j]
        deg = self.indptr[j + 1] - start
        ev = torch.repeat_interleave(torch.arange(len(j)), deg)                # event of each edge
        first = torch.repeat_interleave(deg.cumsum(0) - deg, deg)
        edge = start[ev] + (torch.arange(len(ev)) - first)                       # synapse index
        out.index_add_(0, b[ev] * self.n + self.post[edge], self.w[edge])
        return out.view(B, self.n)

    @torch.no_grad()
    def run(self, input_rates_hz, duration_ms, generator=None, record=False, on_progress=None):
        """Simulate a batch. input_rates_hz: (B, n_inputs) Poisson rates of the input neurons.

        Returns spike counts per neuron, (B, N) int16. With record=True also returns
        the spikes of the first sample as (step, neuron) index arrays. on_progress(n) is called
        as the batch advances, with n summing to B (fractions of finished samples).
        """
        p, dt = self.p, self.dt
        B, N = input_rates_hz.shape[0], self.n
        n_steps = int(round(duration_ms / dt))

        v = torch.full((B, N), p["vRest"])
        g = torch.zeros(B, N)
        spikes = torch.zeros(B, N, dtype=torch.bool)
        refrac = self.refrac_steps.expand(B, N).clone()
        ring = torch.zeros(self.delay_steps + 1, B, N)   # synaptic delay line
        ptr = 0
        counts = torch.zeros(B, N, dtype=torch.int16)

        p_spike = input_rates_hz * (dt / 1000.0) * self.input_alive  # silenced eye neurons get no input
        kick = p["wScale"] * p["scalePoisson"]
        syn_decay = 1 - dt / p["tauSyn"]
        mem = dt / p["tauMem"]

        ev_steps, ev_neurons = [], []
        for step in range(n_steps):
            recurrent = self._propagate(spikes)
            refrac = torch.where(spikes, 0, refrac + 1)
            can_receive = refrac >= self.refrac_steps
            g_new = g * syn_decay + ring[ptr] * can_receive
            ring[ptr] = recurrent                         # arrives delay_steps+1 steps later
            ptr = (ptr + 1) % ring.shape[0]

            v[:, self.input_idx] += kick * torch.bernoulli(p_spike, generator=generator)
            v += mem * (g - (v - p["vRest"]))
            spikes = v > p["vThreshold"]
            v = torch.where(spikes, p["vReset"], v)
            g = torch.where(spikes, 0.0, g_new)
            counts += spikes
            if on_progress and (step + 1) % 50 == 0:
                on_progress(B * 50 / n_steps)
            if record:
                fired = spikes[0].nonzero().squeeze(1)
                ev_neurons.append(fired)
                ev_steps.append(torch.full_like(fired, step))
        if record:
            return counts, (torch.cat(ev_steps).numpy(), torch.cat(ev_neurons).numpy())
        return counts
