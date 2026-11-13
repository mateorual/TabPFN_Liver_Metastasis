"""Negative Cox partial log-likelihood with the Breslow approximation for ties.

    loss = -(1 / D) * sum_{i: event_i = 1} [ eta_i - log sum_{j: T_j >= T_i} exp(eta_j) ]

D is the number of events. Under Breslow every member of a tied event time shares
the full risk set {j : T_j >= t}, including the other tied subjects. Censored rows
appear only inside risk sets, never as events. The risk-set sums are suffix
log-sum-exps over an ascending, stable sort, so no exp() of a raw linear predictor
is ever formed.
"""
import torch

SUPPORTED_TIES = ("breslow",)


def cox_ph_loss(log_risk, durations, events, ties="breslow"):
    if ties not in SUPPORTED_TIES:
        raise ValueError(f"ties={ties!r} is not implemented; supported: {SUPPORTED_TIES}")
    eta = log_risk.reshape(-1)
    t = durations.reshape(-1).to(eta.dtype)
    e = events.reshape(-1).to(eta.dtype)
    if not ((e == 0) | (e == 1)).all():
        raise ValueError("events must be 0 (censored) or 1 (event)")
    n_events = e.sum()
    if n_events == 0:
        # No event: the partial likelihood is constant; keep the graph, contribute nothing.
        return eta.sum() * 0.0
    t_sorted, order = torch.sort(t, stable=True)
    eta_s, e_s = eta[order], e[order]
    # suffix_lse[k] = log sum_{m >= k} exp(eta_s[m])
    suffix_lse = torch.flip(torch.logcumsumexp(torch.flip(eta_s, [0]), 0), [0])
    # first index of each row's time value: the start of its risk set {T_j >= T_i}
    first = torch.searchsorted(t_sorted.contiguous(), t_sorted.contiguous(), right=False)
    log_denominator = suffix_lse[first]
    return -((eta_s - log_denominator) * e_s).sum() / n_events
