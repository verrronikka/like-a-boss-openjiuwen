"""Deterministic budget planning and uncertainty summaries; no LLM calls."""
import math


def knapsack_select(tasks, budget, quantum=10, margin=1.25):
    """Exact 0/1 DP for conservatively rounded estimated costs and known rewards.

    Optimality concerns this surrogate problem only, not actual agent success/cost.
    """
    if budget < 0 or not math.isfinite(budget) or quantum < 1 or margin < 1 or not math.isfinite(margin):
        raise ValueError("Invalid planning budget/quantum/margin")
    capacity = math.floor(budget / quantum)
    values, masks = [0.0] * (capacity + 1), [0] * (capacity + 1)
    for index, task in enumerate(tasks):
        cost = task.estimated_cost
        if not math.isfinite(cost) or cost <= 0 or not math.isfinite(task.reward) or task.reward < 0:
            raise ValueError("Task costs must be positive and rewards nonnegative")
        weight = max(1, math.ceil(cost * margin / quantum))
        for b in range(capacity, weight - 1, -1):
            candidate = values[b - weight] + task.reward
            if candidate > values[b] + 1e-12:
                values[b] = candidate
                masks[b] = masks[b - weight] | (1 << index)
    best = max(range(capacity + 1), key=lambda b: (values[b], -b))
    mask = masks[best]
    return [task for i, task in enumerate(tasks) if mask & (1 << i)]
