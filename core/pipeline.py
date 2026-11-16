"""Decision workflow for already-trained agents; training and studies live outside core.

This entry preserves the existing qualification, proposal, and best-of solver calls.
Contribution/payment/pruning remain explicit research operations on HostServer.
"""


def solve_decision(server, agents, *, qualification_options=None, custom_iterations=30):
    """Qualify agents, collect proposals, and solve using the supplied server policy."""
    server.phase1_filter_agents(agents, **(qualification_options or {}))
    server.phase2_collect_proposals()
    solution, history, states = server.phase3_best_of_optimization(
        custom_iterations=custom_iterations,
    )
    return {"solution": solution, "history": history, "states": states}
