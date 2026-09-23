"""Element allocation guards for detached component topology changes."""


def next_unoccupied_shell_id(mesh, lower_bound):
    """Advance a local allocator past IDs created by another topology operation."""
    return max(int(lower_bound), max(mesh.shells, default=0) + 1)
