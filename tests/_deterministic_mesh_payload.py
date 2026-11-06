"""Test-only deterministic payload comparison; runtime timing samples are observational."""

GEOMETRY_TIMING_KEYS = ('geometry.intersection_application_time_ns', 'geometry.intersection_arrangements_time_ns', 'geometry.intersection_candidates_time_ns', 'geometry.intersection_clipping_time_ns', 'geometry.intersection_planning_time_ns', 'geometry.intersection_support_and_clipping_time_ns')


def deterministic_mesh_payload(payload):
    """Compare semantics and timing-key membership, excluding elapsed samples.

    The geometry owner defines these six integer counters as inclusive elapsed
    nanoseconds. Match their precise serialized path: similarly named values
    anywhere else and every other operation counter remain part of equality.
    """
    timing_path = ("hybrid_diagnostics", "runtime", "operation_counts")

    def normalize(value, path=()):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if path == timing_path and key in GEOMETRY_TIMING_KEYS:
                    if type(item) is not int or item < 0:
                        raise ValueError(f"{key} must be a nonnegative exact integer")
                    result[key] = 0
                elif key == "phase_seconds" and isinstance(item, dict):
                    result[key] = sorted(item)
                else:
                    result[key] = normalize(item, (*path, key))
            return result
        if isinstance(value, list):
            return [normalize(item, (*path, index)) for index, item in enumerate(value)]
        return value

    return normalize(payload)
