"""Frozen SG1 limits and mandatory inventory (schema 1)."""
from itertools import product

SCHEMA = 'anymesher.sg1/1'
LIMITS = dict(support_relative=1e-10, size_median_min=.5, size_median_max=1.5,
    size_fraction_min=.95, size_lower=.25, size_upper=2., size_max=4.,
    remote_change=.20, geometry_over_h=.02, normal_degrees=5., area_relative=.005,
    owner_area_convergence=1e-8, normalized_jacobian=.05, aspect=20.,
    free_residual=1e-8, constraint_residual=1e-8, equilibrium=1e-6,
    analytical=1e-6, reference_agreement=.01, solution_agreement=.05,
    operation_seconds=600, attempt_seconds=7200)
BASELINES = {'ANYmesh':'d9df8795402a94c4f7ac76e771258ae722744c30','ANYsolver':'be41a77a'}


def inventory():
    rows=[]
    for case in ('P','F','CY','CO','R','C','T-R','T-C'):
        sizes=(.5,.25,.125) if case in ('P','F') else (.6,.3,.15)
        for level,h in enumerate(sizes):
            for graded in ((True,) if case.startswith('T-') else (False,True)):
                for member in ((False,True) if case in ('CY','CO','R','C') else (False,)):
                    rows.append(dict(id=f'{case}-L{level}-g{int(graded)}-b{int(member)}',
                                     case=case,h=h,level=level,graded=graded,member=member))
    return rows
