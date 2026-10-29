"""Sufficient rejection of flips against one unchanged physical T3 state."""
import numpy as np


_METRICS = frozenset(('scaled_jacobian','aspect_ratio','minimum_angle','maximum_angle','warpage'))
_THRESHOLDS = ('min_scaled_jacobian','max_aspect_ratio','min_angle','max_angle','max_warpage','max_element_growth')


def _thresholds(settings):
    try:
        values = tuple(getattr(settings,name) for name in _THRESHOLDS)
        return values if all(type(value) in (int,float) and np.isfinite(value) for value in values) else None
    except AttributeError:
        return None


def _signature(report):
    try:
        counts = report['violation_counts']
        if type(counts) is not dict or set(counts) != _METRICS:
            return None
        if any(type(value) is not int or value < 0 for value in counts.values()):
            return None
        poor = report['poor_element_ids']
        if type(poor) not in (list,tuple) or any(type(value) is not int for value in poor):
            return None
        if len(set(poor)) != len(poor):
            return None
        if (report['quality_scope'] != 'published_elements'
                or type(report['invalid_element_count']) is not int or report['invalid_element_count'] != 0
                or type(report['quality_violation_count']) is not int
                or report['quality_violation_count'] != len(poor)
                or type(report['elements_above_maximum_growth']) is not int
                or report['elements_above_maximum_growth'] < 0):
            return None
        growth = report['max_element_growth']
        if type(growth) not in (int,float) or not np.isfinite(growth) or growth < 1.:
            return None
        return (tuple(sorted(counts.items())),tuple(poor),growth,
                report['elements_above_maximum_growth'],report['quality_violation_count'])
    except (KeyError,TypeError,ValueError):
        return None


def prepare_flip_rejection(current, xyz, settings, stored):
    """Return a frame only after complete current-report and connectivity checks.

    ``stored`` is the actual Native point/triangle representation, not its
    rotated/sorted canonical export. This never certifies a trial. A changed,
    differently represented or unknown baseline retains full validation.
    """
    from .core import MeshCore
    from .surface_mesh import _physical_quality_candidate_from_xyz
    signature = _signature(current.report)
    thresholds = _thresholds(settings)
    if signature is None or thresholds is None or type(stored) is not tuple or len(stored) != 2:
        return None
    points,cells = np.asarray(current.points),np.asarray(current.triangles)
    xyz = np.asarray(xyz)
    if (points.ndim != 2 or points.shape[1] != 2 or cells.ndim != 2 or cells.shape[1] != 3
            or xyz.shape != (len(points),3) or not len(cells)
            or points.dtype.kind not in 'fiu' or xyz.dtype.kind not in 'fiu'
            or cells.dtype.kind not in 'iu' or np.any(cells < 0) or np.any(cells >= len(points))
            or not np.isfinite(points).all() or not np.isfinite(xyz).all()
            or not np.array_equal(stored[0],points) or not np.array_equal(stored[1],cells)):
        return None
    lengths = np.linalg.norm(np.roll(xyz[cells],-1,axis=1)-xyz[cells],axis=2).mean(axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        return None
    core = MeshCore(xyz,cells)
    ids = tuple(map(int,core.triangle_ids))
    # Bind report identifiers to the exact OLD stored rows, never a reordered trial.
    rows = {identifier:row for row,identifier in enumerate(ids)}
    if len(rows) != len(cells) or not set(signature[1]) <= set(rows):
        return None
    refreshed = _physical_quality_candidate_from_xyz(current,settings,xyz)
    if _signature(refreshed.report) != signature:
        return None
    incidence = {}
    for row,cell in enumerate(cells):
        if len(set(map(int,cell))) != 3:
            return None
        for a,b in zip(cell,np.roll(cell,-1)):
            incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(row)
    if any(len(owners) > 2 for owners in incidence.values()):
        return None
    pairs = tuple((edge,tuple(owners)) for edge,owners in incidence.items() if len(owners) == 2)
    ratios = tuple(max(lengths[a]/lengths[b],lengths[b]/lengths[a]) for _,(a,b) in pairs)
    if not all(np.isfinite(ratio) for ratio in ratios):
        return None
    maximum = max((1.,*ratios))
    if maximum != signature[2]:
        return None
    witnesses = tuple((edge,frozenset(owners)) for (edge,owners),ratio in zip(pairs,ratios)
                      if ratio == maximum)
    return _FlipRejection(points,cells,xyz,signature,settings,thresholds,
        frozenset(rows[identifier] for identifier in signature[1]),incidence,witnesses,maximum)


class _FlipRejection:
    def __init__(self,points,cells,xyz,signature,settings,thresholds,poor,incidence,witnesses,maximum):
        self.frame = (points.shape,points.dtype.str,points.tobytes(),
                      cells.shape,cells.dtype.str,cells.tobytes(),xyz.shape,xyz.dtype.str,xyz.tobytes())
        self.signature,self.poor,self.incidence = signature,poor,incidence
        self.settings,self.thresholds = settings,thresholds
        self.witnesses,self.maximum = witnesses,maximum

    def rejects(self,edge,current,xyz):
        points,cells,xyz = np.asarray(current.points),np.asarray(current.triangles),np.asarray(xyz)
        if ((points.shape,points.dtype.str,points.tobytes(),cells.shape,cells.dtype.str,cells.tobytes(),
             xyz.shape,xyz.dtype.str,xyz.tobytes()) != self.frame
                or _signature(current.report) != self.signature):
            return False
        if _thresholds(self.settings) != self.thresholds:
            return False
        owners = self.incidence.get(edge,())
        changed = frozenset(owners)
        if len(owners) != 2 or len(changed) != 2 or changed & self.poor:
            return False
        opposites = tuple(int(node) for row in owners for node in cells[row] if int(node) not in edge)
        if len(opposites) != 2 or len(set((*edge,*opposites))) != 4:
            return False
        diagonal = tuple(sorted(opposites))
        if diagonal in self.incidence:
            return False
        # Growth has a hard floor of one. Otherwise retain a concrete exact
        # maximum adjacency with both old cells and its incidence unchanged.
        return self.maximum == 1. or any(not (witness_owners & changed) and witness != diagonal
            for witness,witness_owners in self.witnesses)
