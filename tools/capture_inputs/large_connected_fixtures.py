"""Public-owner fixtures with analytic connectivity and material expectations."""
from dataclasses import dataclass, field
import math

import numpy as np

from anygeometry import GeometryModel, OrientedEdge, trim_face
from anygeometry.generators import cone, cylinder


@dataclass
class ConnectedFixture:
    name: str
    model: GeometryModel
    operands: list = field(default_factory=list)
    areas: dict = field(default_factory=dict)
    joints: list = field(default_factory=list)
    point_joints: list = field(default_factory=list)
    curve_joints: list = field(default_factory=list)

    def plate(self, corners, area):
        face = self.model.add_plate(self.model.add_points(corners))
        self.model.add_sheet((face,))
        self.operands.append(self.model.handle('face', face))
        self.areas[face] = area
        return ('face', face)

    def beam(self, start, end):
        edge = self.model.add_line(*self.model.add_points((start, end)))
        member = self.model.add_member((edge,))
        self.operands.append(self.model.handle('member', member))
        return ('member', member)

    def joint(self, first, second, start, end):
        self.joints.append({'owners': (first, second), 'start': start, 'end': end})


def connected_strip(count=10):
    """Exactly count operands; adjacent panels share boundaries, each has a stiffener."""
    if count < 2 or count % 2:
        raise ValueError('strip operand count must be positive and even')
    fixture = ConnectedFixture(f'connected-strip-{count}', GeometryModel())
    previous = None
    for i in range(count // 2):
        panel = fixture.plate(((i, 0., 0.), (i+1., 0., 0.),
                               (i+1., 1., 0.), (i, 1., 0.)), 1.)
        if previous is not None:
            fixture.joint(previous, panel, (i, 0., 0.), (i, 1., 0.))
        start, end = (i+.5, 0., 0.), (i+.5, 1., 0.)
        if i % 2:
            stiffener = fixture.plate((start, end, (i+.5, 1., 1.), (i+.5, 0., 1.)), 1.)
        else:
            stiffener = fixture.beam(start, end)
        fixture.joint(panel, stiffener, start, end)
        previous = panel
    return fixture


def connected_hub(connections=100):
    """One plate with crossing shell and beam stiffeners; connections+1 operands."""
    if connections < 2 or connections % 2:
        raise ValueError('hub connection count must be positive and even')
    fixture = ConnectedFixture(f'connected-hub-{connections}', GeometryModel())
    host = fixture.plate(((0., 0., 0.), (10., 0., 0.),
                          (10., 10., 0.), (0., 10., 0.)), 100.)
    shells, beams = [], []
    for i in range(connections // 2):
        t = 10. * (i+1) / (connections//2+1)
        start, end = (t, 0., 0.), (t, 10., 0.)
        shell = fixture.plate((start, end, (t, 10., 1.), (t, 0., 1.)), 10.)
        fixture.joint(host, shell, start, end)
        a, b = (0., t, 0.), (10., t, 0.)
        beam = fixture.beam(a, b)
        fixture.joint(host, beam, a, b)
        shells.append((shell, t))
        beams.append((beam, t))
    for shell, x in shells:
        for beam, y in beams:
            fixture.point_joints.append({'owners': (host, shell, beam), 'position': (x, y, 0.)})
    return fixture


def _wall_area(controls, vector):
    """Independent Gauss integral of the polynomial surface Jacobian."""
    controls = np.asarray(controls, float)
    degree = len(controls)-1
    derivative = degree*np.diff(controls, axis=0)
    def integral(order):
        x, weights = np.polynomial.legendre.leggauss(order)
        t = (x+1.)/2.
        du = sum(math.comb(degree-1, i)*(t**i*(1.-t)**(degree-1-i))[:, None]*p
                 for i, p in enumerate(derivative))
        return float(weights @ np.linalg.norm(np.cross(du, vector), axis=1)/2.)
    low, high = integral(64), integral(128)
    assert abs(low-high) < 1e-10
    return high


def concave_boundary_junctions():
    """Concave host, circular void, through-hole stiffeners and boundary T/corner joints."""
    fixture = ConnectedFixture('concave-boundary-junctions', GeometryModel())
    host = fixture.plate(((0., 0., 0.), (6., 0., 0.), (6., 2., 0.),
                          (2., 2., 0.), (2., 6., 0.), (0., 6., 0.)), 20.-math.pi/16.)
    def point(angle):
        return (1.+.25*math.cos(angle), 1.+.25*math.sin(angle), 0.)
    ring = fixture.model.add_points([point(i*math.pi/2.) for i in range(4)])
    arcs = [fixture.model.add_arc(ring[i], fixture.model.add_point(*point((i+.5)*math.pi/2.)),
                                   ring[(i+1)%4]) for i in range(4)]
    trim_face(fixture.model, host[1], (tuple(OrientedEdge(e, True) for e in arcs),))
    shell = fixture.plate(((1., 0., 0.), (1., 2., 0.), (1., 2., 1.), (1., 0., 1.)), 2.)
    cross = fixture.beam((0., 1., 0.), (2., 1., 0.))
    for low, high in ((0., .75), (1.25, 2.)):
        fixture.joint(host, shell, (1., low, 0.), (1., high, 0.))
        fixture.joint(host, cross, (low, 1., 0.), (high, 1., 0.))
    across = fixture.beam((0., 2., 0.), (6., 2., 0.))
    stub = fixture.beam((1., 2., 0.), (1., 5., 0.))
    boundary = fixture.plate(((0., 0., 0.), (6., 0., 0.), (6., 0., 1.), (0., 0., 1.)), 6.)
    side = fixture.beam((0., 0., 0.), (0., 6., 0.))
    for owner, a, b in ((across, (0., 2., 0.), (6., 2., 0.)),
                        (stub, (1., 2., 0.), (1., 5., 0.)),
                        (boundary, (0., 0., 0.), (6., 0., 0.)),
                        (side, (0., 0., 0.), (0., 6., 0.))):
        fixture.joint(host, owner, a, b)
    fixture.point_joints.extend([
        {'owners': (host, shell, across, stub), 'position': (1., 2., 0.)},
        {'owners': (host, boundary, side), 'position': (0., 0., 0.)},
        {'owners': (shell, cross), 'position': (1., 1., 0.)}])
    return fixture


def connected_mixed(count=10, *, start_bay=0):
    """Ten operands per connected bay: floor, extruded wall and eight quadric panels."""
    if count < 10 or count % 10:
        raise ValueError('mixed operand count must be a positive multiple of ten')
    fixture = ConnectedFixture(f'connected-mixed-{count}', GeometryModel())
    model, previous = fixture.model, None
    for i in range(start_bay, start_bay+count//10):
        dx = 10.*i
        floor = fixture.plate(((dx-3., -2., 0.), (dx+7., -2., 0.),
                               (dx+7., 4., 0.), (dx-3., 4., 0.)), 60.)
        if previous is not None:
            fixture.joint(previous, floor, (dx-3., -2., 0.), (dx-3., 4., 0.))
        previous = floor
        controls = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
        vector = (.25, 0., 1.5)
        if i % 3 == 1:
            controls, vector = ((0., 0., 0.), (2., 2., 0.), (4., 0., 0.)), (0., 0., 2.)
        vertices = model.add_points([(x+dx, y, z) for x, y, z in controls])
        edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
        wall, = model.extrude((edge,), vector)
        fixture.operands.append(model.handle('face', wall))
        fixture.areas[wall] = _wall_area(controls, vector)
        before = set(model.faces)
        if i % 3 == 2:
            generated = cone(.1, .9, 8., origin=(dx-2., .5, .8), axis=(1., 0., 0.),
                             radial_direction=(0., 1., 0.), circumferential_segments=8)
            panel_area = math.pi*(.1+.9)*math.hypot(8., .8)/8.
        else:
            generated = cylinder(.7, 8., origin=(dx-2., .4, .6),
                axis=(1., .2, .1) if i % 3 == 0 else (1., 0., 0.),
                radial_direction=(0., 1., 0.), circumferential_segments=8)
            panel_area = 2.*math.pi*.7
        # Revolved generators also author longitudinal Members by default.
        # This fixture counts bare panels; stiffener Members are exercised by
        # strip/hub fixtures. Remove those owners explicitly before insertion
        # so every authored face/member is among the selected operands.
        for member in tuple(generated.members):
            generated.remove_member(member)
        model.insert_model(generated)
        panels = sorted(set(model.faces)-before)
        assert len(panels) == 8
        fixture.operands.extend(model.handle('face', f) for f in panels)
        fixture.areas.update({f: panel_area for f in panels})
        fixture.curve_joints.append({'wall': ('face', wall),
            'panels': [('face', f) for f in panels],
            'required_curve': 'QuadricIntersectionCurve' if i % 3 == 1 else 'BezierQuadricCurve'})
    assert len(model.faces)+len(model.members) == len(fixture.operands) == count
    return fixture
