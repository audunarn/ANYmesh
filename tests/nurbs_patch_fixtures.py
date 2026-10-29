"""Non-translation NURBS patch fixtures (S5): four iso-parametric edges on the support, per-patch builders."""
import math

from anygeometry import GeometryModel
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface

K2 = (0., 0., 0., 1., 1., 1.)
K1 = (0., 0., 1., 1.)


def biquadratic(height):
    return NurbsSurface(2, 2, K2, K2, tuple(tuple((i / 2., j / 2., height(i, j)) for j in range(3)) for i in range(3)))


def bilinear(p00, p01, p10, p11):
    return NurbsSurface(1, 1, K1, K1, ((p00, p01), (p10, p11)))


def cone_quarter():
    """Quarter cone: circle arc in u (rational), radius 1 -> .5 along v."""
    w = math.sqrt(.5)
    def ring(r):
        return ((r, 0., 0.), (r, r, 0.), (0., r, 0.))
    rows = []
    for i in range(3):
        rows.append(tuple((ring(r)[i][0], ring(r)[i][1], z) for r, z in ((1., 0.), (.5, 1.))))
    weights = tuple((w if i == 1 else 1.,) * 2 for i in range(3))
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), K1, tuple(rows), weights)


PATCHES = {
    'warp_0.1': lambda: bilinear((0., 0., 0.), (0., 1., 0.), (1., 0., 0.), (1., 1., .1)),
    'warp_0.5': lambda: bilinear((0., 0., 0.), (0., 1., 0.), (1., 0., 0.), (1., 1., .5)),
    'twisted_pm_0.3': lambda: bilinear((0., 0., .3), (0., 1., -.3), (1., 0., -.3), (1., 1., .3)),
    'trapezoid_planar': lambda: bilinear((0., 0., 0.), (.3, 1., 0.), (2., 0., 0.), (1.6, 1., 0.)),
    'saddle_biquadratic': lambda: biquadratic(lambda i, j: .6 * (i - 1) * (j - 1)),
    'saddle_mild': lambda: biquadratic(lambda i, j: .1 * (i - 1) * (j - 1)),
    'bump_mild': lambda: biquadratic(lambda i, j: .1 * (i == 1) * (j == 1)),
    'bump_biquadratic': lambda: biquadratic(lambda i, j: .8 * (i == 1) * (j == 1)),
    'quarter_cone_rational': cone_quarter,
}


def patch_model(name):
    support = PATCHES[name]()
    model = GeometryModel()
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, K1, (corners[i], corners[(i + 1) % 4]))) for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], curve) for i, curve in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face
