from fractions import Fraction
from anygeometry import GeometryModel, Plane, OrientedEdge, plan_intersections, apply_intersections, to_dict
from anygeometry.structural import ParameterRange


def _edge_between(model, first, second):
    return next(edge for edge in model.edges.values()
                if {edge.start, edge.end} == {first, second})

def build_chain_network(operand_count=4, central=False, repeated_carrier=False,
                        station=Fraction(3,4), mixed=False):
    """Small connected network whose Members are multi-edge boundary chains.

    Each operand plate carries a two-use chain (split side edge plus unsplit
    bottom edge) and a three-use chain sharing the bottom carrier, with the
    two-use chain reversed on alternating operands. The base plate carries one
    two-use boundary chain: split by decomposition seams, or the repeatedly cut bottom
    boundary edge plus the uncut right edge when ``central`` is set.
    """
    if not 3 <= operand_count <= 10:
        raise ValueError('This disposable fixture covers 3-10 operands only, not a kernel cap.')
    model = GeometryModel()
    last = operand_count-2
    base_points = model.add_points(((-1,-2,0),(last+1,-2,0),(last+1,2,0),(-1,2,0)))
    base = model.add_plate(base_points)
    model.set_face_surface(base, Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_sheet((base,))
    half = 2 if central else Fraction(3,2)
    operands = []
    for x in range(operand_count-1):
        corners = model.add_points(((x,-half,-1),(x,half,-1),(x,half,1),(x,-half,1)))
        if mixed:
            edges = (model.add_line(corners[1],corners[0]),
                     model.add_line(corners[1],corners[2]),
                     model.add_line(corners[2],corners[3]),
                     model.add_line(corners[3],corners[0]))
            face = model.add_face(edges)
        else:
            face = model.add_plate(corners)
        model.set_face_surface(face, Plane((x,0,0),(0,1,0),(0,0,1)))
        model.add_sheet((face,))
        bottom = _edge_between(model, corners[0], corners[1])
        right = _edge_between(model, corners[1], corners[2])
        top = _edge_between(model, corners[2], corners[3])
        left = _edge_between(model, corners[3], corners[0])
        two = model.add_member((left.id, bottom.id))
        three = model.add_member((OrientedEdge(bottom.id,not mixed), right.id, top.id))
        if x % 2:
            model.reverse_member(two)
        operands.append({'two':two,'three':three,'bottom':bottom.id,'right':right.id,
                         'top':top.id,'left':left.id,'reversed':bool(x%2)})
    if central:
        first = _edge_between(model, base_points[0], base_points[1])
        second = _edge_between(model, base_points[1], base_points[2])
    else:
        first = _edge_between(model, base_points[1], base_points[2])
        second = _edge_between(model, base_points[2], base_points[3])
    base_member = model.add_member((first.id, second.id))
    if central:
        model.reverse_member(base_member)
    if repeated_carrier:
        model.add_member((second.id, second.id))
    points = []
    for edge, st in ((bottom, Fraction(0)), (left, station), (first, station)):
        a, b = (tuple(Fraction(float(c)) for c in model.vertex_position(v))
                for v in (edge.start, edge.end))
        xyz = tuple((1-st)*u+st*v for u, v in zip(a, b))
        vertex = model.add_point(*(float(c) for c in xyz))
        model.add_attachment(None, 'vertex_on_edge', 'edge', edge.id,
            ParameterRange.point(0.), (ParameterRange.point(float(st)),),
            source_kind='vertex', source_id=vertex, evidence='exact', tolerance_used=1e-9)
        points.append({'carrier':edge.id,'station':st,'xyz':tuple(map(float,xyz)),
                       'vertex':vertex})
    original = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    return model,joint,original,{'operands':operands,'base':{'member':base_member,
                                 'first':first.id,'second':second.id},'points':points}
