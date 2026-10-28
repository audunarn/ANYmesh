"""Conservative routing for source-bound material mesh publication.

This query issues no transfer or qualification authority. A caller that needs
curvilinear transfer must use a separately available owner publication service.
"""
from ._material_region_binding import face_requires_curvilinear, region_requires_curvilinear


def requires_material_source_transfer(geometry, mesh):
    from anygeometry import ExtrudedSurface, CoonsSurface, query_material_surface_regions, GeometryError
    from anygeometry.surfaces import Plane, Cylinder, Cone, RuledSurface
    from anygeometry.curves import Straight, Arc
    from anygeometry.exact_curves import EXACT_CURVES
    
    face_ids = tuple(face for face, cells in getattr(mesh, 'elements_of_face', {}).items() if cells)
    if not face_ids:
        face_ids = tuple(geometry.faces)
    if any(face not in geometry.faces for face in face_ids):
        return True
    if any(face_requires_curvilinear(geometry, face) for face in face_ids):
        return True
    for key in face_ids:
        face = geometry.faces[key]
        if face.surface is not None and not isinstance(
                face.surface, (Plane, Cylinder, Cone, RuledSurface, ExtrudedSurface, CoonsSurface)):
            return True
        if any(not isinstance(geometry.edges[use.edge].curve, (Straight, Arc, *EXACT_CURVES))
               for loop in (face.loop, *face.holes) for use in loop):
            return True
    candidates=tuple(face.id for face in (geometry.faces[key] for key in face_ids)
                     if isinstance(face.surface,(ExtrudedSurface,CoonsSurface)))
    if candidates:
        try:
            regions=query_material_surface_regions(geometry,candidates)
        except GeometryError:
            return True
        return any(region_requires_curvilinear(region) for region in regions.regions)
    return False
