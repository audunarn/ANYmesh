# Headless volume consumer example

`solid_volume_controlled.py` uses candidate installed ANYgeometry and ANYmesher
wheels plus optional installed ANYsolid (with NumPy, SciPy and scikit-fem). Copy
the script outside source trees and run:

```text
python -I solid_volume_controlled.py --output result.json
```

The six explicit Tet4 cells form a controlled box fixture. The script validates
the complete owner snapshot/mesh handoff and checks stiffness, full strain/stress
tensors and loads for P1/P2. It launches no native provider and establishes no
generated-mesh qualification. ANYsolid remains an optional consumer dependency.

See [volume API guidance](../docs/volume_mesh.md) for the production guard, local
provider artifact caveats and measured native studies; see
[installed delivery evidence](../docs/installed_delivery_20261028.md) for the
fresh offline wheel smoke and exact commands.
