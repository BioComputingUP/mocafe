import fenics
import mocafe.fenut.fenut as fu


def is_point_in_mesh(mesh):
    point = fenics.Point(1, 1)
    is_point_in_local_mesh = fu.is_point_inside_mesh(mesh, point)
    # gather
    is_point_in_local_mesh_array = fenics.MPI._comm_world.gather(is_point_in_local_mesh)
    if fenics.MPI._comm_world.Get_rank():
        is_point_in_global_mesh = any(is_point_in_local_mesh_array)
    else:
        is_point_in_global_mesh = None
    is_point_in_global_mesh = fenics.MPI._comm_world.bcast(is_point_in_global_mesh, 0)
    assert is_point_in_global_mesh
