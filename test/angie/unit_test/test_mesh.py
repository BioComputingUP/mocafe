from mpi4py import MPI
import mocafe.fenut.fenut as fu


def test_is_point_in_mesh(mesh):
    point = [1., 1., 0.]
    is_point_in_local_mesh = fu.is_point_inside_mesh(mesh, point)
    # gather
    is_point_in_local_mesh_array = MPI.COMM_WORLD.gather(is_point_in_local_mesh, 0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        is_point_in_global_mesh = any(is_point_in_local_mesh_array)
    else:
        is_point_in_global_mesh = None
    is_point_in_global_mesh = MPI.COMM_WORLD.bcast(is_point_in_global_mesh, 0)
    assert is_point_in_global_mesh
