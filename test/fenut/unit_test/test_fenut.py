import fenics
from mocafe.fenut.fenut import setup_xdmf_files


def test_setup_xdmf_files(get_p0_tmpdir):
    file_names = ["test_1", "test_2", "test_3"]
    files = setup_xdmf_files(file_names, get_p0_tmpdir)

    assert len(files) == 3, "Initialized 3 files, it should be three"

    assert all([isinstance(f, fenics.XDMFFile) for f in files]), "They should be all XDMFFiles"
