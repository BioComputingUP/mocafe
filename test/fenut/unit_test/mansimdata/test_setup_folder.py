import mocafe.fenut.mansimdata as mansim


def test_setup_data_folder(tmpdir):
    # setup data_folder
    data_folder = mansim.setup_data_folder(mansim.test_sim_name, base_location=tmpdir)
    assert data_folder.exists() and str(data_folder) == f"{tmpdir}/{mansim.default_saved_sim_folder_name}" \
                                                        f"/{mansim.test_sim_name}/0000", \
        "The folder should exist and its name should be saved_sim/test/0000"


def test_setup_multiple_data_folder(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(mansim.test_sim_name, base_location=tmpdir)
    data_folder2 = mansim.setup_data_folder(mansim.test_sim_name, base_location=tmpdir)

    assert data_folder1.exists() \
           and str(data_folder1) == f"{tmpdir}/{mansim.default_saved_sim_folder_name}/{mansim.test_sim_name}/0000" \
           and data_folder2.exists() \
           and str(data_folder2) == f"{tmpdir}/{mansim.default_saved_sim_folder_name}/{mansim.test_sim_name}/0001", \
           "Both folder should exist and their name should be saved_sim/test/0000 and saved_sim/test/0001"


def test_setup_runtime(tmpdir):
    data_folder = mansim.setup_data_folder(base_location=tmpdir)
    assert data_folder.exists() \
           and str(data_folder) == f"{tmpdir}/{mansim.default_runtime_folder_name}", \
           "The folder should be 'tmpdir/runtime/'"
