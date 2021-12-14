import pathlib
import mocafe.fenut.mansimdata as mansim
from mocafe.fenut.parameters import from_ods_sheet


def test_save_sim_info(tmpdir, odf_sheet_test2):
    data_folder = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name))
    parameters = from_ods_sheet(odf_sheet_test2, "SimParams")
    mansim.save_sim_info(data_folder, 1.0, parameters, "test")
    print(str(data_folder / pathlib.Path("sim_info.html")))
    assert (data_folder / pathlib.Path("sim_info.html")).exists()


def test_save_sim_info_format(tmpdir, odf_sheet_test):
    data_folder = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name))
    execution_time = 1.0
    parameters = from_ods_sheet(odf_sheet_test, "Sheet1")
    sim_name = mansim.test_sim_name
    dateandtime = "test"
    mansim.save_sim_info(data_folder, execution_time, parameters, sim_name, dateandtime)
    sim_info_file = data_folder / mansim.sim_info_file
    # how the sim_info file should look like
    expected_result = f"<article>\n" \
                      f"  <h1>Simulation report </h1>\n" \
                      f"  <h2>Basic informations </h2>\n" \
                      f"  <p>Simulation name: {sim_name} </p>\n" \
                      f"  <p>Execution time: {execution_time / 60} min </p>\n" \
                      f"  <p>Date and time: {dateandtime} </p>\n" \
                      f"  <h2>Simulation rationale </h2>\n" \
                      f"  <p>test </p>\n" \
                      f"  <h2>Parameters used </h2>\n" + \
                      parameters.as_dataframe().to_html() + "\n" +\
                      f"</article>"
    # load the sim_info file
    with open(sim_info_file, "r") as report_file:
        report_txt = report_file.read()
    # see result
    assert report_txt == expected_result, "The two texts should be equal"


def test_save_sim_info_rationale(tmpdir, odf_sheet_test):
    data_folder = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name))
    execution_time = 1.0
    parameters = from_ods_sheet(odf_sheet_test, "Sheet1")
    sim_name = "another_test"
    dateandtime = "test"
    sim_rationale = "A rationale"
    mansim.save_sim_info(data_folder, execution_time, parameters, sim_name,
                         dateandtime=dateandtime, sim_rationale=sim_rationale)
    sim_info_file = data_folder / mansim.sim_info_file
    # how the sim_info file should look like
    expected_result = f"<article>\n" \
                      f"  <h1>Simulation report </h1>\n" \
                      f"  <h2>Basic informations </h2>\n" \
                      f"  <p>Simulation name: {sim_name} </p>\n" \
                      f"  <p>Execution time: {execution_time / 60} min </p>\n" \
                      f"  <p>Date and time: {dateandtime} </p>\n" \
                      f"  <h2>Simulation rationale </h2>\n" \
                      f"  <p>{sim_rationale} </p>\n" \
                      f"  <h2>Parameters used </h2>\n" + \
                      parameters.as_dataframe().to_html() + "\n" +\
                      f"</article>"
    # load the sim_info file
    with open(sim_info_file, "r") as report_file:
        report_txt = report_file.read()
    # see result
    assert report_txt == expected_result, "The two dictionaries should be equal"


def test_sim_info_error(tmpdir, odf_sheet_test):
    data_folder = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name))
    execution_time = 1.0
    parameters = from_ods_sheet(odf_sheet_test, "Sheet1")
    sim_name = "another_test"
    dateandtime = "test"
    sim_rationale = "A rationale"
    error_msg = "An error"
    error = RuntimeError(error_msg)
    mansim.save_sim_info(data_folder, execution_time, parameters, sim_name,
                         dateandtime=dateandtime, sim_rationale=sim_rationale, error_msg=str(error))
    sim_info_file = data_folder / mansim.sim_info_file
    # how the sim_info file should look like
    expected_result = f"<article>\n" \
                      f"  <h1>Simulation report </h1>\n" \
                      f"  <h2>Basic informations </h2>\n" \
                      f"  <p>Simulation name: {sim_name} </p>\n" \
                      f"  <p>Execution time: {execution_time / 60} min </p>\n" \
                      f"  <p>Date and time: {dateandtime} </p>\n" \
                      f"  <h2>Simulation rationale </h2>\n" \
                      f"  <p>{sim_rationale} </p>\n" \
                      f"  <h2>Parameters used </h2>\n" + \
                      parameters.as_dataframe().to_html() + "\n" + \
                      f"  <h2>Errors </h2>\n" \
                      f"  <p>\n" \
                      f"    Error message: \n" \
                      f"    {error_msg} \n" \
                      f"  </p>\n" \
                      f"</article>"
    # load the sim_info file
    with open(sim_info_file, "r") as report_file:
        report_txt = report_file.read()
    # see result
    assert report_txt == expected_result, "The two dictionaries should be equal"
