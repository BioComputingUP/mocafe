import pandas as pd
import pathlib
from pandas_ods_reader import read_ods


def from_dict(parameters: dict):
    """
    Creates a mocafe Parameters object from a Python dictionary.

    The structure of the Python dictionary must be::

        param_dict = {
            "name_param_a": <value_param_a>,
            "name_param_b": <value_param_b>,
            ...
        }


    :param parameters: the parameters dictionary
    :return: the Parameters object
    """
    param_df = pd.DataFrame({
        "name": parameters.keys(),
        "sim_value": [parameters[param] for param in parameters]
    })
    return Parameters(param_df)


def from_ods_sheet(file: pathlib.Path, sheet: str):
    """
    Creates a mocafe Parameters object form a sheet of an ``.odf`` file.

    The given sheet must have at least to columns: ``name``, with the parameters' names; and ``sim_value``, with the
    value of the parameters. Any other column, for instance containing the measure units, the reference of the
    parameters, or other informations, will be stored inside the Parameters object and used for reporting informations,
    but are not compulsory.

    :param file: file path
    :param sheet: sheet of the file to be used to load parameters
    :return: the Parameters object
    """
    # check if file exists
    if not file.exists():
        raise RuntimeError(f"Parameters file {str(file)} does not exist.")
    # check suffix
    if file.suffix != ".ods":
        raise NotImplementedError(f"Detected file with suffix {file.suffix}. Only .ods files are supported.")
    # import ods file
    param_df: pd.DataFrame = read_ods(str(file), sheet=sheet)
    return Parameters(param_df)


class Parameters:
    """
    Class representing the simulation parameters.

    It is basically a wrapper for a pandas.DataFrame, which contains everything about the simulation parameters.
    """
    def __init__(self, param_df: pd.DataFrame):
        """
        inits a Parameters object from a given ``pandas.DataFrame`` which contains the names and the values of the
        parameters.

        :param param_df: the ``pandas.DataFrame`` containing the parameters names and values.
        """
        # set param_df as parameters dataframe
        self.param_df: pd.DataFrame = param_df
        # set param name as index
        self.param_df.set_index("name", inplace=True)

    def get_value(self, name: str):
        """
        Get the parameter value from the given name.

        :param name: parameter name
        :return: the value of the parameter with the given name.
        """
        return self.param_df.loc[name, "sim_value"]

    def set_value(self, name: str, new_value):
        """
        Set a value for the parameter of the given name.

        :param name: parameter name
        :param new_value: the new value for the parameter
        :return:
        """
        self.param_df.loc[name, "sim_value"] = new_value

    def as_dataframe(self):
        """
        Get the ``pandas.DataFrame`` representing the Parameters object

        :return: the dataframe representing the Parameters object
        """
        return self.param_df

    def is_value_present(self, name):
        """
        Check if the parameters object contains a value for the parameter of the given name

        :param name: the name of the given value
        :return: True if a value is present for the given parameter name; False otherwise.
        """
        return pd.isna(self.get_value(name))

    def is_parameter(self, name):
        """
        Check if the given parameter name correspond to a parameter inside the parameters object

        :param name: the name of the putative parameter
        :return: True if there is a reference for the given parameter, False otherwise
        """
        return name in self.param_df.index


