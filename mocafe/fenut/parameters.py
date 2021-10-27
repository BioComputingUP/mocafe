import pandas as pd
import pathlib
from pandas_ods_reader import read_ods


class Parameters:
    def __init__(self, file: pathlib.Path, sheet: str):
        # check if file exists
        if not file.exists():
            raise RuntimeError(f"Parameters file {str(file)} does not exist.")
        # check suffix
        if file.suffix != ".ods":
            raise NotImplementedError(f"Detected file with suffix {file.suffix}. Only .ods files are supported.")
        # import ods file
        self.param_df: pd.DataFrame = read_ods(str(file), sheet=sheet)
        # set param name as index
        self.param_df.set_index("name", inplace=True)

    def get_value(self, name: str):
        return self.param_df.loc[name, "sim_value"]

    def set_value(self, name: str, new_value):
        self.param_df.loc[name, "sim_value"] = new_value

    def as_dataframe(self):
        return self.param_df
