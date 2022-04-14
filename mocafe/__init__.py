import configparser

# get version
config = configparser.ConfigParser()
with open("./setup.cfg", 'r') as cfg_file:
    config.read_file(cfg_file)

__version__ = config["metadata"]["version"]
