import configparser


class GetConfig():
    def __init__(self, config_path):
        self.config_path = config_path
    def __call__(self):
        return self.collect_config()
    def collect_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config

# cnf = GetConfig("D:\Detection_Classification\configs\data_config.INI")
# cnf.collect_config()


