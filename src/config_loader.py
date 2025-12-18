import yaml 

def load_config(path="config.yaml"):
    """ Loads configuration settings from a YAML file.
    
    args:
        path (str): Path to the YAML configuration file."""
        
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)