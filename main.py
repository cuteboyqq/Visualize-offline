from task.visualize import Visualize
import yaml
from config.args import Args

# Load the YAML configuration file and return its content as a dictionary
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__=="__main__":

    # Load configuration settings from the specified YAML file
    config = load_config('config/config.yaml')
    
    # Initialize Args object with the loaded configuration
    args = Args(config)

    # Create Historical object using the Args object
    visual = Visualize(args)

    # Visualize historical data based on mode and JSON log source specified in config
    visual.visualize(mode=args.mode, jsonlog_from=args.jsonlog_from)

  

    
 