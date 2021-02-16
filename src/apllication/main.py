from src.apllication.config import Configuration
from src.apllication.manager.manager import Manager

config = Configuration('.env')
manager = Manager(config=config)
manager.process()
