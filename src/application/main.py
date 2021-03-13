from grpc_manager import GrpcManager
from src.application.config import Configuration
from src.application.manager.manager import Manager

# application config
config = Configuration('.env')
# manager singleton
manager = Manager(config)
# grpc singleton
grpc_manager = GrpcManager(manager)
