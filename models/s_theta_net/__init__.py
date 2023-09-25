from .s_theta import TLPENetwork
def get_model(config):
    if config.network == 'TLPE':
        return TLPENetwork(config)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
