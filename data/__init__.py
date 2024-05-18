from .data_simmim import build_loader_simmim, build_loader_ExpandingMask
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_ExpandingMask(config, logger)
    else:
        return build_loader_finetune(config, logger)