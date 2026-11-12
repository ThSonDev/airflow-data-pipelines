import torch
from absa_core import get_logger

log = get_logger("checkGPU")
log.info("PyTorch version: %s", torch.__version__)
if torch.cuda.is_available():
    log.info("CUDA is available")
    log.info("GPU: %s (total %d)", torch.cuda.get_device_name(0), torch.cuda.device_count())
else:
    log.warning("CUDA is not available, running on CPU")
