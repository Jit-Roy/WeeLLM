from pathlib import Path
from typing import Union

def get_seeker(model_dir: Union[str, Path], cache_to_ram: bool = False):
    """
    Factory function to return the appropriate Safetensors seeker.
    If cache_to_ram is True, it returns SafetensorsRAMSeeker which loads the
    full tensor file into CPU RAM to bypass slow disk I/O on cloud instances.
    Otherwise, it returns SafetensorsDiskSeeker which streams directly from disk.
    """
    if cache_to_ram:
        from weellm.ram_seek import SafetensorsRAMSeeker
        return SafetensorsRAMSeeker(model_dir)
    else:
        from weellm.disk_seek import SafetensorsDiskSeeker
        return SafetensorsDiskSeeker(model_dir)
