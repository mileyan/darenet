import os
from shutil import which
from collections import OrderedDict


def update_key(s):
    for prefix in ["conv", "norm", "relu"]:
        for suffix in ["1", "2"]:
            s = s.replace(f"{prefix}.{suffix}", f"{prefix}{suffix}")
    return s


# Handle state dict from early PyTorch versions
def update_state_dict(state_dict_in):
    state_dict_out = OrderedDict()
    for k, v in state_dict_in.items():
        state_dict_out[update_key(k)] = v
    return state_dict_out


def is_tool(name):
    return which(name) is not None


def get_gdrive():
    if is_tool("gdrive"):
        return "gdrive"
    if not os.path.isfile("/tmp/gdrive"):
        os.system("wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64 -d /tmp/gdrive --no-check-certificate")
    os.system("chmod +x /tmp/gdrive")
    return "/tmp/gdrive"


def download_gdrive(token, dst):
    os.makedirs(dst, exist_ok=True)
    command = f"{get_gdrive()} download {token} --path {dst}"
    os.system(command)
