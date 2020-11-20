import os
import argparse
from utils import download_gdrive


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="checkpoints", help='checkpoint folder path')
args = parser.parse_args()


if __name__ == "__main__":
    model_zoo = {
        "market1501_res50":       "1u4HD-9vlyfpc9sKEqUTcsm1-whR3bjTo",
        "market1501_dense201":    "1nJ_GYXbkFI26BCkcEmuCIsJbYqzEk6YL",
        "mars_res50":    "1_WS38dhRNp8C9t0itEdI2LI6A4rqYKJ_",
        "mars_dense201":          "1Adv3dbL_2PWURWYA5TA1HErdVu2DVOGv",
        "cuhk_detected_res50":    "12qrsilTGQ9X9MhFwR2g3AHHDT7UsKnIn",
        "cuhk_detected_dense201": "1EEHhAff28_L2u-G14jg0MHbO_ManQnfD",
        "cuhk_labeled_res50":     "1AJY2u8PMWtTkLoRvOEcnSF_QR3Cx9gnX",
        "cuhk_labeled_dense201":  "1IsVEYc2AV2cGovt015cQ3WcL48U-tFik",
        "duke_res50":             "1B1BR9p6K-wW1oOkmDQZPfiyj2l4zcdc9",
        "duke_dense201":          "1BwfjlMk3K7sgBPcBs6gzCciBC6X8Q9hL"
    }
    for name, token in model_zoo.items():
        download_gdrive(token, dst=args.dir)
    os.system(f"mv {args.dir}/market1501_hed201.pth.tar {args.dir}/market1501_dense201.pth.tar")
