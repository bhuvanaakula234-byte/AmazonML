"""
One entry point for the whole pipeline.

    python run_pipeline.py train    # blocking -> features -> train model -> validate
    python run_pipeline.py test     # blocking -> features -> predict -> write submission files

Set ER_DATA_DIR and ER_OUTPUT_DIR environment variables first if your
dataset/output folders aren't in the default ./dataset and ./output
locations (see config.py).
"""

import argparse
import os

import config


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("stage", choices=["train", "test"], help="which stage to run")
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if args.stage == "train":
        import train_model
        train_model.main()
    else:
        import predict
        predict.main()


if __name__ == "__main__":
    main()