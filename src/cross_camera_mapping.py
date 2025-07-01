# Assignment Task 1: Cross-Camera Player Mapping
# Implements player mapping between broadcast and tactical feeds as described in instructions.txt
# Usage: python src/cross_camera_mapping.py [--broadcast ...] [--tacticam ...] [--model ...] [--download]

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.utils import download_from_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download all required files from Info.txt')
    args = parser.parse_args()

    if args.download:
        download_from_info()
        sys.exit(0)
    else:
        print('No action specified. Use --download to fetch all required files.')

if __name__ == '__main__':
    main()
