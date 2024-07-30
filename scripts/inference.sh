#! /bin/bash

[ "$#" -ne 1 ] && printf "Expected one argument, but received %d\n" "$#" && exit 1;

printf "Starting inference -- Generating %d images ...\n" "$1"

python3 src/EntryPoint.py --inference --inference-count "$1"