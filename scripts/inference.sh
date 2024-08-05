#! /bin/bash

[ "$#" -ne 2 ] && printf "Expected two arguments, but received %d\n" "$#" && exit 1;

printf "Starting inference -- Generating %d images ...\n" "$1"

python3 src/EntryPoint.py --inference --inference-count "$1" --target "$2"