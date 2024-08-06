#! /bin/bash

[ "$#" -ne 2 -a "$#" -ne 1 ] && printf "Expected argument range \
[1-2], but received %d\n" "$#" && exit 1;

[ "$#" -eq 1 ] && set -- "$@" "64" 

cmd="python3 src/EntryPoint.py --inference --inference-count $1 --target $2"

printf "Starting inference -- Generating %d images , using target %d ...\n" "$1" "$2"

eval "${cmd}"