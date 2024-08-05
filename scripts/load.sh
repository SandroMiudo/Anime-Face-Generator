#! /bin/bash

[ "$#" -ne 4 ] && printf "Exptected four positional arguments but received\
 $#. Aborting ...\n" && exit 1

printf "generate images per epochs = %d -- total epochs = %d --\
 no data augmentation = %d\n -- target = %d\n" "$1" "$2" "$3" "$4"

cmd="python3 ${PWD}/src/EntryPoint.py --load --epochs $2\
 --generate-per-epoch $1 --target $4 --no-augment"
[ "$3" -eq 1 ] && cmd="${cmd%' --no-augment'}"

eval "${cmd}"