#! /bin/bash

[ "$#" -ne 4 -a "$#" -ne 3 ] && printf "Exptected argument range [3-4], but received\
 $#. Aborting ...\n" && exit 1

gen_str="Generate images per epochs = %d -- total epochs = %d --\
 no data augmentation = %d\n"

[ "$#" -eq 3 ] && set -- "$@" "64"

printf "Generate images per epochs = %d -- total epochs = %d --\
 no data augmentation = %d\n -- target = %d\n" "$1" "$2" "$3" "$4"

cmd="python3 ${PWD}/src/EntryPoint.py --load --epochs $2\
 --generate-per-epoch $1 --target $4 --no-augment"

[ "$3" -eq 1 ] && cmd="${cmd%' --no-augment'}"

eval "${cmd}"