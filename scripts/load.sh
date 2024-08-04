#! /bin/bash

[ "$#" -ne 3 ] && printf "Exptected two positional arguments but only received\
 $#. Aborting ...\n" && exit 1

printf "generate images per epochs = %d -- total epochs = %d --\
 no data augmentation = %d\n" "$1" "$2" "$3"

cmd="python3 ${PWD}/src/EntryPoint.py --load --epochs "$2" --generate-per-epoch "$1" --no-augment"
[ "$3" -eq 1 ] && cmd="${cmd%' --no-augment'}"

eval "${cmd}"