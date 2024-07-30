#! /bin/bash

[ "$#" -ne 3 ] && printf "Exptected two positional arguments but only received\
 $#. Aborting ...\n" && exit 1

printf "generate images per epochs = %d -- total epochs = %d --\
 no data augmentation = %d\n" "$1" "$2" "$3"

cmd="python3 ${PWD}/src/EntryPoint.py --load --epochs "$1" --generate-per-epoch "$2" --no-augment"
[ "$3" -eq 1 ] && cmd="${cmd%' --no-augment'}"

eval "${cmd}"