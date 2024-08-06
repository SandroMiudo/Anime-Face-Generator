#! /bin/bash

[ ! -f ~/.alias ] && touch ~/.alias

e=$(grep -e "ENTRY=.*" ~/.alias)
l=$(grep -e "LOAD=*" ~/.alias)
i=$(grep -e "INF=*" ~/.alias)
t=$(grep -e "TRAIN=*" ~/.alias)

[ -z "${e}" ] && tee --append ~/.alias <<< "alias ENTRY='python3 src/EntryPoint.py'" &>/dev/null
[ -z "${l}" ] && chmod u+x scripts/load.sh && tee --append ~/.alias <<< "alias LOAD=./scripts/load.sh" &>/dev/null
[ -z "${i}" ] && chmod u+x scripts/inference.sh && tee --append ~/.alias <<< "alias INF=./scripts/inference.sh" &>/dev/null
[ -z "${t}" ] && chmod u+x scripts/train.sh && tee --append ~/.alias <<< "alias TRAIN=./scripts/train.sh" &>/dev/null