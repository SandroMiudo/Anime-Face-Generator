#! /bin/bash

################################
# predefined run invocations : 

# S,1 , S,2 , S,3
# M,1 , M,2 , M,3
# L,1 , L,2 , L,3

# predefined decays : 

# D,E,? , D,C,? , D,P,? : ? = [1-3]

# predefined learning rates :

# LR,1, LR,2 , LR,3

# predefined epochs :

# E,1, E,2 , E,3

# predefined noise vectors :

# N,1 , N,2 , N,3

# predefined batch size :

# B,1 , B,2 , B,3

# example usage : 

################################

readonly DECAY_P=\
"--polynomial-decay"
readonly DECAY_E=\
"--exponential-decay"
readonly DECAY_C=\
"--constant-decay"
readonly DECAY_C_BOUND=\
"--constant-decay-boundaries"
readonly DECAY_C_VALUE=\
"--constant-decay-values"
readonly BATCH_S=\
"--batch-size"
readonly EPOCHS=\
"--epochs"
readonly NOISE_V=\
"--noise-vector"
readonly LEARNING=\
"--learning-rate"

readonly LEVEL_1=1
readonly LEVEL_2=2
readonly LEVEL_3=3

readonly BATCH="B"
readonly NOISE="N"
readonly EPOCH="E"
readonly LEARN="LR"
readonly DECAY="D"

readonly DECAY_P_LEVEL_1="${DECAY_P} 0.0004 100000 0.0001 1.0"
readonly DECAY_P_LEVEL_2="${DECAY_P} 0.0004 100000 0.0001 0.75"
readonly DECAY_P_LEVEL_3="${DECAY_P} 0.0004 100000 0.0001 0.5"

readonly DECAY_E_LEVEL_1="${DECAY_E} 0.0004 100000 0.95"
readonly DECAY_E_LEVEL_2="${DECAY_E} 0.0004 100000 0.80"
readonly DECAY_E_LEVEL_3="${DECAY_E} 0.0004 100000 0.75"

readonly DECAY_C_LEVEL_1="${DECAY_C} 2 ${DECAY_C_BOUND} 5000 10000 15000 5000 10000 15000 \
${DECAY_C_VALUE} 0.0005 0.0004 0.0003 0.0002 0.0004 0.0003 0.0002 0.0001"
readonly DECAY_C_LEVEL_2="${DECAY_C} 2 ${DECAY_C_BOUND} 5000 10000 15000 5000 10000 15000 \
${DECAY_C_VALUE} 0.0004 0.0003 0.0002 0.0001 0.0005 0.0004 0.0003 0.0002"
readonly DECAY_C_LEVEL_3="${DECAY_C} 2 ${DECAY_C_BOUND} 5000 10000 15000 5000 10000 15000 \
${DECAY_C_VALUE} 0.0004 0.0003 0.0002 0.0001 0.0004 0.0003 0.0002 0.0001"

readonly LEARNING_LEVEL_1="${LEARNING} 0.0004"
readonly LEARNING_LEVEL_2="${LEARNING} 0.0003"
readonly LEARNING_LEVEL_3="${LEARNING} 0.0001"

readonly EPOCHS_LEVEL_1="${EPOCHS} 50"
readonly EPOCHS_LEVEL_2="${EPOCHS} 75"
readonly EPOCHS_LEVEL_3="${EPOCHS} 100"

readonly NOISE_V_LEVEL_1="${NOISE_V} 64"
readonly NOISE_V_LEVEL_2="${NOISE_V} 100"
readonly NOISE_V_LEVEL_3="${NOISE_V} 128"

readonly BATCH_S_LEVEL_1="${BATCH_S} 64"
readonly BATCH_S_LEVEL_2="${BATCH_S} 128"
readonly BATCH_S_LEVEL_3="${BATCH_S} 256"

readonly S_LEVEL_1="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_1} ${NOISE_V_LEVEL_1} ${BATCH_S_LEVEL_3}"
readonly S_LEVEL_2="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_2} ${NOISE_V_LEVEL_1} ${BATCH_S_LEVEL_3}"
readonly S_LEVEL_3="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_3} ${NOISE_V_LEVEL_1} ${BATCH_S_LEVEL_3}"

readonly M_LEVEL_1="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_1} ${NOISE_V_LEVEL_2} ${BATCH_S_LEVEL_2}"
readonly M_LEVEL_2="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_2} ${NOISE_V_LEVEL_2} ${BATCH_S_LEVEL_2}"
readonly M_LEVEL_3="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_3} ${NOISE_V_LEVEL_2} ${BATCH_S_LEVEL_2}"

readonly L_LEVEL_1="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_1} ${NOISE_V_LEVEL_3} ${BATCH_S_LEVEL_1}"
readonly L_LEVEL_2="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_2} ${NOISE_V_LEVEL_3} ${BATCH_S_LEVEL_1}"
readonly L_LEVEL_3="${DECAY_C_LEVEL_1} ${EPOCHS_LEVEL_3} ${NOISE_V_LEVEL_3} ${BATCH_S_LEVEL_1}"

readonly ENTRY="python3 src/EntryPoint.py"

CMD=""

parse_args () {
  for option in "$@"; do
    local active_option=$(cut -d ',' -f1  <<< "${option}")
    local active_level=$(cut -d ',' -f2 <<< "${option}")
    [ "${active_option}" = "S" ] || [ "${active_option}" = "M" ] || [ "${active_option}" = "L" ] && 
    [ "${#CMD}" -gt 0 ] && exit 1
    case "${active_option}" in
      "${BATCH}")
        [ ${active_level} -eq ${LEVEL_1} ] && CMD="${CMD} ${BATCH_S_LEVEL_1}"  
        [ ${active_level} -eq ${LEVEL_2} ] && CMD="${CMD} ${BATCH_S_LEVEL_2}"
        [ ${active_level} -eq ${LEVEL_3} ] && CMD="${CMD} ${BATCH_S_LEVEL_3}"
        ;;
      "${NOISE}")
        [ ${active_level} -eq ${LEVEL_1} ] && CMD="${CMD} ${NOISE_V_LEVEL_1}"
        [ ${active_level} -eq ${LEVEL_2} ] && CMD="${CMD} ${NOISE_V_LEVEL_2}"
        [ ${active_level} -eq ${LEVEL_3} ] && CMD="${CMD} ${NOISE_V_LEVEL_3}"
        ;;
      "${EPOCH}")
        [ ${active_level} -eq ${LEVEL_1} ] && CMD="${CMD} ${EPOCHS_LEVEL_1}"
        [ ${active_level} -eq ${LEVEL_2} ] && CMD="${CMD} ${EPOCHS_LEVEL_2}"
        [ ${active_level} -eq ${LEVEL_3} ] && CMD="${CMD} ${EPOCHS_LEVEL_3}"
        ;;
      "${LEARN}")
        [ ${active_level} -eq ${LEVEL_1} ] && CMD="${CMD} ${LEARNING_LEVEL_1}"
        [ ${active_level} -eq ${LEVEL_2} ] && CMD="${CMD} ${LEARNING_LEVEL_2}"
        [ ${active_level} -eq ${LEVEL_3} ] && CMD="${CMD} ${LEARNING_LEVEL_3}"
        ;;
      "${DECAY}")
        local active_decay="${active_level}"
        active_level=$(cut -d ',' -f3 <<< "${option}")
        [ "${active_decay}" = "E" ] && [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${DECAY_E_LEVEL_1}"
        [ "${active_decay}" = "E" ] && [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${DECAY_E_LEVEL_2}"
        [ "${active_decay}" = "E" ] && [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${DECAY_E_LEVEL_3}"

        [ "${active_decay}" = "C" ] && [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${DECAY_C_LEVEL_1}"
        [ "${active_decay}" = "C" ] && [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${DECAY_C_LEVEL_2}"
        [ "${active_decay}" = "C" ] && [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${DECAY_C_LEVEL_3}"

        [ "${active_decay}" = "P" ] && [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${DECAY_P_LEVEL_1}"
        [ "${active_decay}" = "P" ] && [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${DECAY_P_LEVEL_2}"
        [ "${active_decay}" = "P" ] && [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${DECAY_P_LEVEL_3}"
        ;;
      "L" )
        [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${L_LEVEL_1}"
        [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${L_LEVEL_2}"
        [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${L_LEVEL_3}"
        ;;
      "M")
        [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${M_LEVEL_1}"
        [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${M_LEVEL_2}"
        [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${M_LEVEL_3}"
        ;;
      "S")
        [ "${active_level}" -eq ${LEVEL_1} ] && CMD="${CMD} ${S_LEVEL_1}"
        [ "${active_level}" -eq ${LEVEL_2} ] && CMD="${CMD} ${S_LEVEL_2}"
        [ "${active_level}" -eq ${LEVEL_3} ] && CMD="${CMD} ${S_LEVEL_3}"
        ;;
      *)
        printf "Received invalid option. Aborting ...\n";exit 1
        ;;
  esac
done
}

print_help () {
    printf "# predefined decays : D,E , D,C , D,P\n\t\
# decay levels : D,1 , D,2 , D,3\n\
# predefined learning rates : LR,1, LR,2 , LR,3\n\
# predefined epochs : E,1, E,2 , E,3\n\
# predefined noise vectors : N,1 , N,2 , N,3\n\
# predefined batch size : B,1 , B,2 , B,3\n"
}

[ $# -eq 0 ] && exit 1

[ ${1} = "-h" ] && print_help && exit 0

parse_args "$@"

CMD="${ENTRY} ${CMD}"

printf "$CMD\n"

#eval "${CMD}"