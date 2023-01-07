#!/bin/sh
set -eu

dir_name=${1}

mkdir ${dir_name}
mv loss_log*.txt ${dir_name}
mv *.png ${dir_name}
