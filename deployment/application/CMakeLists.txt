# Copyright (c) 2022 GreenWaves Technologies SAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of GreenWaves Technologies SAS nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Modified by: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Modified by: Jose Fonseca, ETH Zurich (jcastro@student.ethz.ch)

cmake_minimum_required(VERSION 3.19)

###############################################################################
# Panel Control
###############################################################################
set(TARGET_NAME "app")

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

set(BOARD_NAME gap9_evk)
set(PMSIS_OS freertos)
set(platform gvsoc)

###############################################################################
# App's options interpretation
###############################################################################
# set (FLASH_TYPE DEFAULT)
# set (RAM_TYPE   DEFAULT)

set (FREQ_CL 370) # MEASUREMENT
set (FREQ_FC 370) # MEASUREMENT
set (FREQ_SFU 370) # MEASUREMENT
set (FREQ_PE 370) # MEASUREMENT
set (VOLTAGE 800) # MEASUREMENT

# set (FREQ_CL 240) # MEASUREMENT
# set (FREQ_FC 240) # MEASUREMENT
# set (FREQ_SFU 240) # MEASUREMENT
# set (FREQ_PE 240) # MEASUREMENT
# set (VOLTAGE 650) # MEASUREMENT

# set (CLUSTER_STACK_SIZE 8192)
# set (CLUSTER_SLAVE_STACK_SIZE 4096)
# set (TOTAL_STACK_SIZE $(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES)))
# set (MODEL_L1_MEMORY $(shell expr 120000 \- $(TOTAL_STACK_SIZE)))
# set (MODEL_L2_MEMORY 1000000)
# set (MODEL_L3_MEMORY 8000000)
set (HELLOWORLD_CLUSTER 1)


# set (NUM_CORES 8)
# set (GAP_SDK 1)
# set (TARGET_CHIP_FAMILY_GAP9 1)


# get proper file path
set(FILENAME_INPUT_DATA        input_test.dat)
set(FILENAME_INPUT_WEIGHTS     weights.dat)
set(FILENAME_INPUT_SS_MEANS    means.dat)
set(FILENAME_INPUT_SS_STDS     stds.dat)
set(FILENAME_INPUT_CLASSF_B    weights_bias.dat)
set(FILENAME_INPUT_CLASSF_W    weights_classf.dat)

set(HYDRA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(TARGET_SRCS              
                             app.c  
                             ${HYDRA_SRCS}/hydra_init.c 
                             ${HYDRA_SRCS}/hydra_reset.c 
                             ${HYDRA_SRCS}/hydra_utils.c 
                             ${HYDRA_SRCS}/hydra_forward.c 
                             ${HYDRA_SRCS}/hydra_convolve.c 
                             ${HYDRA_SRCS}/hydra_sparse_scale.c 
                             ${HYDRA_SRCS}/hydra_classifier.c
                            )


list(APPEND TARGET_PREPROCESSOR 

                                # -DNUM_CORES=${NUM_CORES}
                                # -DGAP_SDK=${GAP_SDK}
                                # -DCONFIG_GAP9_EVK
                                # -DTARGET_CHIP_FAMILY_GAP9

                                # -DFS_READ_FS
                                -DCONFIG_HELLOWORLD_CLUSTER=${HELLOWORLD_CLUSTER}
                                # -DFLASH_TYPE=${FLASH_TYPE}
                                # -DUSE_${FLASH_TYPE}
                                # -DUSE_${RAM_TYPE}
                                # -DSTACK_SIZE=${CLUSTER_STACK_SIZE}
                                # -DSLAVE_STACK_SIZE=${CLUSTER_SLAVE_STACK_SIZE}
                                # -DMODEL_L2_MEMORY=${MODEL_L2_MEMORY}
                                # -DFABRIC
                                # -DCLUSTER

                                -DFILE_INPUT_DATA=${FILENAME_INPUT_DATA}
                                -DFILE_INPUT_WEIGHTS=${FILENAME_INPUT_WEIGHTS}
                                -DFILE_INPUT_SS_MEANS=${FILENAME_INPUT_SS_MEANS}
                                -DFILE_INPUT_SS_STDS=${FILENAME_INPUT_SS_STDS}
                                -DFILE_INPUT_CLASSF_B=${FILENAME_INPUT_CLASSF_B}
                                -DFILE_INPUT_CLASSF_W=${FILENAME_INPUT_CLASSF_W}

                                -DTARGET_GAP9=1
                                -DVECTORIZE=1
                                #-DPARALLELIZE=1
                                -DQUANT_8BIT=1
)


list (APPEND TARGET_INCS 
                         -I$ENV{GAP_SDK_HOME}/libs/openmp/include
                         -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include/bsp
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include/bsp/ram
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include/bsp/flash
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include/bsp/hyperflash
                         -I$ENV{GAP_SDK_HOME}/rtos/pmsis/bsp/include/bsp/fs
                         -I$ENV{GAP_SDK_HOME}/rtos/sfu/include                       
)

list(APPEND TARGET_CFLAGS   -O3
                            # -s
                            # -w
                            # -mno-memcpy
                            # -fno-tree-loop-distribute-patterns
                            # -Wno-incompatible-pointer-types
)


project (${TARGET_NAME} C ASM)

add_executable (${TARGET_NAME} ${TARGET_SRCS})

target_compile_options(${TARGET_NAME} PUBLIC
                                                ${TARGET_INCS} 
                                                ${TARGET_PREPROCESSOR}
                                                ${TARGET_CFLAGS}
                        )

project(${TARGET_NAME} C ASM)

###############################################################################
# Adding a readfs file --> defaut layout | comment out for version with layout
###############################################################################


list (APPEND READFS_FILES 
                         ${FILENAME_INPUT_DATA} 
                         ${FILENAME_INPUT_WEIGHTS} 
                         ${FILENAME_INPUT_SS_MEANS} 
                         ${FILENAME_INPUT_SS_STDS} 
                         ${FILENAME_INPUT_CLASSF_W} 
                         ${FILENAME_INPUT_CLASSF_B}
                         ${FILENAME_DUMMY}
                     )


###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
