// Copyright (C) 2024-2025 ETH Zurich
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0
// ==============================================================================
// 
// Author: Jose Fonseca, ETH Zurich (jcastro@student.ethz.ch)
// Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)


#include "pmsis.h"
#include "omp.h"
#include <bsp/bsp.h>
#include <bsp/ram.h>
#include <bsp/fs/hostfs.h>
// #include <bsp/flash/hyperflash.h>
#include "../include/hydra.h"
#include "../include/hydra_defines.h"

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

// Board Defines (for power measuerments)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)


// Problem Defines
#define INPUT_SZ  2
#define NUM_SAMPLES 500
#define BUFFER_SZ   1500

//#define DETAILED_PROFILING 1

#ifdef PARALLELIZE
static PI_L1 RCKINT   inX[BUFFER_SZ], inX_diff[BUFFER_SZ], inW[BUFFER_SZ];
static PI_L1 int16_t  featVec[4*BUFFER_SZ], featMean[BUFFER_SZ];
static PI_L1 uint8_t  featStd[BUFFER_SZ];
static PI_L1 int8_t   classf_weights[5][BUFFER_SZ];
static PI_L1 int8_t   classf_bias[5];
static PI_L1 int32_t  classf_scores[5];

static PI_L1 uint32_t cycles_dma_copyin_s1     = 0;
static PI_L1 uint32_t cycles_dma_copyin_s2     = 0;
static PI_L1 uint32_t cycles_dma_copyin_s3     = 0;
static PI_L1 uint32_t cycles_dma_wait_s1  = 0;
static PI_L1 uint32_t cycles_dma_wait_s2  = 0;
static PI_L1 uint32_t cycles_dma_wait_s3  = 0;
static PI_L1 uint32_t cycles_compute_s1   = 0;
static PI_L1 uint32_t cycles_compute_s2   = 0;
static PI_L1 uint32_t cycles_compute_s3   = 0;
static PI_L1 uint32_t cycles_dma_copyout  = 0;

void hydra_forward_gap9(void *args) {
    
    #ifdef DETAILED_PROFILING
    pi_perf_reset();
    pi_perf_start();
    #endif

    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t dil;

    Hydra* hydra = (Hydra *) args;
    
    // Copy Step 1 Input Vector and Weights into L1
    pi_cl_dma_copy_t copy_L2_to_L1_inX, copy_L2_to_L1_inX_diff, copy_L2_to_L1_inW;
    pi_cl_dma_copy_t copy_L2_to_L1_cw[5];
    pi_cl_dma_copy_t copy_L2_to_L1_cb, copy_L2_to_L1_norm_avg, copy_L2_to_L1_norm_std;

    copy_L2_to_L1_inX.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inX.merge = 0;
    copy_L2_to_L1_inX.size  = (uint16_t) hydra->lenX*sizeof(RCKINT);
    copy_L2_to_L1_inX.id    = 0;
    copy_L2_to_L1_inX.ext   = (uint32_t) &(hydra->inX[0][hydra->lenXpad]);
    copy_L2_to_L1_inX.loc   = (uint32_t) &(inX[hydra->lenXpad]);

    copy_L2_to_L1_inX_diff.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inX_diff.merge = 0;
    copy_L2_to_L1_inX_diff.size  = (uint16_t) (hydra->lenX-1)*sizeof(RCKINT);
    copy_L2_to_L1_inX_diff.id    = 1;
    copy_L2_to_L1_inX_diff.ext   = (uint32_t) &(hydra->inX_diff[0][hydra->lenXpad]);
    copy_L2_to_L1_inX_diff.loc   = (uint32_t) &(inX_diff[hydra->lenXpad]);

    copy_L2_to_L1_inW.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inW.merge = 0;
    copy_L2_to_L1_inW.size  = (uint16_t) 2*hydra->lenW*hydra->K*hydra->H;
    copy_L2_to_L1_inW.id    = 2;
    copy_L2_to_L1_inW.ext   = (uint32_t) hydra->inW;
    copy_L2_to_L1_inW.loc   = (uint32_t) inW;

    pi_cl_dma_memcpy(&copy_L2_to_L1_inX);
    pi_cl_dma_memcpy(&copy_L2_to_L1_inX_diff);
    pi_cl_dma_memcpy(&copy_L2_to_L1_inW);

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_dma_copyin_s1 += pi_perf_read(PI_PERF_CYCLES);

    // Copy Step 2 Weights
    pi_perf_reset();
    pi_perf_start();
    #endif

    copy_L2_to_L1_norm_avg.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_norm_avg.merge = 0;
    copy_L2_to_L1_norm_avg.size  = (uint16_t) 2*hydra->len_feat_vec;
    copy_L2_to_L1_norm_avg.id    = 3;
    copy_L2_to_L1_norm_avg.ext   = (uint32_t) hydra->featMean;
    copy_L2_to_L1_norm_avg.loc   = (uint32_t) featMean;

    copy_L2_to_L1_norm_std.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_norm_std.merge = 0;
    copy_L2_to_L1_norm_std.size  = (uint16_t) hydra->len_feat_vec;
    copy_L2_to_L1_norm_std.id    = 4;
    copy_L2_to_L1_norm_std.ext   = (uint32_t) hydra->featStd;
    copy_L2_to_L1_norm_std.loc   = (uint32_t) featStd;

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_dma_copyin_s2 += pi_perf_read(PI_PERF_CYCLES);

    // Copy Step 3 weights
    pi_perf_reset();
    pi_perf_start();
    #endif

    for(int c=0; c < hydra->N_classes; c++) {
        copy_L2_to_L1_cw[c].dir   = PI_CL_DMA_DIR_EXT2LOC;
        copy_L2_to_L1_cw[c].merge = 0;
        copy_L2_to_L1_cw[c].size  = (uint16_t) hydra->len_feat_vec;
        copy_L2_to_L1_cw[c].id    = 5+c;
        copy_L2_to_L1_cw[c].ext   = (uint32_t) hydra->classf_weights[c];
        copy_L2_to_L1_cw[c].loc   = (uint32_t) classf_weights[c];
    }

    copy_L2_to_L1_cb.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_cb.merge = 0;
    copy_L2_to_L1_cb.size  = (uint16_t) hydra->N_classes;
    copy_L2_to_L1_cb.id    = 5+hydra->N_classes+1;
    copy_L2_to_L1_cb.ext   = (uint32_t) hydra->classf_bias;
    copy_L2_to_L1_cb.loc   = (uint32_t) classf_bias;

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_dma_copyin_s3 += pi_perf_read(PI_PERF_CYCLES);

    pi_perf_reset();
    pi_perf_start();
    #endif

    // Wait for step 1 transfers to be concluded, start step 2 transfers    
    pi_cl_dma_wait(&copy_L2_to_L1_inX);
    pi_cl_dma_wait(&copy_L2_to_L1_inX_diff);
    pi_cl_dma_wait(&copy_L2_to_L1_inW);
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cw[0]));
    pi_cl_dma_memcpy(&copy_L2_to_L1_norm_avg);
    pi_cl_dma_memcpy(&copy_L2_to_L1_norm_std);

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_dma_wait_s1 += pi_perf_read(PI_PERF_CYCLES);

    pi_perf_reset();
    pi_perf_start();
    #endif

    // ******* - Perform Step 1 Calculations - ******* //
    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for(int i=0; i < hydra->len_feat_vec; i+=1) {
            featVec[i] = 0;
        }
    }

    TeamForkArgs_T fork_args;
    fork_args.inW   = inW;
    fork_args.hydra = hydra;

    for (dil_idx = 0; dil_idx < hydra->N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < hydra->N_diff; diff_idx++) {
            for (chan = 0; chan < hydra->N_chan; chan++) {
                fork_args.dil = dil;
                fork_args.diff_idx = diff_idx;
                fork_args.featVec  = &(featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]);
                if(diff_idx == 0) {
                    fork_args.inX = inX;
                    pi_cl_team_fork(pi_cl_cluster_nb_cores(), hydra_convolve, &fork_args);
                }
                else {
                    fork_args.inX = inX_diff;
                    pi_cl_team_fork(pi_cl_cluster_nb_cores(), hydra_convolve, &fork_args);
                }
            }
        }
    }
    
    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_compute_s1 += pi_perf_read(PI_PERF_CYCLES);

    // ******* - Perform Step 2 Calculations - ******* //
    pi_perf_reset();
    pi_perf_start();
    #endif

    pi_cl_dma_wait(&copy_L2_to_L1_norm_avg);
    pi_cl_dma_wait(&copy_L2_to_L1_norm_std);
    pi_cl_dma_wait(&(copy_L2_to_L1_cw[0]));
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cw[1]));
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cw[2]));
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cw[3]));

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_dma_wait_s2 += pi_perf_read(PI_PERF_CYCLES);

    pi_perf_reset();
    pi_perf_start();
    #endif
    
    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for(uint32_t f=0; f < hydra->len_feat_vec; f++) {

            // Under-clip to zero, skip normalization if feature is zero
            featVec[f] = (featVec[f] < 0 ? 0 : featVec[f]);
            
            if(featVec[f] > 0) {
                featVec[f] = featVec[f] - featMean[f];

                if(featVec[f] < 0) {
                    // Most values are positive, but arithmetic shift of negative numbers 
                    // is not equivalent to division by powers of two, since it does not round to zero.
                    for(int s = 0; s < featStd[f]; s++) {
                        featVec[f] = (featVec[f]) / 2;
                    }
                }
                else {
                    featVec[f] = (featVec[f]) >> featStd[f];
                }
            }
        }
    }

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_compute_s2 += pi_perf_read(PI_PERF_CYCLES);

    // ******* - Perform Step 3 Calculations - ******* //
    pi_perf_reset();
    pi_perf_start();
    #endif

    pi_cl_dma_wait(&(copy_L2_to_L1_cw[1]));
    pi_cl_dma_wait(&(copy_L2_to_L1_cw[2]));
    pi_cl_dma_wait(&(copy_L2_to_L1_cw[3]));
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cw[4]));
    pi_cl_dma_memcpy(&(copy_L2_to_L1_cb));
    
    pi_cl_dma_wait(&copy_L2_to_L1_cb);
    pi_cl_dma_wait(&(copy_L2_to_L1_cw[4]));

    #ifdef DETAILED_PROFILING    
    pi_perf_stop();
    cycles_dma_wait_s3 += pi_perf_read(PI_PERF_CYCLES);

    pi_perf_reset();
    pi_perf_start();
    #endif

    v4s v_featVec;
    v4s v_classf_weights;

    for(uint8_t c=0; c < hydra->N_classes; c++) {
        classf_scores[c] = classf_bias[c];
    }

    #pragma omp parallel num_threads(5)
    {
        #pragma omp for
        for(uint8_t c=0; c < hydra->N_classes; c++) {
            #if defined (VECTORIZE)
            for(int f=0; f < hydra->len_feat_vec; f+=4) {
            #else
            for(int f=0; f < hydra->len_feat_vec; f+=1) {
            #endif
                #if defined (TARGET_GAP9) && defined (VECTORIZE)
                v_featVec                  = __builtin_pulp_pack4(featVec[f], featVec[f+1], featVec[f+2], featVec[f+3]);
                v_classf_weights           = *((v4s*) &classf_weights[c][f]);
                classf_scores[c]  = __builtin_pulp_sdotsp4(v_featVec, v_classf_weights, classf_scores[c]);
                #else
                classf_scores[c] += featVec[f] * classf_weights[c][f];
                #endif
            }
            hydra->classf_scores[c] = classf_scores[c];
        }
    }

    #ifdef DETAILED_PROFILING
    pi_perf_stop();
    cycles_compute_s3 += pi_perf_read(PI_PERF_CYCLES);
    #endif
}
#endif


int main()
{

    int VOLTAGE = 650;
    int FREQ_FC = 100;
    int FREQ_PE = 100;
    int FREQ_CL = 100;

    printf("Application started \n");

    // Voltage-Frequency settings
    uint32_t voltage = VOLTAGE;
    pi_freq_set(PI_FREQ_DOMAIN_FC,      FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH,  FREQ_PE*1000*1000);

    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,VOLTAGE);
    pi_time_wait_us(100000);
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,VOLTAGE);
    pi_time_wait_us(100000);

    // Setup perf counters
    float cycles_million = 0.0;
    uint32_t   cycles = 0, cycles_prev = 0;

    // Allocate space for buffer
    int16_t *values;
    int16_t sum=0;
    values = pi_l2_malloc(INPUT_LEN*INPUT_SZ);

    // Power Measurements
    pi_pad_function_set(GPIOs,1);
    pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
    pi_gpio_pin_write(GPIOs, 0);
    WRITE_GPIO(0);

    /************* SECTION 1: Setup of Readfs from File *************/
    static pi_fs_file_t *fd[2] = {NULL};

    static struct pi_device flash;
    static struct pi_default_flash_conf flash_conf;

    static struct pi_device fs;
    static struct pi_readfs_conf fs_conf;

    // Reads one line of input
    char flash_buffer[32];

    pi_default_flash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);
    if (pi_flash_open(&flash)) {
        printf("ERROR: Cannot open flash! Exiting...\n");
        pmsis_exit(-1);
    }

    printf("Mounting the filesystem \n");

    pi_readfs_conf_init(&fs_conf);
    fs_conf.fs.flash = &flash;

    // if using default layout, uncoment next line
    // fs_conf.fs.partition_name = "readfs_mram";
    // if using custom layout, comment next line
    // conf.fs.partition_name = "readfs_app";

    // Mounting the File System
    pi_open_from_conf(&fs, &fs_conf);
    if (pi_fs_mount(&fs)) {
        printf("ERROR: Cannot mount filesystem! Exiting...\n");
        pmsis_exit(-2);
    }
    printf("ReadFS mounted\n");
    /****************************************************************/

    // static struct pi_device ram;
    // static pi_default_ram_conf ram_conf;
    // pi_default_ram_conf_init(&ram_conf);
    // pi_open_from_conf(&ram, &ram_conf);
    // if (pi_ram_open(&ram)) {
    //     printf("ERROR: Cannot open ram! Exiting...\n");
    //     pmsis_exit(-3);
    // }


    printf("Data file path: '%s'!\n", STR(FILE_INPUT_DATA));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_WEIGHTS));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_CLASSF_W));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_CLASSF_B));


    /************* SECTION 4: Setup Cluster Task*************/
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;  
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |   
                       // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                       PI_CLUSTER_ICACHE_PREFETCH_ENABLE |      
                       // Enable the icache for all the cores
                       PI_CLUSTER_ICACHE_ENABLE;
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);


    /* Prepare cluster task and send it to cluster. */
    struct pi_cluster_task cl_task;
    /************* SECTION 2: Init Hydra model, load weights *************/
    // Initialize Hydra model
    Hydra *hydra;
    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       NUM_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);


    #ifdef PARALLELIZE
    // Zero padding to input vectors in L1
    for(int i=0; i < hydra->lenXpad; i++) {
        inX[i] = 0;
    }
    for(int i=hydra->lenXpad+hydra->lenX; i < 2*hydra->lenXpad+hydra->lenX+1; i++) {
        inX[i] = 0;
    }
    for(int i=0; i < hydra->lenXpad; i++) {
        inX_diff[i] = 0;
    }
    for(int i=hydra->lenXpad+hydra->lenX-1; i < 2*hydra->lenXpad+hydra->lenX; i++) {
        inX_diff[i] = 0;
    }
    printf("Hydra model successfully initialized!\n");
    pi_cluster_close(&cluster_dev);
    #endif

    
    // STEP A: Load RCK Weights
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_WEIGHTS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_WEIGHTS));
        return -2;
    }
    else {
        printf("Weights file opened successfully!\n");
    }

    pi_fs_read(fd[0], hydra->inW, 2*hydra->H*hydra->K*hydra->lenW);
    pi_fs_close(fd[0]);

    // STEP B: Load Sparse Scaler Means
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_SS_MEANS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_SS_MEANS));
        return -2;
    }

    for(int f=0; f < hydra->len_feat_vec; f++) {
        pi_fs_read(fd[0], flash_buffer, 2);
        hydra->featMean[f] = (flash_buffer[1] << 8 | flash_buffer[0]);
        //printf("Read from file (featMean) @[%d]: %d\n", f, hydra->featMean[f]);
    }
    pi_fs_close(fd[0]);

    // STEP C: Load Sparse Scaler STDS
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_SS_STDS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_SS_STDS));
        return -2;
    }
    pi_fs_read(fd[0], hydra->featStd, hydra->len_feat_vec);

    for(int f=0; f < hydra->len_feat_vec; f++) {
        //printf("Read from file (featStd) @[%d]: %d\n", f, hydra->featStd[f]);
    }
    pi_fs_close(fd[0]);

    // STEP D: Load Classifier Weights
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_CLASSF_W), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_CLASSF_W));
        return -2;
    }

    for(int c=0; c < hydra->N_classes; c++) {
        for(int f=0; f < hydra->len_feat_vec; f++) {
            pi_fs_read(fd[0], &(hydra->classf_weights[c][f]), 1);
        }
    }
    pi_fs_close(fd[0]);

    // STEP E: Load Classifier Biases
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_CLASSF_B), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_CLASSF_B));
        return -2;
    }

    for(int c=0; c < hydra->N_classes; c++) {
        pi_fs_read(fd[0], &(hydra->classf_bias[c]), 1);
        //printf("Read from file (classfBias) @[%d]: %d\n", c, hydra->classf_bias[c]);
    }
    pi_fs_close(fd[0]);

    /************* SECTION 3a: Opening Input Data file descriptor *************/
    // Open FD for Flash Section with input vector
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_DATA), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_DATA));
        return -2;
    }
    /**********************************************************************/
    

    /************* SECTION 3b: Setup of Host FS (for output dump) *************/
    /** HostFs dump to PC **/
    //pi_device_t host_fs;
    //struct pi_hostfs_conf hostfs_conf;
    //pi_hostfs_conf_init(&hostfs_conf);

    //pi_open_from_conf(&host_fs, &hostfs_conf);

    //if (pi_fs_mount(&host_fs))
    //{
    //    printf("Failed to mount host fs\n");
    //    return -3;
    //}
    //printf("Hostfs mounted\n");
    //
    //char *filename = "output.dat";
    //fd[1] = pi_fs_open(&host_fs, filename, PI_FS_FLAGS_WRITE);
    //if (fd[1] == NULL)
    //{
    //    printf("Failed to open file, OUTPUT\n");
    //    return -4;
    //}
    //printf("Output file opened\n");
    /**************************************************************************/


    /************* SECTION 5: Performing forward passes on test samples *************/
    for(int s=0; s < NUM_SAMPLES; s++) {
        /************* SECTION 4a: Reading the input data into mem *************/
        for(int i = 0; i < INPUT_LEN; i++) {
            if(sizeof(RCKINT) == 2) {
                pi_fs_read(fd[0], flash_buffer, 2);
                hydra->inX[0][i+hydra->lenXpad] = (flash_buffer[1] << 8 | flash_buffer[0]);
            }
            else {
                pi_fs_read(fd[0], flash_buffer, 1);
                hydra->inX[0][i+hydra->lenXpad] = flash_buffer[0];
            }
            //printf("Read from file @[%d]: %d\n", i, hydra->inX[0][i]);
        }
        for (int xi=0; xi < hydra->lenX-1; xi++) {
            hydra->inX_diff[0][xi+hydra->lenXpad] = hydra->inX[0][xi+1+hydra->lenXpad]-hydra->inX[0][xi+hydra->lenXpad];
        }
        /***********************************************************************/
        
        pi_open_from_conf(&cluster_dev, &cl_conf);
        if (pi_cluster_open(&cluster_dev))
        {
            printf("Cluster open failed !\n");
            pmsis_exit(-1);
        }

        pi_perf_reset();
        pi_perf_start();
	WRITE_GPIO(1);
        #ifdef PARALLELIZE
        pi_cluster_task(&cl_task, hydra_forward_gap9, hydra);
        pi_cluster_send_task_to_cl(&cluster_dev, &cl_task);
        pi_cluster_close(&cluster_dev);
        #else
        hydra_reset(hydra);
        hydra_forward(hydra);
        hydra_sparse_scale(hydra);
        hydra_classifier(hydra);
        #endif
        #ifndef DETAILED_PROFILING
        WRITE_GPIO(0);
	pi_perf_stop();


        /************* SECTION 4c: Collect benchmarks *************/
        cycles = pi_perf_read(PI_PERF_CYCLES);
        cycles_million  += (float)(cycles) / 1000000;
        if(s % 100 == 0) {
            printf("Processed %6d samples. # Cycles: %ld\n", s, cycles);
        }
        cycles_prev = cycles;
        #endif
        /**********************************************************/

        /************* SECTION 4d: Dumping output data to file *************/
        // Test output values by writing the input as it was
        //for(int i = 0; i < hydra->N_classes; i++) {
        //    flash_buffer[0] = (hydra->classf_scores[i]      ) & 0xFF;
        //    flash_buffer[1] = (hydra->classf_scores[i] >>  8) & 0xFF;
        //    flash_buffer[2] = (hydra->classf_scores[i] >> 16) & 0xFF;
        //    flash_buffer[3] = (hydra->classf_scores[i] >> 24) & 0xFF;
        //    pi_fs_write(fd[1], &hydra->classf_scores[i], sizeof(hydra->classf_scores[i]));
        //}
        /******************************************************************/

    }

    // Close FD for Flash Section with input vector
    //pi_fs_close(fd[1]);
    //pi_fs_unmount(&fs);

    #ifndef DETAILED_PROFILING
    float avg_cycles = (float)cycles_million / NUM_SAMPLES;
    float avg_inf_time_ms = (avg_cycles  * 1e6) / (FREQ_CL* 1e6) * 1e3;
    printf("Average inference time: %.3f ms. -- In raw perf cycles: ~= %.3f Million cycles\n", avg_inf_time_ms, avg_cycles);
    #else
    float avg_time_us[10];
    avg_time_us[0] = ((float)cycles_dma_copyin_s1 / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[1] = ((float)cycles_dma_copyin_s2 / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[2] = ((float)cycles_dma_copyin_s3 / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[3] = ((float)cycles_dma_wait_s1   / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[4] = ((float)cycles_dma_wait_s2   / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[5] = ((float)cycles_dma_wait_s3   / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[6] = ((float)cycles_compute_s1    / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[7] = ((float)cycles_compute_s2    / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[8] = ((float)cycles_compute_s3    / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    avg_time_us[9] = ((float)cycles_dma_copyout   / NUM_SAMPLES) / (FREQ_CL *1e6) * 1e6;
    
    float avg_inf_time_us = 0;
    for(int i=0; i < 10; i++) {
        avg_inf_time_us += avg_time_us[i];
    }
    printf("Total Inference Time: %.6f us\n", avg_inf_time_us);

    printf("DMA Copy-In Step 1: %04.6f us, (%03.2f %%)\n", avg_time_us[0], avg_time_us[0] / avg_inf_time_us * 100);
    printf("DMA Copy-In Step 2: %04.6f us, (%03.2f %%)\n", avg_time_us[1], avg_time_us[1] / avg_inf_time_us * 100);
    printf("DMA Copy-In Step 3: %04.6f us, (%03.2f %%)\n", avg_time_us[2], avg_time_us[2] / avg_inf_time_us * 100);
    printf("DMA Wait Step 1:    %04.6f us, (%03.2f %%)\n", avg_time_us[3], avg_time_us[3] / avg_inf_time_us * 100);
    printf("DMA Wait Step 2:    %04.6f us, (%03.2f %%)\n", avg_time_us[4], avg_time_us[4] / avg_inf_time_us * 100);
    printf("DMA Wait Step 3:    %04.6f us, (%03.2f %%)\n", avg_time_us[5], avg_time_us[5] / avg_inf_time_us * 100);
    printf("Compute  Step 1:    %04.6f us, (%03.2f %%)\n", avg_time_us[6], avg_time_us[6] / avg_inf_time_us * 100);
    printf("Compute  Step 2:    %04.6f us, (%03.2f %%)\n", avg_time_us[7], avg_time_us[7] / avg_inf_time_us * 100);
    printf("Compute  Step 3:    %04.6f us, (%03.2f %%)\n", avg_time_us[8], avg_time_us[8] / avg_inf_time_us * 100);
    printf("DMA Copy-Out:       %04.6f us, (%03.2f %%)\n", avg_time_us[9], avg_time_us[9] / avg_inf_time_us * 100);
    #endif

    return 0;
}
