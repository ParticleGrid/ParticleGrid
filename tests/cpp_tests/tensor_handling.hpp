
#pragma once

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

static const char* fill = " ------++++#####";

inline void display_tensor(int W, int H, int D, float* tensor, int cols){
    float min_v = tensor[0];
    float max_v = tensor[0];
    for(int k = 0; k < D; k++){
        for(int j = 0; j < H; j++){
            for(int i = 0; i < W; i++){
                float v = tensor[k*W*H + j*W + i];
                min_v = std::min(v, min_v);
                max_v = std::max(v, max_v);
            }
        }
    }
    float total = 0;
    float corr = 0;
    if(min_v != max_v){
        for(int k = 0; k < D; k+=cols){
            for(int j = 0; j < H; j+=2){
                for(int k2 = 0; k2 + k < D && k2 < cols; k2++){
                    for(int i = 0; i < W; i++){
                        // printf("%.2e ", tensor[H*W*D/2 + i*W + j]);
                        float v1 = tensor[(k+k2)*W*H + j*W + i];
                        float v2 = tensor[(k+k2)*W*H + (j+1)*W + i];
                        float v = std::max(v1, v2);
                        float vm = std::min(v1, v2);
                        int idx = (int)(((v-min_v)/(max_v-min_v))*(strlen(fill)-1));
                        char c = fill[std::max(idx, 0)];
                        if(min_v < -0.005){
                            if(vm == min_v){
                                c = 'm';
                            }
                            else if(v == max_v){
                                c = 'M';
                            }
                            else if(abs(v) < 0.005*max_v){
                                c = '_';
                            }
                        }
                        printf("%c", c);
                        fflush(stdout);
                        // printf("%cc", c, c);
                        float y = (v1 + v2) - corr;
                        float t = total + y;
                        corr = (t - total) - y;
                        total = t;
                    }
                    printf(" | ");
                }
                printf("%d\n", j);
            }
            printf("\n");
        }
    }
    printf("total: %f \tmin: %f \tmax: %f\n", total, min_v, max_v);
}


inline void display_tensor_xy(int W, int H, int D, float* tensor, int cols){
    float yx[H][W] = {0};
    memset(yx, 0, sizeof(yx));
    float total = 0;
    float corr = 0;
    for(int k = 0; k < D; k++){
        for(int j = 0; j < H; j++){
            for(int i = 0; i < W; i++){
                float v = tensor[k*W*H + j*W + i];
                yx[j][i] += v;
                float y = v - corr;
                float t = total + y;
                corr = (t - total) - y;
                total = t;
            }
        }
    }
    float min_v = yx[0][0];
    float max_v = yx[0][0];
    for(int j = 0; j < H; j++){
        for(int i = 0; i < W; i++){
            float v = yx[j][i];
            min_v = std::min(v, min_v);
            max_v = std::max(v, max_v);
        }
    }
    for(int j = 0; j < H; j++){
        for(int i = 0; i < W; i++){
            float v = yx[j][i];
            int idx = (int)(((v-min_v)/(max_v-min_v))*(strlen(fill)-1));
            char c = fill[std::max(idx, 0)];
            printf("%c", c);
        }
        printf("|\n");
    }
    printf("total: %f \tmin: %f \tmax: %f\n", total, min_v, max_v);
}

inline void display_tensor_yz(int W, int H, int D, float* tensor, int cols){
    float yz[H][D] = {0};
    memset(yz, 0, sizeof(yz));
    float total = 0;
    float corr = 0;
    for(int k = 0; k < D; k++){
        for(int j = 0; j < H; j++){
            for(int i = 0; i < W; i++){
                float v = tensor[k*W*H + j*W + i];
                yz[j][k] += v;
                float y = v - corr;
                float t = total + y;
                corr = (t - total) - y;
                total = t;
            }
        }
    }
    float min_v = yz[0][0];
    float max_v = yz[0][0];
    for(int j = 0; j < H; j++){
        for(int k = 0; k < D; k++){
            float v = yz[j][k];
            min_v = std::min(v, min_v);
            max_v = std::max(v, max_v);
        }
    }
    for(int j = 0; j < H; j++){
        for(int k = 0; k < D; k++){
            float v = yz[j][k];
            int idx = (int)(((v-min_v)/(max_v-min_v))*(strlen(fill)-1));
            char c = fill[std::max(idx, 0)];
            printf("%c", c);
        }
        printf("|\n");
    }
    printf("total: %f \tmin: %f \tmax: %f\n", total, min_v, max_v);
}

inline void display_tensor_vals(int H, int W, int D, float* tensor){
    float min_v = tensor[0];
    float max_v = tensor[0];
    for(int k = 0; k < H; k++){
        for(int i = 0; i < W; i++){
            for(int j = 0; j < D; j++){
                float v = tensor[k*W*D + i*W + j];
                min_v = std::min(v, min_v);
                max_v = std::max(v, max_v);
            }
        }
    }
    int cols = 1;
    double total = 0;
    double corr = 0;
    for(int k = 0; k < H; k+=cols){
        for(int i = 0; i+2 <= W; i+=2){
            for(int k2 = 0; k2 + k < H && k2 < cols; k2++){
                for(int j = 0; j < D; j++){
                    // printf("%.2e ", tensor[H*W*D/2 + i*W + j]);
                    float v1 = tensor[(k+k2)*W*D + i*D + j];
                    float v2 = tensor[(k+k2)*W*D + (i+1)*D + j];
                    float v = std::max(v1, v2);
                    printf("%.1e ", v);
                    // printf("%cc", c, c);
                    double y = (v1 + v2) - corr;
                    double t = total + y;
                    corr = (t - total) - y;
                    total = t;
                }
                printf("| ");
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("total: %f\n", total);
}

inline void validate_tensor(int M, int N, int W, int H, int D, float* full_tensor, bool check_zero = false){
    size_t low_totals = 0;
    for(size_t m = 0; (int)m < M; m++){
        size_t mol_low_totals = 0;
        for(size_t n = 0; (int)n < N; n++){
            float* tensor = full_tensor + m*N*W*H*D + n*W*H*D;
            float min_v = tensor[0];
            float max_v = tensor[0];
            for(int k = 0; k < D; k++){
                for(int j = 0; j < H; j++){
                    for(int i = 0; i < W; i++){
                        float v = tensor[k*W*H + j*W + i];
                        min_v = std::min(v, min_v);
                        max_v = std::max(v, max_v);
                    }
                }
            }
            float total = 0;
            float corr = 0;
            if(min_v != max_v){
                for(int k = 0; k < D; k++){
                    for(int j = 0; j < H; j++){
                        for(int i = 0; i < W; i++){
                            float v = tensor[k*W*H + j*W + i];
                            float y = v - corr;
                            float t = total + y;
                            corr = (t - total) - y;
                            total = t;
                        }
                    }
                }
            }
            if(abs(round(total) - total) > 0.05){
                fprintf(stderr, "VALIDATOR WARNING: molecule %lu channel %lu total %f is non-integral\n", m, n, total);
            }
            if(abs(total) < 0.05){
                low_totals++;
                mol_low_totals++;
                if(check_zero){
                    fprintf(stderr, "VALIDATOR WARNING: molecule %lu channel %lu total %f is low\n", m, n, total);
                }
            }if(abs(total) > 500){
                fprintf(stderr, "VALIDATOR WARNING: molecule %lu channel %lu total %f is high\n", m, n, total);
            }
            if(min_v < 0){
                fprintf(stderr, "VALIDATOR WARNING: molecule %lu channel %lu min %f < 0\n", m, n, min_v);
            }
        }
        if(mol_low_totals >= (size_t)N){
            fprintf(stderr, "VALIDATOR WARNING: molecule %lu has %lu low channels\n", m, mol_low_totals);
        }
    }
    if((low_totals * 40) / N > (size_t)M*N){
        fprintf(stderr, "VALIDATOR WARNING: many totals are low: counted %lu\n", low_totals);
    }
}
