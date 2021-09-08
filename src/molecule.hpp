#pragma once

#include <cstddef>

struct Atom {
    int channel;
    float x, y, z;
};

struct Molecule {
    size_t n_atoms = 0;
    Atom atoms[];
};

struct OutputSpec {
    float ext[2][3];
    float erf_inner_c, erf_outer_c;
    int W, H, D;
    int N;
};

struct GridSpec
{
    float width;
    float height;
    float depth;
    float variance;
    int grid_size; // Only supporting cubic grids for now
    int num_molecules;
    
};

inline Molecule* next_molecule(Molecule* a){
    size_t step = offsetof(Molecule, atoms);
    step += a->n_atoms * sizeof(a->atoms[0]);
    return (Molecule*)((char*)a + step);
}

void gaussian_erf_avx_fit_points(size_t n_atoms, float* points, OutputSpec* output, float* tensor);