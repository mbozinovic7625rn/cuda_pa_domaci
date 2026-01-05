extern "C"{
    //Bonus zadatak-iterativna analiza rizika unutar regiona
    __global__ void count_danger_kernel_bonus(float *conc,
                                          int *danger_count,
                                          int H, int W,
                                          int R, int C,
                                          unsigned char *region_type) {

        //Izracunaj globalnu poziciju niti                                      
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        //Provera granica
        if (i >= H || j >= W) return;

        //Linearni indeks za pristup 2D nizu: indeks = i * W + j
        int idx = i * W + j;

        //Odredi kom regionu ova celija pripada
        int region_row = i / R;
        int region_col = j / C;
        int region_id = region_row * (W / C) + region_col;

        //Tip regiona (indeks u niz profila)
        unsigned char type = region_type[region_id];

        //Pristup konstantnoj memoriji-brzo
        float danger_threshold = d_region_profiles[type].danger_threshold;

        //Ako koncentracija prelazi prag specifican za region
        //povecaj brojac opasnih koraka za ovu celiju
        if (conc[idx] >= danger_threshold) {
            danger_count[idx]++;
        }
    }
}