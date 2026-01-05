extern "C"{
    //Bonus zadatak-iterativna analiza rizika unutar regiona

    #define MAX_TYPES 8  //Maksimalan broj tipova regiona

    //Struktura koja opisuje parametre jednog tipa regiona
    struct RegionProfile {
        float danger_threshold;  //Nivo koncentracije smatran opasnim
        float alpha;             //Faktor detekcije hotspota
    };

    //Konstantna memorija na GPU-u za profile regiona
    //Brza za citanje i ista za sve niti
    __constant__ RegionProfile d_region_profiles[MAX_TYPES];

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

    __global__ void region_hotspot_stats_kernel_bonus(int *danger_steps,
                                                   int H, int W,
                                                   int R, int C,
                                                   unsigned char *region_type,
                                                   float *region_means,
                                                   unsigned char *is_hotspot) {

        //Odredi region i preuzmi profil
        int region_row = blockIdx.y;
        int region_col = blockIdx.x;
        int region_id = region_row * (W / C) + region_col;

        //Preuzmi parametre specificne za region iz KONSTANTNE MEMORIJE
        unsigned char type = region_type[region_id];
        float alpha = d_region_profiles[type].alpha;

        //Gornji levi ugao regiona u globalnoj mrezi
        int region_start_i = region_row * R;
        int region_start_j = region_col * C;

        //Lokalna pozicija niti unutar regiona
        int local_i = threadIdx.y;
        int local_j = threadIdx.x;

        //Globalna pozicija niti
        int global_i = region_start_i + local_i;
        int global_j = region_start_j + local_j;

        //Ukupan broj celija u regionu
        int region_size = R * C;

        //Alokacija deljene memorije
        extern __shared__ char shared_mem[];

        int *s_danger = (int*) shared_mem;//Prva sekcija-danger_steps (int)
        float *s_reduce = (float*) &s_danger[region_size]; //Druga sekcija-redukcioni bafer (float)

        //Linearni indeks niti u regionu
        int tid = local_i * C + local_j;

        //Ucitaj podatke
        if (global_i < H && global_j < W) {
            s_danger[tid] = danger_steps[global_i * W + global_j];
        } else {
            s_danger[tid] = 0;
        }
        __syncthreads();

        //Izracunaj srednju vrednost
        s_reduce[tid] = (float) s_danger[tid];
        __syncthreads();

        //Paralelna redukcija-suma
        for (int stride = region_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        //Nit 0 racuna srednju vrednost
        __shared__ float s_mean;
        if (tid == 0) {
            s_mean = s_reduce[0] / region_size;
            region_means[region_id] = s_mean;
        }
        __syncthreads();

        //Izracunaj standardnu devijaciju
        float diff = (float)s_danger[tid] - s_mean;
        s_reduce[tid] = diff * diff;
        __syncthreads();

        //Redukcija sume kvadratnih razlika
        for (int stride = region_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        //Nit 0 racuna varijansu i standardnu devijaciju
        __shared__ float s_std_dev;
        if (tid == 0) {
            float variance = s_reduce[0] / region_size;
            s_std_dev = sqrtf(variance);
        }
        __syncthreads();

        //Identifikuj hotspotove (koristi alpha specifican za region!)
        float threshold = s_mean + alpha * s_std_dev;

        if (global_i < H && global_j < W) {
            int global_idx = global_i * W + global_j;
            if ((float)s_danger[tid] > threshold) {
                is_hotspot[global_idx] = 1; //hotspot
            } else {
                is_hotspot[global_idx] = 0; //nije hotspot
            }
        }
    }
}