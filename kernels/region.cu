extern "C"{
    //Zadatak 3-analiza regiona i lokalnih žarišta
    __global__ void region_hotspot_stats_kernel(int *danger_steps,
                                            int H, int W,
                                            int R, int C,
                                            float alpha,
                                            float *region_means,
                                            unsigned char *is_hotspot) {

        //Odredi koji region obradjuje ovaj blok
        int region_row = blockIdx.y;//indeks reda regiona
        int region_col = blockIdx.x;//indeks kolone regiona
        int region_id = region_row * (W / C) + region_col;//Jedinstveni ID regiona

        //Gornji levi ugao regiona u globalnoj mrezi
        int region_start_i = region_row * R;
        int region_start_j = region_col * C;

        //Lokalna pozicija niti unutar regiona
        int local_i = threadIdx.y;
        int local_j = threadIdx.x;

        //Globalna pozicija niti u mrezi
        int global_i = region_start_i + local_i;
        int global_j = region_start_j + local_j;

        //Velicina regiona
        int region_size = R * C;

        //Alociraj deljenu memoriju
        //Deklarise dinamicki alociranu shared memory kao niz bajtova
        //Velicina se prosledjuje pri pozivanju kernela kroz shared= parametar
        extern __shared__ char shared_mem[];

        int *s_danger = (int*) shared_mem;  //Prva sekcija-int
        float *s_reduce = (float*) &s_danger[region_size];  //Druga sekcija-float

        //Linearni indeks niti unutar regiona
        int tid = local_i * C + local_j;

        //Ucitaj podatke u deljenu memoriju
        if (global_i < H && global_j < W) {
            s_danger[tid] = danger_steps[global_i * W + global_j];//Validna celija-ucitaj vrednost
        } else {
            s_danger[tid] = 0;//Van mreze-postavi na nulu
        }

        __syncthreads();

        //Kopiraj u redukcioni bafer (konvertuj int u float)
        s_reduce[tid] = (float) s_danger[tid];
        __syncthreads();

        //Tree reduction pattern
        for (int stride = region_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        //Nit 0 izracunava srednju vrednost
        __shared__ float s_mean;
        if (tid == 0) {
            s_mean = s_reduce[tid] / region_size;
            region_means[region_id] = s_mean;
        }

        __syncthreads();

        //Izracunaj standardnu devijaciju
        //Formula-sigma = sqrt(mean((danger_steps - mean)^2))

        float diff = (float)s_danger[tid] - s_mean;
        s_reduce[tid] = diff * diff;
        __syncthreads();

        //Tree reduction za sumu kvadratnih razlika
        for (int stride = region_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        //Nit 0 izracunava varijansu i standardnu devijaciju
        __shared__ float s_std_dev;
        if (tid == 0) {
            float variance = s_reduce[0] / region_size;
            s_std_dev = sqrtf(variance);
        }
        __syncthreads();

        //Identifikuj hotspotove
        //Celija je hotspot ako-danger_steps[i,j] > mean + alpha * std_dev

        float threshold = s_mean + alpha * s_std_dev;

        if (global_i < H && global_j < W) {
            int global_idx = global_i * W + global_j;
            if ((float)s_danger[tid] > threshold) {
                is_hotspot[global_idx] = 1;
            } else {
                is_hotspot[global_idx] = 0;
            }
        }
    }
}