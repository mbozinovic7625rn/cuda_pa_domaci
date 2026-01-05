extern "C" {
    //Zadatak 1-simulacija difuzije na GPU    
    __global__ void diffusion_step_kernel(float *conc_in, float *conc_out,
                                      int H, int W, float decay) {
        //Izracunaj globalnu poziciju niti
        int i = blockIdx.y * blockDim.y + threadIdx.y;  // red
        int j = blockIdx.x * blockDim.x + threadIdx.x;  // kolona

        //Provera granica - da li smo unutar mreze
        if (i >= H || j >= W) return;

        // Linearni indeks za pristup 2D nizu: indeks = i * W + j
        int idx = i * W + j;

        //DIFUZIJA-prosek sa 4 suseda
        float sum = conc_in[idx];  // centralna celija
        int count = 1;  // broj celija za prosek

        //GORE (i-1, j)
        if (i > 0) {
            sum += conc_in[(i-1) * W + j];
            count++;
        }

        //DOLE (i+1, j)
        if (i < H - 1) {
            sum += conc_in[(i+1) * W + j];
            count++;
        }

        //LEVO (i, j-1)
        if (j > 0) {
            sum += conc_in[i * W + (j-1)];
            count++;
        }

        //DESNO (i, j+1)
        if (j < W - 1) {
            sum += conc_in[i * W + (j+1)];
            count++;
        }

        //Izracunaj prosek
        float avg = sum / count;

        //Prirodno smanjenje zagadjenja
        avg = avg * (1.0f - decay);

        //Upisi u izlazni bafer
        conc_out[idx] = avg;
    }
}