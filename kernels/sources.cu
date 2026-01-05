extern "C"{
    //Zadatak 1-simulacija difuzije na GPU 
    __global__ void apply_sources_kernel(float *conc, unsigned char *cell_type,
                                     int H, int W, float source_value,
                                     float absorb_value) {

        //Izracunaj globalnu poziciju niti                                   
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        //Provera granica - da li smo unutar mreze
        if (i >= H || j >= W) return;

        //Linearni indeks za pristup 2D nizu: indeks = i * W + j
        int idx = i * W + j;

        //Tip celije (0 = normalna, 1 = izvor, 2 = apsorber)
        unsigned char type = cell_type[idx];

        //Ako je celija izvor postavi koncentraciju na fiksnu vrednost
        if (type == 1) {
            conc[idx] = source_value;
        }
        //Ako je celija apsorber smanji koncentraciju za absorb_value
        //ali ne dozvoli da padne ispod nule
        else if (type == 2) {
            conc[idx] = fmaxf(0.0f, conc[idx] - absorb_value);
        }
    }
}