extern "C"{
      //Zadatak 2-pracenje rizika kroz vreme
    __global__ void count_danger_kernel(float *conc, int *danger_count,
                                    int H, int W, float danger_threshold) {

        //Izracunaj globalnu poziciju niti                                
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        //Provera granica - da li smo unutar mreze
        if (i >= H || j >= W) return;

        //Linearni indeks za pristup 2D nizu: indeks = i * W + j
        int idx = i * W + j;

        //Ako je koncentracija u celiji veca ili jednaka pragu opasnosti
        //povecaj brojac za tu celiju
        if (conc[idx] >= danger_threshold) {
            danger_count[idx]++;
        }
    }
}