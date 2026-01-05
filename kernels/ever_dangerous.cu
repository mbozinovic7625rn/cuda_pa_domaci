extern "C"{
    //Zadatak 2-pracenje rizika kroz vreme
    __global__ void count_ever_dangerous_kernel(int *danger_count, int H, int W,
                                            int *danger_counter) {

        //Izracunaj globalnu poziciju niti  
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        //Provera granica - da li smo unutar mreze
        if (i >= H || j >= W) return;

        //Linearni indeks za pristup 2D nizu: indeks = i * W + j
        int idx = i * W + j;

        //Ako je brojac veci od nule znaci da je ova celija makar jednom
        //imala opasnu koncentraciju
        if (danger_count[idx] > 0) {
            //Vise niti moze istovremeno povecavati globalni brojac
            //zato koristimo atomicAdd da izbegnemo race condition
            atomicAdd(danger_counter, 1);
        }
    }
}