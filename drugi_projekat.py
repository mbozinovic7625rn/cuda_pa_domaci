import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit 
from pycuda.compiler import SourceModule
import time

#CUDA Kernel

def load_kernel(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()+ "\n"
    
cuda_code = SourceModule(
    load_kernel("kernels/sources.cu")
    + load_kernel("kernels/region.cu")
    + load_kernel("kernels/ever_dangerous.cu")
    + load_kernel("kernels/diffusion.cu")
    + load_kernel("kernels/danger.cu")
    + load_kernel("kernels/bonus_profle.cu")
    + load_kernel("kernels/bonus_region.cu")
    + load_kernel("kernels/bonus_danger.cu"),
    arch="sm_89",  
    options=[
        "--use_fast_math",
        "-allow-unsupported-compiler"
    ]
)

#Preuzimanje CUDA kernel funkcija 
diffusion_step_kernel = cuda_code.get_function("diffusion_step_kernel")
apply_sources_kernel = cuda_code.get_function("apply_sources_kernel")
count_danger_kernel = cuda_code.get_function("count_danger_kernel")
count_ever_dangerous_kernel = cuda_code.get_function("count_ever_dangerous_kernel")
region_hotspot_stats_kernel = cuda_code.get_function("region_hotspot_stats_kernel")
count_danger_kernel_bonus = cuda_code.get_function("count_danger_kernel_bonus")
region_hotspot_stats_kernel_bonus = cuda_code.get_function("region_hotspot_stats_kernel_bonus")
d_region_profiles_ptr = cuda_code.get_global("d_region_profiles")[0]

def initialize_grid(H, W):
    #Inicijalizuj matricu koncentracije sa nulama (nema pocetnog zagadjenja)
    conc = np.zeros((H, W), dtype=np.float32)

    #Inicijalizuj matricu tipova celija (0=normalna, 1=izvor, 2=filter)
    cell_type = np.zeros((H, W), dtype=np.uint8)

    #Postavi izvore zagadjenja (tip 1) na strateskim pozicijama
    if H >= 10 and W >= 10:
        cell_type[H // 4, W // 4] = 1  #Gornji levi kvadrant
        cell_type[H // 4, 3 * W // 4] = 1  #Gornji desni kvadrant
        cell_type[H // 2, W // 2] = 1  #Centar
    else:
        cell_type[H // 2, W // 2] = 1  #Za male mreze samo centar

    #Postavi filtere/cistace (tip 2) na ivicama
    if H >= 20 and W >= 20:
        cell_type[H // 8, W // 2] = 2  #Gore
        cell_type[7 * H // 8, W // 2] = 2  #Dole
        cell_type[H // 2, W // 8] = 2  #Levo
        cell_type[H // 2, 7 * W // 8] = 2  #Desno

    return conc, cell_type


def run_simulation_task1(H, W, T, decay, source_value, absorb_value):
    #Inicijalizuj mrezu sa izvorima i filterima
    conc, cell_type = initialize_grid(H, W)

    #Alociraj GPU memoriju za koncentracije (potrebna 2 bafera za double buffering)
    conc_gpu = cuda.mem_alloc(conc.nbytes)  #Trenutna koncentracija
    conc_gpu_tmp = cuda.mem_alloc(conc.nbytes)  #Privremeni bafer
    cell_type_gpu = cuda.mem_alloc(cell_type.nbytes)  #Tipovi celija

    #Kopiraj pocetne podatke sa CPU na GPU
    cuda.memcpy_htod(conc_gpu, conc)
    cuda.memcpy_htod(cell_type_gpu, cell_type)

    BLOCK_SIZE = 16  #16x16 = 256 niti po bloku
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = (
        (W + BLOCK_SIZE - 1) // BLOCK_SIZE,  #Broj blokova u x pravcu
        (H + BLOCK_SIZE - 1) // BLOCK_SIZE,  #Broj blokova u y pravcu
        1,
    )

    start = cuda.Event()
    end = cuda.Event()
    start.record()

    for t in range(T):
        #Korak 1-Difuzija sa opadanjem
        diffusion_step_kernel(
            conc_gpu,  #Ulazna koncentracija
            conc_gpu_tmp,  #Izlazna koncentracija
            np.int32(H),
            np.int32(W),
            np.float32(decay),
            block=block,
            grid=grid,
        )

        #Zameni bafere (izlaz postaje ulaz za sledeći korak)
        conc_gpu, conc_gpu_tmp = conc_gpu_tmp, conc_gpu

        apply_sources_kernel(
            conc_gpu,  #Modifikuje se in-place
            cell_type_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(source_value),
            np.float32(absorb_value),
            block=block,
            grid=grid,
        )

    #Sacekaj da GPU zavrsi sve kernele
    end.record()
    end.synchronize()
    gpu_time = start.time_till(end) * 1e-3 

    #Kopiraj rezultat sa GPU-a nazad na CPU
    cuda.memcpy_dtoh(conc, conc_gpu)

    return conc, gpu_time


def benchmark_task1():
    T = 100  #Broj vremenskih koraka
    decay = 0.01  #Faktor opadanja (1% po koraku)
    source_value = 1.0  #Vrednost izvora zagađenja
    absorb_value = 0.1  #Koliko filter apsorbuje

    #Razlicite velicine mreza za testiranje
    test_sizes = [(10, 10), (100, 100), (1000, 1000), (1000, 2000), (3000, 2000)]

    print("\nZADATAK 1 - Benchmark")
    print(f"{'Grid':<12} {'Cells':<12} {'Time(s)':<10} {'Cells/sec':<12}")

    #svaka velicina mreze
    for H, W in test_sizes:
        conc, gpu_time = run_simulation_task1(
            H, W, T, decay, source_value, absorb_value
        )
        cells_per_sec = (H * W * T) / gpu_time  #Propusnost (cells * timesteps / sec)
        print(f"{H}x{W:<9} {H*W:<12,} {gpu_time:<10.4f} {cells_per_sec:<12,.0f}")


def run_simulation_task2(H, W, T, decay, source_value, absorb_value, danger_threshold):
    #Inicijalizuj mrezu
    conc, cell_type = initialize_grid(H, W)
    #Matrica za brojanje koliko koraka je svaka celija bila opasna
    danger_count = np.zeros((H, W), dtype=np.int32)

    #Alociraj GPU memoriju
    conc_gpu = cuda.mem_alloc(conc.nbytes)
    conc_gpu_tmp = cuda.mem_alloc(conc.nbytes)
    cell_type_gpu = cuda.mem_alloc(cell_type.nbytes)
    danger_count_gpu = cuda.mem_alloc(danger_count.nbytes) 

    #Kopiraj podatke na GPU
    cuda.memcpy_htod(conc_gpu, conc)
    cuda.memcpy_htod(cell_type_gpu, cell_type)
    cuda.memcpy_htod(danger_count_gpu, danger_count)

    #CUDA konfiguracija
    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)

    start_time = time.time()

    for t in range(T):
        #Difuzija sa opadanjem
        diffusion_step_kernel(
            conc_gpu,
            conc_gpu_tmp,
            np.int32(H),
            np.int32(W),
            np.float32(decay),
            block=block,
            grid=grid,
        )
        conc_gpu, conc_gpu_tmp = conc_gpu_tmp, conc_gpu

        apply_sources_kernel(
            conc_gpu,
            cell_type_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(source_value),
            np.float32(absorb_value),
            block=block,
            grid=grid,
        )

        count_danger_kernel(
            conc_gpu,
            danger_count_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(danger_threshold),
            block=block,
            grid=grid,
        )

    #Alociraj GPU memoriju za rezultat (jedan int32)
    ever_dangerous_count_gpu = cuda.mem_alloc(4)
    cuda.memset_d32(ever_dangerous_count_gpu, 0, 1)  #Inicijalizuj na 0

    count_ever_dangerous_kernel(
        danger_count_gpu,
        np.int32(H),
        np.int32(W),
        ever_dangerous_count_gpu,
        block=block,
        grid=grid,
    )

    #Sinhronizuj i izmeri vreme
    cuda.Context.synchronize()
    gpu_time = time.time() - start_time

    #Kopiraj rezultate sa GPU-a
    cuda.memcpy_dtoh(conc, conc_gpu)
    cuda.memcpy_dtoh(danger_count, danger_count_gpu)

    ever_dangerous_count = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(ever_dangerous_count, ever_dangerous_count_gpu)

    return conc, danger_count, ever_dangerous_count[0], gpu_time


def task2():
    print("\nZADATAK 2 - Pracenje rizika")

    H, W = 100, 100  #Dimenzije mreze
    T = 200  #Broj vremenskih koraka
    decay = 0.01
    source_value = 1.0
    absorb_value = 0.1
    danger_threshold = 0.3  #Prag iznad kojeg je celija opasna

    conc, danger_count, ever_dangerous, gpu_time = run_simulation_task2(
        H, W, T, decay, source_value, absorb_value, danger_threshold
    )

    print(f"Vreme: {gpu_time:.4f}s")
    print(
        f"Opasne celije: {ever_dangerous:,} / {H*W:,} ({100*ever_dangerous/(H*W):.1f}%)"
    )
    print(f"Maks vreme opasnosti: {np.max(danger_count)} koraka")


def run_simulation_task3(
    H,
    W,
    R,
    C,
    T,
    decay,
    source_value,
    absorb_value,
    danger_threshold,
    alpha,
):
    #Validacija-grid mora biti deljiv sa velicinom regiona
    if H % R != 0 or W % C != 0:
        raise ValueError(f"Grid ({H}x{W}) mora biti deljiv sa ({R}x{C})")

    #Inicijalizacija (isto kao Task 2)
    conc, cell_type = initialize_grid(H, W)
    danger_count = np.zeros((H, W), dtype=np.int32)

    #Alociraj GPU memoriju
    conc_gpu = cuda.mem_alloc(conc.nbytes)
    conc_gpu_tmp = cuda.mem_alloc(conc.nbytes)
    cell_type_gpu = cuda.mem_alloc(cell_type.nbytes)
    danger_count_gpu = cuda.mem_alloc(danger_count.nbytes)

    #Kopiraj na GPU
    cuda.memcpy_htod(conc_gpu, conc)
    cuda.memcpy_htod(cell_type_gpu, cell_type)
    cuda.memcpy_htod(danger_count_gpu, danger_count)

    #CUDA konfiguracija za simulaciju
    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)

    start_time = time.time()

    for t in range(T):
        diffusion_step_kernel(
            conc_gpu,
            conc_gpu_tmp,
            np.int32(H),
            np.int32(W),
            np.float32(decay),
            block=block,
            grid=grid,
        )
        conc_gpu, conc_gpu_tmp = conc_gpu_tmp, conc_gpu

        apply_sources_kernel(
            conc_gpu,
            cell_type_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(source_value),
            np.float32(absorb_value),
            block=block,
            grid=grid,
        )

        count_danger_kernel(
            conc_gpu,
            danger_count_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(danger_threshold),
            block=block,
            grid=grid,
        )

    #Pripremi izlazne matrice
    num_regions = (H // R) * (W // C)  #Ukupan broj regiona
    region_means = np.zeros(
        num_regions, dtype=np.float32
    )  #Srednja vrednost za svaki region
    is_hotspot = np.zeros((H, W), dtype=np.uint8)  #Mapa hotspotova

    #Alociraj GPU memoriju za regionalne rezultate
    region_means_gpu = cuda.mem_alloc(region_means.nbytes)
    is_hotspot_gpu = cuda.mem_alloc(is_hotspot.nbytes)

    #1 blok = 1 region sa RxC niti
    region_block = (C, R, 1)  # RxC niti po bloku
    region_grid = (W // C, H // R, 1)  # Broj regiona u x i y pravcu

    #Izracunaj velicinu shared memorije (int + float za svaku celiju regiona)
    region_size = R * C
    shared_mem_size = region_size * 4 + region_size * 4  # 4bajta po int/float

    #Pokreni kernel za regionalnu analizu (tree reduction + statistika + hotspotovi)
    region_hotspot_stats_kernel(
        danger_count_gpu,
        np.int32(H),
        np.int32(W),
        np.int32(R),
        np.int32(C),
        np.float32(alpha),  #Faktor za hotspot threshold
        region_means_gpu,
        is_hotspot_gpu,
        block=region_block,
        grid=region_grid,
        shared=shared_mem_size,  #Dinamicki alocirana shared memory
    )

    #Sinhronizuj i izmeri ukupno vreme
    cuda.Context.synchronize()
    gpu_time = time.time() - start_time

    #Kopiraj sve rezultate sa GPU-a
    cuda.memcpy_dtoh(conc, conc_gpu)
    cuda.memcpy_dtoh(danger_count, danger_count_gpu)
    cuda.memcpy_dtoh(region_means, region_means_gpu)
    cuda.memcpy_dtoh(is_hotspot, is_hotspot_gpu)

    return conc, danger_count, is_hotspot, region_means, gpu_time


def task3():
    print("\nZADATAK 3 - Regionalna analiza hotspotova")

    H, W = 128, 128  #Dimenzije mreze (mora biti deljivo sa R, C)
    R, C = 16, 16  #Dimenzije regiona (svaki region je 16x16 ćelija)
    T = 200
    decay = 0.01
    source_value = 1.0
    absorb_value = 0.1
    danger_threshold = 0.3
    alpha = 1.0  #Hotspot ako je danger_time > mean + 1*std_dev

    conc, danger_count, is_hotspot, region_means, gpu_time = run_simulation_task3(
        H, W, R, C, T, decay, source_value, absorb_value, danger_threshold, alpha
    )

    #Prebroj i prikazi hotspotove
    num_hotspots = np.sum(is_hotspot == 1)
    print(f"Vreme: {gpu_time:.4f}s")
    print(f"Hotspotovi: {num_hotspots:,} / {H*W:,} ({100*num_hotspots/(H*W):.1f}%)")
    print(
        f"Srednja opasnost regiona - Min: {np.min(region_means):.2f}, Maks: {np.max(region_means):.2f}"
    )


def run_simulation_bonus(
    H,
    W,
    R,
    C,
    T,
    decay,
    source_value,
    absorb_value,
    region_profiles,  #Lista profila: svaki tip regiona ima svoj danger_threshold i alpha
    region_type_map,  #2D mapa tipova regiona
):
    if H % R != 0 or W % C != 0:
        raise ValueError(f"Grid ({H}x{W}) mora biti deljiv sa ({R}x{C})")

    num_regions_y = H // R
    num_regions_x = W // C

    if region_type_map.shape != (num_regions_y, num_regions_x):
        raise ValueError(f"region_type_map oblik ne odgovara")

    #Kopiramo profile regiona u konstantnu memoriju GPU-a
    #Kreiranje numpy structured array za profile
    dtype = np.dtype([("danger_threshold", np.float32), ("alpha", np.float32)])
    profiles_array = np.zeros(8, dtype=dtype)  #MAX_TYPES = 8

    #Popuni array sa profilima
    for i, profile in enumerate(region_profiles):
        profiles_array[i]["danger_threshold"] = profile["danger_threshold"]
        profiles_array[i]["alpha"] = profile["alpha"]

    #Kopiraj u KONSTANTNU GPU memoriju (brz pristup za sve niti)
    cuda.memcpy_htod(d_region_profiles_ptr, profiles_array)

    #Priprema podataka
    region_type_flat = region_type_map.flatten().astype(np.uint8)  #Flatten 2D u 1D
    conc, cell_type = initialize_grid(H, W)
    danger_count = np.zeros((H, W), dtype=np.int32)

    #Alociraj GPU memoriju
    conc_gpu = cuda.mem_alloc(conc.nbytes)
    conc_gpu_tmp = cuda.mem_alloc(conc.nbytes)
    cell_type_gpu = cuda.mem_alloc(cell_type.nbytes)
    danger_count_gpu = cuda.mem_alloc(danger_count.nbytes)
    region_type_gpu = cuda.mem_alloc(region_type_flat.nbytes)  #tip regiona

    #Kopiraj na GPU
    cuda.memcpy_htod(conc_gpu, conc)
    cuda.memcpy_htod(cell_type_gpu, cell_type)
    cuda.memcpy_htod(danger_count_gpu, danger_count)
    cuda.memcpy_htod(region_type_gpu, region_type_flat)

    #CUDA konfiguracija
    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)

    start_time = time.time()

    for t in range(T):
        #Difuzija
        diffusion_step_kernel(
            conc_gpu,
            conc_gpu_tmp,
            np.int32(H),
            np.int32(W),
            np.float32(decay),
            block=block,
            grid=grid,
        )
        conc_gpu, conc_gpu_tmp = conc_gpu_tmp, conc_gpu

        #Primeni izvore i filtere
        apply_sources_kernel(
            conc_gpu,
            cell_type_gpu,
            np.int32(H),
            np.int32(W),
            np.float32(source_value),
            np.float32(absorb_value),
            block=block,
            grid=grid,
        )

        count_danger_kernel_bonus(
            conc_gpu,
            danger_count_gpu,
            np.int32(H),
            np.int32(W),
            np.int32(R),
            np.int32(C),
            region_type_gpu,  #Kernel čita tip regiona
            block=block,
            grid=grid,
        )

    #Regionalna analiza hotspotova sa parametrima po regionima
    num_regions = (H // R) * (W // C)
    region_means = np.zeros(num_regions, dtype=np.float32)
    is_hotspot = np.zeros((H, W), dtype=np.uint8)

    region_means_gpu = cuda.mem_alloc(region_means.nbytes)
    is_hotspot_gpu = cuda.mem_alloc(is_hotspot.nbytes)

    #CUDA konfiguracija za regionalni kernel
    region_block = (C, R, 1)
    region_grid = (W // C, H // R, 1)
    region_size = R * C
    shared_mem_size = region_size * 4 + region_size * 4

    #Koristimo alpha parametre iz konstantne memorije za svaki tip regiona
    region_hotspot_stats_kernel_bonus(
        danger_count_gpu,
        np.int32(H),
        np.int32(W),
        np.int32(R),
        np.int32(C),
        region_type_gpu,  #Kernel čita alpha iz konstantne memorije
        region_means_gpu,
        is_hotspot_gpu,
        block=region_block,
        grid=region_grid,
        shared=shared_mem_size,
    )

    #Sinhronizuj i izmeri vreme
    cuda.Context.synchronize()
    gpu_time = time.time() - start_time

    #Kopiraj rezultate nazad na CPU
    cuda.memcpy_dtoh(conc, conc_gpu)
    cuda.memcpy_dtoh(danger_count, danger_count_gpu)
    cuda.memcpy_dtoh(region_means, region_means_gpu)
    cuda.memcpy_dtoh(is_hotspot, is_hotspot_gpu)

    return conc, danger_count, is_hotspot, region_means, gpu_time


def bonus_task():
    print("\nBONUS ZADATAK - Regionalni parametri")

    H, W = 128, 128  #Dimenzije mreze
    R, C = 16, 16  #Dimenzije regiona
    T = 200
    decay = 0.01
    source_value = 1.0
    absorb_value = 0.1

    #3 tipa regiona sa različitim parametrima
    region_profiles = [
        {"danger_threshold": 0.2, "alpha": 0.8},  # Tip 0:Stambeni - strogi prag
        {"danger_threshold": 0.5, "alpha": 2.0},  # Tip 1:Industrijski - blaži prag
        {"danger_threshold": 0.15, "alpha": 0.5},  # Tip 2:Parkovi - najstroži prag
    ]

    #Kreiraj mapu tipova regiona (8x8 regiona)
    num_regions_y = H // R  #8 regiona vertikalno
    num_regions_x = W // C  #8 regiona horizontalno
    region_type_map = np.zeros((num_regions_y, num_regions_x), dtype=np.uint8)

    #Postavi tipove regiona u obrazac
    region_type_map[3:5, 3:5] = 1  #Centar = Industrijski (tip 1)
    region_type_map[0, :] = 2  #Gornja ivica = Parkovi (tip 2)
    region_type_map[-1, :] = 2  #Donja ivica = Parkovi (tip 2)
    region_type_map[:, 0] = 2  #Leva ivica = Parkovi (tip 2)
    region_type_map[:, -1] = 2  #Desna ivica = Parkovi (tip 2)
    #Ostalo ostaje tip 0 (Stambeni)

    conc, danger_count, is_hotspot, region_means, gpu_time = run_simulation_bonus(
        H,
        W,
        R,
        C,
        T,
        decay,
        source_value,
        absorb_value,
        region_profiles,
        region_type_map,
    )

    num_hotspots = np.sum(is_hotspot == 1)
    print(f"Vreme: {gpu_time:.4f}s")
    print(f"Hotspotovi: {num_hotspots:,} / {H*W:,} ({100*num_hotspots/(H*W):.1f}%)")


if __name__ == "__main__":
    benchmark_task1()
    task2()
    task3()
    bonus_task()
