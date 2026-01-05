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
}