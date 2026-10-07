# FD001 data setup

Obtain the Turbofan Engine Degradation Simulation dataset using the download link in [NASA's official PCoE dataset repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), section 6.

Extract and arrange these three files:

```text
data/
├── train/train_FD001.txt
├── test/test_FD001.txt
└── rul/RUL_FD001.txt
```

The source has 26 whitespace-separated columns without headers: engine identifier, cycle, three operating settings, and 21 sensor readings. FD001 contains 100 training trajectories and 100 truncated test trajectories. The RUL file gives one label for the last observed cycle of each test engine, ordered by engine number.

Raw dataset files are ignored by Git. Check the source's terms before redistributing datasets or using them commercially. The model bundle and recorded results identify the exact local inputs with SHA-256 fingerprints in `results/run_metadata.json`.

Dataset citation: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
