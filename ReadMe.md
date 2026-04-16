Source Code for *TV Subgradient-Guided Multi-Source Fusion for Spectral Imaging in Dual-Camera CASSI Systems*

This version optimizes the running time based on the analysis detailed in Supplementary S4.

For instructions on acquiring all public datasets utilized in this work, please refer to the guidelines in the corresponding subfolders under `./datasets`.

1. **To generate noisy demosaiced ARAD RGB images:**
   Download the original ARAD hyperspectral images, clone the required code for the `externals` directory from GitHub, and run `generate_noisy_dataset.py`.
2. **Evaluation of the proposed methods:**
   Simply execute the corresponding code in `./TVDS` with the default settings.
   For improved performance (at the cost of increased runtime), you can set `use_net_init=True` and `rank=4`.
