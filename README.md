# Compton Slab Emission Model

This repository includes neutron star (NS) atmosphere look-up tables in .fits files for accretion-powered millisecond pulsars (AMPs). Stokes I and Q intensities are saved as a function of photon energy, emission angle, electron temperature of the slab, optical depth of the slab, and temperature of the seed blackbody photons coming from the NS surface. One example of how to read and interpolate the tables is provided in `interpolations.py` and it can be called in Python with
```
python driver_for_interpolations.py
```

These tables can be used when modeling the polarized X-ray pulses from AMPs (an example of that is shown in `polpulse.py`).

For further information and advice, we ask you to contact and refer to the authors of [Bobrikova et al. 2023, in prep.].
