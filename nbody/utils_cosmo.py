import numpy as np, sys

# avoid division by zero
eps = sys.float_info.epsilon

def power_spectrum(k, A=1e4, n=0.9665):
    """
    Generate a simple power-law matter power spectrum: P(k) ~ k^n

    Parameters
    ----------
    k : ndarray
        Wavenumber magnitudes.
    A : float
        Normalization constant.
    n : float
        Spectral index (n=1 for scale-invariant).

    Returns
    -------
    Pk : ndarray
        Power spectrum values.
    """
    Pk = A * (k + eps)**n  
    return Pk

def displacement_field(N, box_size, power_spectrum_func):
    """
    Generate 3D Zel'dovich displacement field using a Gaussian random field.

    Parameters
    ----------
    N : int
        Number of particles per dimension.
    box_size : float
        Size of the simulation box (Mpc/h).
    power_spectrum_func : callable
        Function P(k) defining the power spectrum.

    Returns
    -------
    disp_field : ndarray of shape (N, N, N, 3)
        Displacement vectors in real space.
    """
    # Define Fourier space grid
    kx = np.fft.fftfreq(N, d=box_size/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=box_size/N) * 2 * np.pi
    kz = np.fft.fftfreq(N, d=box_size/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    k_squared = kx**2 + ky**2 + kz**2
    k_magnitude = np.sqrt(k_squared)

    # Power spectrum
    Pk = power_spectrum(k_magnitude)

    # Generate Gaussian random complex field with Hermitian symmetry
    noise = np.random.normal(size=(N, N, N)) + 1j * np.random.normal(size=(N, N, N))
    delta_k = noise * np.sqrt(Pk / 2)

    # Compute displacement field in Fourier space: i * k̂ * delta(k) / k
    disp_kx = 1j * kx * delta_k / (k_squared + 1e-10)
    disp_ky = 1j * ky * delta_k / (k_squared + 1e-10)
    disp_kz = 1j * kz * delta_k / (k_squared + 1e-10)

    # Inverse FFT to real space
    disp_x = np.real(np.fft.ifftn(disp_kx))
    disp_y = np.real(np.fft.ifftn(disp_ky))
    disp_z = np.real(np.fft.ifftn(disp_kz))

    # Combine into a displacement field array
    disp_field = np.stack([disp_x, disp_y, disp_z], axis=-1)  # shape (N, N, N, 3)
    
    return disp_field