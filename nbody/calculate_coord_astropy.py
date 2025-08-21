import numpy as np, h5py
import astropy.units as u

from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

# Set high-precision ephemeris (requires jplephem)
solar_system_ephemeris.set('jpl')

# J2000 epoch
#t_string = 'J2000'
t_string = '2025-08-01T12:00:00'
t = Time(t_string, scale='tdb')

# List of planets
planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

planet_masses = {
    'Sun': 1.9885e30,        # kg
    'Mercury': 3.3011e23,
    'Venus': 4.8675e24,
    'Earth': 5.97237e24,
    'Mars': 6.4171e23,
    'Jupiter': 1.8982e27,
    'Saturn': 5.6834e26,
    'Uranus': 8.6810e25,
    'Neptune': 1.02413e26,
    'Pluto': 1.303e22
}

print("HELIOCENTRIC COORDINATES at %s (ICRS frame)\n" %t_string)
sun_pos, sun_vel = get_body_barycentric_posvel('sun', t)

solar_system = {}
for planet in planets:
    # Get barycentric position and velocity of planet and Sun
    planet_pos, planet_vel = get_body_barycentric_posvel(planet, t)

    # Heliocentric = planet - sun
    heliocentric_pos = planet_pos - sun_pos
    heliocentric_vel = planet_vel - sun_vel

    # Convert to units
    x = heliocentric_pos.x.to('AU')
    y = heliocentric_pos.y.to('AU')
    z = heliocentric_pos.z.to('AU')

    vx = heliocentric_vel.x.to('AU/day')
    vy = heliocentric_vel.y.to('AU/day')
    vz = heliocentric_vel.z.to('AU/day')

    print(f"{planet.capitalize()}:")
    print(f"  Position (AU)   : x = {x:.6f}, y = {y:.6f}, z = {z:.6f}")
    print(f"  Velocity (km/s) : vx = {vx:.6f}, vy = {vy:.6f}, vz = {vz:.6f}\n")

    solar_system[planet] = {
    'position': np.array([x.value, y.value, z.value]),
    'velocity': np.array([vx.value, vy.value, vz.value]),
    'mass': (planet_masses[planet]*u.kg).to('Msun').value}

# Saving to an HDF5 file
with h5py.File("heliocentric_%s.hdf5" %t_string, "w") as f:
    for planet in planets:
        grp = f.create_group(planet)
        grp.create_dataset("position", data=solar_system[planet]["position"])
        grp.create_dataset("velocity", data=solar_system[planet]["velocity"])
        grp.create_dataset("mass", data=solar_system[planet]['mass'])

print("Solar System data saved to 'heliocentric_%s.hdf5'" %t_string)