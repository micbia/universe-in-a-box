import numpy as np
import h5py

# Solar mass, AU, and conversion constants remain the same

# Dictionary for the Solar System with planets and major moons
solar_system = {
    "Sun": {
        "position": np.array([0.0, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 0.0, 0.0]),  # AU/day
        "mass": 1.0  # Solar masses
    },
    "Mercury": {
        "position": np.array([0.387, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 4.79e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 1.651e-7  # Solar masses
    },
    "Venus": {
        "position": np.array([0.723, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 3.5e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 2.447e-6  # Solar masses
    },
    "Earth": {
        "position": np.array([1.0, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 2.978e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 3.003e-6,  # Solar masses
        "Moon": {
            "position": np.array([1.00257, 0.0, 0.0]),  # AU, distance from Earth to Moon in AU
            "velocity": np.array([0.0, 1.023 * 5.775e-7, 0.0]),  # AU/day
            "mass": 3.694e-8  # Solar masses
        }
    },
    "Mars": {
        "position": np.array([1.524, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 2.41e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 3.213e-7  # Solar masses
    },
    "Jupiter": {
        "position": np.array([5.202, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 1.31e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 9.545e-4,  # Solar masses
        "moons": {
            "Io": {
                "position": np.array([5.202 + 4.217e-5, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 17.334 * 5.775e-7, 0.0]),  # AU/day
                "mass": 4.86e-8  # Solar masses
            },
            "Europa": {
                "position": np.array([5.202 + 6.713e-5, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 13.74 * 5.775e-7, 0.0]),  # AU/day
                "mass": 2.53e-8  # Solar masses
            },
            "Ganymede": {
                "position": np.array([5.202 + 1.0704e-4, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 10.88 * 5.775e-7, 0.0]),  # AU/day
                "mass": 7.8e-8  # Solar masses
            },
            "Callisto": {
                "position": np.array([5.202 + 1.8827e-4, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 8.204 * 5.775e-7, 0.0]),  # AU/day
                "mass": 5.59e-8  # Solar masses
            }
        }
    },
    "Saturn": {
        "position": np.array([9.537, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 9.68e3 * 5.775e-7, 0.0]),  # AU/day
        "mass": 2.857e-4,  # Solar masses
        "moons": {
            "Titan": {
                "position": np.array([9.537 + 8.167e-4, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 5.57 * 5.775e-7, 0.0]),  # AU/day
                "mass": 2.37e-7  # Solar masses
            }
        }
    },
    "Uranus": {
        "position": np.array([19.191, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 6.8e3 * 5.775e-7, 0.0]),  # AU/day
        "mass": 4.366e-5,  # Solar masses
        "moons": {
            "Titania": {
                "position": np.array([19.191 + 2.761e-4, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 3.64 * 5.775e-7, 0.0]),  # AU/day
                "mass": 3.4e-8  # Solar masses
            }
        }
    },
    "Neptune": {
        "position": np.array([30.068, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 5.43e3 * 5.775e-7, 0.0]),  # AU/day
        "mass": 5.15e-5,  # Solar masses
        "moons": {
            "Triton": {
                "position": np.array([30.068 + 2.371e-4, 0.0, 0.0]),  # AU
                "velocity": np.array([0.0, 4.39 * 5.775e-7, 0.0]),  # AU/day
                "mass": 2.14e-7  # Solar masses
            }
        }
    },
    "Pluto": {
        "position": np.array([39.482, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 4.74e3 * 5.775e-7, 0.0]),  # AU/day
        "mass": 6.55e-9  # Solar masses
    },
    "Ceres": {
        "position": np.array([2.767, 0.0, 0.0]),  # AU
        "velocity": np.array([0.0, 1.73e4 * 5.775e-7, 0.0]),  # AU/day
        "mass": 4.7e-10  # Solar masses
    },
    "Halley's Comet": {
        "position": np.array([0.586, 0.0, 0.0]),  # perihelion distance in AU
        "velocity": np.array([0.0, 5.452e4 * 5.775e-7, 0.0]),  # perihelion velocity in AU/day
        "mass": 1.106e-16  # Solar masses
    }
}

# Saving to an HDF5 file
with h5py.File("solar_system.hdf5", "w") as f:
    for body, data in solar_system.items():
        grp = f.create_group(body)
        grp.create_dataset("position", data=data["position"])
        grp.create_dataset("velocity", data=data["velocity"])
        grp.create_dataset("mass", data=data["mass"])
        
        # Save moons if available
        if "moons" in data:
            moons_grp = grp.create_group("moons")
            for moon, moon_data in data["moons"].items():
                moon_grp = moons_grp.create_group(moon)
                moon_grp.create_dataset("position", data=moon_data["position"])
                moon_grp.create_dataset("velocity", data=moon_data["velocity"])
                moon_grp.create_dataset("mass", data=moon_data["mass"])
        elif "Moon" in data:
            # Save Earth's moon specifically
            moon_grp = grp.create_group("Moon")
            moon_grp.create_dataset("position", data=data["Moon"]["position"])
            moon_grp.create_dataset("velocity", data=data["Moon"]["velocity"])
            moon_grp.create_dataset("mass", data=data["Moon"]["mass"])

print("Solar System data saved to 'solar_system.hdf5'")