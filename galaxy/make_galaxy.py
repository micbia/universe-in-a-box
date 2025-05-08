import math
import numpy as np


class Galaxy:
    def __init__(self, kpc_size=17.5, b=-0.3, num_arms=4, rot_spacing=None, trail_delay=0.1, fuzz=1.5, nr_core_part=3000, nr_arm_part=1000, nr_haze_part=2000):
        """
        Galaxy
        ------
        Create a spiral galaxy model with core stars, spiral arms, and haze/gas particles.

        Parameters:
        - size (float): Radius of the galactic disc (in kpc units).
        - b (float): Spiral tightness (logarithmic spiral coefficient).
        - num_arms (int): Number of spiral arms (will double into leading/trailing).
        - fuzz (float): Positional noise for stars in the spiral arms.
        - haze_density (float): Multiplier for haze particle count.
        """
        self.r = kpc_size / 0.05 # 1 unit = 0.05 kpc
        self.b = b
        self.num_arms = num_arms
        self.trail_delay = trail_delay
        if(rot_spacing == None):
            self.rot_spacing = 2. / self.num_arms
        else:
            self.rot_spacing = rot_spacing
        self.fuzz = fuzz

        # Number of particles in the spiral arms, the core (inner and outter), and haze (inner outter).  The core and the haze has double the selected value due to inner and outter layers
        self.nr_arm_part = nr_arm_part
        self.nr_core_part = nr_core_part
        self.nr_haze_part = nr_haze_part

        self.arms_info = self._generate_arms_info()


    def _generate_arms_info(self):
        """Generate rotation offsets for a given number of spiral arms."""
        arms = []
        for i in range(self.num_arms):
            # spacing in π units
            rot = i * self.rot_spacing 
            
            # leading arm part
            arms.append((rot, self.fuzz))    
            
            # trailing arm pair
            arms.append((rot-self.trail_delay, self.fuzz))  
        return arms


    def _spherical_coords(self, num_pts, radius):
        """
            spherical_coords
            ----------------
            Generates uniformly distributed points within a sphere.

            Parameters:
            - num_pts (int): Number of points to generate.
            - radius (float): Radius of the spherical volume.

            Returns:
            - np.ndarray: Array of shape (num_pts, 3) with 3D coordinates.
        """
        position_list = np.zeros((num_pts, 3))
        
        for i in range(num_pts):
            coords = np.random.normal(loc=0, scale=1, size=3)
            coords *= radius
        
            # Reduce z range for matplotlib default z-scale
            coords[2] *= 0.02

            position_list[i] = coords
        
        return position_list


    def build_spiral_arms(self):
        """
            build_spiral_arms
            -----------------
            Builds leading and trailing spiral arms of a galaxy based on spiral parameters.

            Parameters:
            - b (float): Spiral tightness (logarithmic spiral coefficient).
            - arms_info (list of tuple): Each tuple contains (rot_fac, fuz_fac) 
            for a spiral arm.

            Returns:
            - tuple of np.ndarray: (leading_arms, trailing_arms) coordinate arrays.
        """
        leading_arms = []
        trailing_arms = []
        for i, self.arm_info in enumerate(self.arms_info):
            
            # Rotation offset for spiral arm placement
            rot_fac = self.arm_info[0]
            
            # Randomization factor to add position fuzziness
            fuz_fac = self.arm_info[1]

            # Scalable initial amount to shift locations
            fuzz = int(0.030 * abs(self.r))  

            # particle in the spiral
            spiral_arm = np.zeros((self.nr_arm_part, 3))
            for j in range(0, self.nr_arm_part):

                theta = math.radians(j)
                
                x = self.r * math.exp(self.b*theta) * math.cos(theta - math.pi * rot_fac) - np.random.randint(-fuzz, fuzz) * fuz_fac
                y = self.r * math.exp(self.b*theta) * math.sin(theta - math.pi * rot_fac) - np.random.randint(-fuzz, fuzz) * fuz_fac
                z = np.random.uniform(-1/3, 1/3)
                
                spiral_arm[j] = [x, y, z]

            if i % 2 != 0:
                leading_arms.extend(spiral_arm)
            else:
                trailing_arms.extend(spiral_arm)

        return np.array(leading_arms), np.array(trailing_arms)


    def build_core(self):
        """
            build_core_stars
            ----------------
            Generates stars for the galactic core using a spherical distribution.

            Parameters:
            - scale_factor (float): Overall galaxy scale for setting core radius.
            - num_rim_stars (int): Number of outer core stars to generate.

            Returns:
            - np.ndarray: Combined 3D coordinates of inner and outer core stars.
        """
        core_radius = self.r / 15
        outer_stars = self._spherical_coords(self.nr_core_part, core_radius)
        inner_stars = self._spherical_coords(int(self.nr_core_part), core_radius/2.5)

        return np.vstack((outer_stars, inner_stars))


    def build_haze(self, r_mult, z_mult):
        """
            haze
            ----
            Creates a distribution of diffuse haze (gas/dust) particles within a disc-shaped volume.

            Parameters:
            - scale_factor (float): Base galaxy radius to scale distribution.
            - r_mult (float): Radius scaling multiplier (controls spread).
            - z_mult (float): Z-axis scaling (thickness of the haze disc).
            - density (float): Particle density multiplier.

            Returns:
            - np.ndarray: Array of shape (N, 3) with haze particle positions.
        """
        #nr_hazes = int(scale_factor * density)
        haze_coords = np.zeros((self.nr_haze_part, 3))

        for i in range(0, self.nr_haze_part):
            n = np.random.uniform(0, 1)

            theta = np.random.uniform(0, 2 * math.pi)
            
            x = round(math.sqrt(n) * math.cos(theta) * self.r) / r_mult
            y = round(math.sqrt(n) * math.sin(theta) * self.r) / r_mult
            z = np.random.uniform(-1, 1) * z_mult
            
            haze_coords[i] = [x, y, z]

        return haze_coords


    def build_galaxy(self, r_in=2, z_in=0.5, r_out=1, z_out=0.3):
        """ Build the full galaxy structure: spiral arms, core, and haze. """
        
        leading_arm, trailing_arm = self.build_spiral_arms()
        
        core_parts = self.build_core()
        
        inner_haze = self.build_haze(r_mult=r_in, z_mult=z_in)
        outer_haze = self.build_haze(r_mult=r_out, z_mult=z_out)

        return {"leading_arm": leading_arm, 
                "trailing_arm": trailing_arm, 
                "core_parts": core_parts, 
                "inner_haze": inner_haze, 
                "outer_haze": outer_haze}
    

    def get_particles(self, r_in=2, z_in=0.5, r_out=1, z_out=0.3):
        """ Build the full galaxy structure: spiral arms, core, and haze. """
        
        leading_arm, trailing_arm = self.build_spiral_arms()
        arm_particles = np.vstack((leading_arm, trailing_arm))

        core_particles = self.build_core()
        
        inner_haze = self.build_haze(r_mult=r_in, z_mult=z_in)
        outer_haze = self.build_haze(r_mult=r_out, z_mult=z_out)
        haze_particles = np.vstack((inner_haze, outer_haze))

        return {"arm": arm_particles, 
                "core": core_particles, 
                "haze": haze_particles}