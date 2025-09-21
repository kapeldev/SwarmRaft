import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from tqdm import tqdm


class SwarmVerification:
    def __init__(self, n_drones=5, seed=43, distance_noise_std=0.0):
        """
        Initialize the swarm verification system.

        Parameters:
        n_drones (int): Number of drones in the swarm
        seed (int): Random seed for reproducibility
        distance_noise_std (float): Standard deviation of distance measurement noise
        """
        np.random.seed(seed)
        self.n_drones = n_drones
        self.distance_noise_std = distance_noise_std
        self.true_positions = None
        self.reported_positions = None
        self.distance_matrix = None
        self.faulty_indices = None
        self.num_faulty = None

    def generate_swarm(self, area_size=100, num_faulty=1, spoof_offset=(30, -20, 25)):
        """
        Generate swarm with multiple faulty drones.

        Parameters:
        area_size (float): Size of 3D area
        num_faulty (int): Number of faulty drones
        spoof_offset (tuple or list of tuples): Offsets for each faulty drone

        Returns:
        true_positions, reported_positions, faulty_indices
        """
        if self.n_drones < 2 * num_faulty + 1:
            raise ValueError(f"Cannot have {num_faulty} faulty drones for n={self.n_drones}")

        # Generate true positions
        self.true_positions = np.random.uniform(0, area_size, size=(self.n_drones, 3))

        # Select multiple faulty drones
        self.faulty_indices = np.random.choice(self.n_drones, num_faulty, replace=False)

        self.reported_positions = self.true_positions.copy()
        for idx, f_idx in enumerate(self.faulty_indices):
            # Generate a random 3D offset within a reasonable range, e.g., [-30, 30] meters
            random_offset = np.random.uniform(-30, 30, size=3)
            offset = spoof_offset if isinstance(spoof_offset[0], (int, float)) else spoof_offset[idx]
            self.reported_positions[f_idx] += random_offset

        return self.true_positions, self.reported_positions, self.faulty_indices

    def generate_distance_matrix(self):
        """
        Generate distance matrix with measurement noise.

        Returns:
        ndarray: Distance matrix between all drones
        """
        self.distance_matrix = np.zeros((self.n_drones, self.n_drones))
        for i in range(self.n_drones):
            for j in range(self.n_drones):
                if i != j:
                    true_distance = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                    noisy_distance = true_distance + np.random.normal(0, self.distance_noise_std)
                    self.distance_matrix[i, j] = true_distance
        return self.distance_matrix

    def verify_all_drones(self, base_epsilon=5.0, tol=0.01, max_iter=3, rel_thresh=0.15):
        """
        Verify all drones using voting + multilateration with only non-faulty neighbors.

        Returns:
        dict: verified_positions, deviations, rel_deviations, fault_flags, adaptive_epsilon
        """
        verified_positions = np.zeros_like(self.reported_positions)
        deviations = np.zeros(self.n_drones)
        rel_deviations = np.zeros(self.n_drones)
        fault_flags = np.zeros(self.n_drones, dtype=bool)

        avg_distance = np.mean(self.distance_matrix[self.distance_matrix > 0])
        adaptive_epsilon = base_epsilon + 2 * self.distance_noise_std + 0.02 * avg_distance

        # Stage 1: Voting for all nodes
        for A in range(self.n_drones):
            neighbors = [i for i in range(self.n_drones) if i != A]
            votes = 0
            for B in neighbors:
                dist_from_positions = np.linalg.norm(self.reported_positions[A] - self.reported_positions[B])
                dist_reported = self.distance_matrix[A, B]
                votes += 1 if abs(dist_from_positions - dist_reported) < tol else -1
            fault_flags[A] = votes < 0
            verified_positions[A] = self.reported_positions[A]

        # Stage 2: Multilateration for flagged nodes using only non-faulty neighbors
        for A in range(self.n_drones):
            if not fault_flags[A]:
                continue

            neighbors = [i for i in range(self.n_drones) if i != A and not fault_flags[i]]
            if len(neighbors) < 3:
                neighbors = [i for i in range(self.n_drones) if i != A]  # fallback

            centroid = np.mean(self.reported_positions[neighbors], axis=0)
            p_est = self.reported_positions[A].copy()
            # p_est = centroid
            for _ in range(max_iter):
                def residual(p):
                    return np.array([np.linalg.norm(p - self.reported_positions[i]) - self.distance_matrix[i, A]
                                     for i in neighbors])

                res = least_squares(residual, p_est, loss="soft_l1", f_scale=5.0)
                new_p_est = res.x
                if np.linalg.norm(new_p_est - p_est) < 1e-3:
                    break
                p_est = new_p_est

            x_approx = p_est
            delta = np.linalg.norm(self.reported_positions[A] - x_approx)
            deviations[A] = delta
            mean_neighbor_dist = np.mean([self.distance_matrix[A, j] for j in neighbors])
            rel_deviations[A] = delta / (mean_neighbor_dist + 1e-6)

            # Final fault check
            adaptive_epsilon = 3
            if (delta > adaptive_epsilon):
                fault_flags[A] = True
                verified_positions[A] = x_approx
            else:
                fault_flags[A] = False
                verified_positions[A] = self.reported_positions[A]

        return {
            "verified_positions": verified_positions,
            "deviations": deviations,
            "rel_deviations": rel_deviations,
            "fault_flags": fault_flags,
            "adaptive_epsilon": adaptive_epsilon
        }

    def run_full_verification(self, area_size=100, num_faulty=2, spoof_offset=(30, -20, 25)):
        """
        Run complete verification pipeline for multiple faulty drones.

        Returns:
        dict: results including positions, verified positions, flags, errors
        """
        # Generate swarm and spoofed reports
        self.generate_swarm(area_size, num_faulty, spoof_offset)
        self.generate_distance_matrix()
        verification_results = self.verify_all_drones()

        # Compute error between verified and true positions (after correction)
        verified_positions = verification_results["verified_positions"]
        verified_errors = {}
        for f_idx in self.faulty_indices:
            err = np.linalg.norm(verified_positions[f_idx] - self.true_positions[f_idx])
            verified_errors[int(f_idx)] = float(err)

        # Compute error between reported and true positions (before correction)
        reported_errors = {}
        for f_idx in self.faulty_indices:
            err = np.linalg.norm(self.reported_positions[f_idx] - self.true_positions[f_idx])
            reported_errors[int(f_idx)] = float(err)

        results = {
            "true_positions": self.true_positions,
            "reported_positions": self.reported_positions,
            "distance_matrix": self.distance_matrix,
            "faulty_indices": self.faulty_indices,
            "verification": verification_results,
            "spoof_offset_applied": spoof_offset,
            "errors_verified": verified_errors,
            "errors_reported": reported_errors,
            "avg_error_verified": float(np.mean(list(verified_errors.values())))
            if verified_errors else 0.0,
            "avg_error_reported": float(np.mean(list(reported_errors.values())))
            if reported_errors else 0.0,
        }
        return results


def plot_scenario_2d(ax, n_drones, num_faulty, verifier_seed=42):
    """
    Run verification for a given scenario and plot 2D simplified visualization
    on the given subplot axis.
    """
    verifier = SwarmVerification(n_drones=n_drones, distance_noise_std=0.5, seed=verifier_seed)
    results = verifier.run_full_verification(num_faulty=num_faulty)

    true_positions = results["true_positions"][:, :2]      # project to 2D
    reported_positions = results["reported_positions"][:, :2]
    verified_positions = results["verification"]["verified_positions"][:, :2]
    faulty_indices = results["faulty_indices"]

    # True positions (green X)
    ax.scatter(true_positions[:, 0], true_positions[:, 1],
               c="green", marker="x", s=80, label="True (X)")

    # Spoofed (red ▲)
    ax.scatter(reported_positions[:, 0], reported_positions[:, 1],
               c="red", marker="^", s=80, label="Spoofed (▲)")

    # Verified (yellow ★)
    ax.scatter(verified_positions[:, 0], verified_positions[:, 1],
               c="gold", marker="*", s=120, label="Recovered (★)")

    # Arrows from true → spoofed
    for idx, f_idx in enumerate(faulty_indices, start=1):
        ax.arrow(true_positions[f_idx, 0], true_positions[f_idx, 1],
                 reported_positions[f_idx, 0] - true_positions[f_idx, 0],
                 reported_positions[f_idx, 1] - true_positions[f_idx, 1],
                 color="gray", linestyle="--", alpha=0.7,
                 width=0.0, head_width=0.7, length_includes_head=True)

        # Label the arrow
        mid_x = (true_positions[f_idx, 0] + reported_positions[f_idx, 0]) / 2
        mid_y = (true_positions[f_idx, 1] + reported_positions[f_idx, 1]) / 2
        ax.text(mid_x, mid_y, f"$e_{idx}$", color="black", fontsize=8, fontweight="bold")

    ax.set_title(f"Swarm Verification (n={n_drones}, f={num_faulty})", fontsize=10)
    ax.set_xlabel("Projected X (m)")
    ax.set_ylabel("Projected Y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis("equal")


def scaling_experiment(N_runs=1000, max_f=8, distance_noise_std=0.5):
    """
    Run scaling experiment: average recovery error vs swarm size n=2f+1.
    Also tracks mean error of reported (spoofed) positions.

    Parameters:
    - N_runs: number of Monte Carlo runs per (n,f) pair
    - max_f: maximum number of faulty drones to test
    - distance_noise_std: standard deviation of distance measurement noise

    Returns:
    - ns: list of swarm sizes tested
    - avg_errors_verified: average error (faulty drones only) after verification
    - avg_errors_reported: average error (faulty drones only) before verification
    """
    ns = []
    avg_errors_verified = []
    avg_errors_reported = []

    for f in tqdm(range(1, max_f + 1), desc="Scaling experiment", unit="f"):
        n = 2 * f + 1
        verifier = SwarmVerification(n_drones=n, distance_noise_std=distance_noise_std)

        errors_verified_runs = []
        errors_reported_runs = []
        for _ in range(N_runs):
            results = verifier.run_full_verification(num_faulty=f)
            errors_verified_runs.append(results["avg_error_verified"])
            errors_reported_runs.append(results["avg_error_reported"])

        ns.append(n)
        avg_errors_verified.append(np.mean(errors_verified_runs))
        avg_errors_reported.append(np.mean(errors_reported_runs))

        print(f"(n={n}, f={f}) → avg verified error = {np.mean(errors_verified_runs):.2f} m, "
              f"avg reported error = {np.mean(errors_reported_runs):.2f} m")

    return ns, avg_errors_verified, avg_errors_reported

def plot_scaling_trend(ns, avg_errors_verified, avg_errors_reported):
    """
    Plot scaling trend of mean errors vs swarm size (log-scaled y-axis)
    for both verified and reported positions, with annotations.

    Parameters:
    - ns: list of swarm sizes
    - avg_errors_verified: mean error after verification
    - avg_errors_reported: mean error of reported/spoofed positions
    """
    fs = [int((n - 1) / 2) for n in ns]

    plt.figure(figsize=(7, 5))

    # Plot reported (spoofed) errors
    plt.plot(ns, avg_errors_reported, marker="s", linestyle="--", color="red", 
             label="Raw / Spoofed Position Error", markersize=8, linewidth=2)

    # Plot verified errors
    plt.plot(ns, avg_errors_verified, marker="o", linestyle="--", color="orange", 
             label="Recovered Position Error (SwarmRaft)", markersize=8, linewidth=2)

    # Annotate each point with (n,f)
    for n, f, err in zip(ns, fs, avg_errors_verified):
        plt.text(n, err * 1.1, f"({n},{f})", ha="center", fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))

    # X-axis: only odd swarm sizes
    x_ticks = [n for n in range(min(ns), max(ns) + 1) if n % 2 == 1]
    plt.xticks(x_ticks, fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("Swarm size $n$", fontsize=14, fontweight='bold')
    plt.ylabel("Mean error (m, log scale)", fontsize=14, fontweight='bold')
    plt.title("Error Scaling with Swarm Size", 
              fontsize=16, fontweight="bold", pad=20)

    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[0.35, 0.5, 0.7, 1, 2, 5], numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))

    # Turn off top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: limit y-range
    plt.ylim(0.2, max(max(avg_errors_verified), max(avg_errors_reported)) * 1.4)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6, which="both")

    # Legend with larger font
    plt.legend(fontsize=12, framealpha=0.9, loc='best')
    plt.tight_layout()
    plt.savefig("Error Scaling with Swarm Size.pdf", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run scaling with defaults
    ns, avg_errors_verified, avg_errors_reported = scaling_experiment(N_runs=10000, max_f=8, distance_noise_std=0.5)
    np.save("avg_errors.npy", np.array(avg_errors_verified))
    np.save("avg_errors_reported.npy", np.array(avg_errors_reported))
    np.save("ns.npy", np.array(ns))
    plot_scaling_trend(ns, avg_errors_verified, avg_errors_reported)
