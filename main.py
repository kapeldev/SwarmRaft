import argparse
import os
import sys
import numpy as np
import subprocess


def run_verification():
    """
    Run Swarm_Drone_Algorithm_Verification.py (verification plots).
    """
    print("\n[INFO] Running swarm verification experiments...")
    subprocess.run([sys.executable, "Swarm_Drone_Algorithm_Verification.py"], check=True)


def run_scaling(N_runs=10000, max_f=8, distance_noise_std=0.5):
    """
    Run scaling experiment from Scaling_Swarm_Error.py.
    Saves results to npy files so they can be reused for plotting.
    """
    print("\n[INFO] Running scaling experiment...")
    from Scaling_Swarm_Error import scaling_experiment, plot_scaling_trend

    ns, avg_errors_verified, avg_errors_reported = scaling_experiment(
        N_runs=N_runs, max_f=max_f, distance_noise_std=distance_noise_std
    )

    # Save results
    np.save("ns.npy", np.array(ns))
    np.save("avg_errors.npy", np.array(avg_errors_verified))
    np.save("avg_errors_reported.npy", np.array(avg_errors_reported))

    print("[INFO] Results saved: ns.npy, avg_errors.npy, avg_errors_reported.npy")

    # Plot
    plot_scaling_trend(ns, avg_errors_verified, avg_errors_reported)


def run_plots_only():
    """
    Only generate plots from precomputed data, if available.
    If files are missing, fallback to running everything.
    """
    print("\n[INFO] Generating plots from existing data...")

    # Always show verification plot
    run_verification()

    # Then check scaling data
    if not (os.path.exists("ns.npy") and
            os.path.exists("avg_errors.npy") and
            os.path.exists("avg_errors_reported.npy")):
        print("[WARNING] Scaling data files not found! Running full scaling experiment...")
        run_scaling()
        return

    from Scaling_Swarm_Error import plot_scaling_trend

    ns = np.load("ns.npy")
    avg_errors = np.load("avg_errors.npy")
    avg_errors_reported = np.load("avg_errors_reported.npy")

    plot_scaling_trend(ns, avg_errors, avg_errors_reported)


def run_all():
    """
    Run both verification and scaling experiments.
    """
    run_verification()
    run_scaling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm verification experiments driver")
    parser.add_argument("--verification", action="store_true", help="Run only verification plots")
    parser.add_argument("--scaling", action="store_true", help="Run only scaling experiments")
    parser.add_argument("--all", action="store_true", help="Run both verification and scaling")
    parser.add_argument("--plots_only", action="store_true", help="Only generate plots from existing npy data")

    args = parser.parse_args()

    if args.verification:
        run_verification()
    elif args.scaling:
        run_scaling()
    elif args.all:
        run_all()
    elif args.plots_only:
        run_plots_only()
    else:
        parser.print_help()
