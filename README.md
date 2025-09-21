# Swarm Verification and Error Scaling Experiments

This repository provides the code and artifacts for **swarm drone verification experiments** and **error scaling analysis**, as used in our research paper.  

It contains:
- **Swarm_Drone_Algorithm_Verification.py** → Runs verification scenarios and saves plots.  
- **Scaling_Swarm_Error.py** → Runs large-scale Monte Carlo experiments and scaling analysis.  
- **main.py** → Unified entry point to run experiments and plots.  

---

## 🔧 Installation

Clone the repository:
```bash
git https://github.com/kapeldev/SwarmRaft.git
cd SwarmRaft

```

Install dependencies (Python ≥ 3.8):

```
pip install -r requirements.txt
```


---

## ▶️ Usage

We provide a single driver script: main.py.

Run only verification experiments:
```
python main.py --verification
```


Run only scaling experiments (Monte Carlo, can take time):

```
python main.py --scaling
```

Run both (verification + scaling):
```
python main.py --all
```

Generate plots only (from saved .npy results if available):
```
python main.py --plots_only
```


If .npy files (ns.npy, avg_errors.npy, avg_errors_reported.npy) are missing, it will automatically re-run the experiments.

All figures are saved as high-resolution .pdf files in the repository folder.

---

## 📂 Output Artifacts

Swarm Verification.pdf → verification scenarios with spoofed vs recovered positions.

Error Scaling with Swarm Size.pdf → scaling trend of average error vs swarm size.

ns.npy, avg_errors.npy, avg_errors_reported.npy → saved experiment data (can be reused).

--- 

## ⏳ Runtime Notes

Verification plots run quickly (seconds).

Scaling experiments are computationally heavy (--scaling or --all).

A progress bar (tqdm) is shown.

Consider lowering N_runs in main.py or Scaling_Swarm_Error.py for quicker tests.

---

## 📜 Citation

If you use this code in your research, please cite our paper:

K. Dev, Y. Madhwal, S. Shevelo, P. Osinenko, and Y. Yanovich, “SwarmRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments,” *arXiv*, vol. arXiv:2508.00622, 2025. Submitted to *IEEE Internet of Things Journal*.

---

## 📄 `requirements.txt`


numpy>=1.22
scipy>=1.9
matplotlib>=3.6
tqdm>=4.65

