import torch as th
import numpy as np
from pathlib import Path
from lcil.certification.config import LyapunovCertificationConfig
from lcil.certification.models import LyapunovCoreVerifier
from lcil.certification.abcrown_region_certifier import CompleteABCrownCertifier
from lcil.utils import IntegrationMethod
from examples.cartpole import load_policy_model, load_lyapunov_model, load_mpc_config, CartpoleDynamics

lyapunov_dir = Path('results/cartpole/20260828_142007/run_001/lyapunov/20260917_104712/run_001')
policy_dir = lyapunov_dir.parent.parent.parent
device = th.device('cuda')
policy_model = load_policy_model(policy_dir, device)
lyap_model = load_lyapunov_model(lyapunov_dir, device)
mpc_cfg = load_mpc_config(policy_dir)
dyn_model = CartpoleDynamics(dt=mpc_cfg.dt, method=IntegrationMethod.EXPLICIT_EULER, abcrown_compatible_ops=True).to(device)

d = np.load('results/cartpole/20260828_142007/run_001/lyapunov/20260917_104712/run_001/certification/evaluated_rhos/rho_0.007331.npz')
failed = th.as_tensor(d['failed_regions'], device=device)
cfg = LyapunovCertificationConfig.load(lyapunov_dir / 'certification' / 'certification_config.json')
certifier = CompleteABCrownCertifier(policy_model, lyap_model, dyn_model, config=cfg, device=device)

rho = float(d['rho'])
print(f"Testing failed regions at rho={rho}:")
for idx in range(len(failed)):
    reg = failed[idx]
    res = certifier.verify_region(reg, rho=rho)
    print(f"Region {idx}: status={res.status}")
    if res.counterexample_found:
        print(f"  Bounds:\n    lower={reg[0].cpu().numpy()}\n    upper={reg[1].cpu().numpy()}")
        # Let's inspect verifier directly
        verifier = certifier.verifier
        # Verifier output: [decrease_residual, V_x, u_min_res, u_max_res, x_next_min_res, x_next_max_res]
        # Let's run PGD attack or sample in this box to find violating points
        lb, ub = reg[0], reg[1]
        # random samples
        x_samples = lb + th.rand(100000, 4, device=device) * (ub - lb)
        with th.no_grad():
            y = verifier(x_samples)
            # y[:, 0] is decrease: V(x_next) - V(x) + kappa*|x|
            # y[:, 1] is V(x)
            dec_residual = y[:, 0]
            v_val = y[:, 1]
            
            # Sublevel violation: V(x) <= rho and dec_residual >= 0
            in_sublevel = v_val <= rho
            bad_decrease = dec_residual >= 0
            cex_mask = in_sublevel & bad_decrease
            
            print(f"  In 100,000 random samples in this box:")
            print(f"    Min V(x): {v_val.min().item():.6f}, Max V(x): {v_val.max().item():.6f}")
            print(f"    Samples in sublevel (V <= rho): {in_sublevel.sum().item()} ({in_sublevel.float().mean()*100:.2f}%)")
            print(f"    Samples with bad decrease (dV >= -kappa|x|): {bad_decrease.sum().item()} ({bad_decrease.float().mean()*100:.2f}%)")
            print(f"    Counterexample samples (V <= rho AND bad decrease): {cex_mask.sum().item()}")
            
            if cex_mask.any():
                cex_idx = cex_mask.nonzero()[0].item()
                x_bad = x_samples[cex_idx]
                print(f"    Example CEX x: {x_bad.cpu().numpy()}")
                print(f"    V(x_bad): {v_val[cex_idx].item():.6f} (vs rho={rho:.6f})")
                print(f"    Decrease residual: {dec_residual[cex_idx].item():.6f}")
                u_bad = policy_model(x_bad.unsqueeze(0))
                x_next = dyn_model(x_bad.unsqueeze(0), u_bad)
                v_next = lyap_model(x_next)
                v_curr = lyap_model(x_bad.unsqueeze(0))
                print(f"    u(x_bad): {u_bad.squeeze().cpu().numpy()}")
                print(f"    x_next: {x_next.squeeze().cpu().numpy()}")
                print(f"    V(x_bad): {v_curr.item():.6f}, V(x_next): {v_next.item():.6f}, delta V: {(v_next - v_curr).item():.6f}")
        break
