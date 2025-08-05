from typing import Dict, List, Optional
import torch
from peft import PeftModel

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAdam(GradientAggregation):
    """
    Server-side FedAdam aggregator for PEFT/LoRA adapters.
    Keeps Adam moments per-parameter across rounds and applies an Adam step
    on the global adapter from the average client delta.

    Args:
        server_lr: Server learning rate (η). Typical range: 1e-3 to 1e-1. Start with 1e-2 or 5e-3.
        beta1: Adam β1 (first-moment decay).
        beta2: Adam β2 (second-moment decay).
        eps: Adam ε for numerical stability.
        bias_correction: Use Adam bias correction terms if True.
        use_fp32_state: Track moments in FP32 regardless of parameter dtype (recommended True).
    """

    def __init__(
        self,
        server_lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        bias_correction: bool = True,
        use_fp32_state: bool = True,
    ):
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction
        self.use_fp32_state = use_fp32_state

        # Adam state
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.t: int = 0  # round counter

    def _get_state_buffers(self, key: str, like: torch.Tensor) -> None:
        """Initialize state buffers for a parameter key if needed."""
        dtype = torch.float32 if self.use_fp32_state else like.dtype
        device = like.device
        if key not in self.m:
            self.m[key] = torch.zeros_like(like, dtype=dtype, device=device)
        elif self.m[key].device != device:
            self.m[key] = self.m[key].to(device)
        if key not in self.v:
            self.v[key] = torch.zeros_like(like, dtype=dtype, device=device)
        elif self.v[key].device != device:
            self.v[key] = self.v[key].to(device)

    @staticmethod
    def _get_client_weight(adapter: Dict) -> float:
        # Prefer num_samples if present, else an explicit weight, else uniform
        if "num_samples" in adapter:
            return float(adapter["num_samples"])
        if "weight" in adapter:
            return float(adapter["weight"])
        return 1.0

    @torch.no_grad()
    def aggregate(self, adapters: List[Dict], global_adapter: Dict, peft_model: PeftModel) -> None:
        """
        Update the global adapter in-place using FedAdam.

        Expected inputs:
            adapters: list of dicts, each with at least {"name": "<client_adapter_name>", ...}
                      Optionally include {"num_samples": int} or {"weight": float} for weighting.
            global_adapter: dict with {"name": "<global_adapter_name>", "keys": [list of state_dict keys to update]}
            peft_model: PeftModel containing the global adapter *and* the client adapters' parameters
                        already loaded under their respective adapter names.

        Behavior:
            - Computes avg client parameter tensor per key using provided weights.
            - Forms delta g_t = w_global - w_avg and applies Adam update: w <- w - η * m_hat / (sqrt(v_hat) + eps)
        """
        state_dict = peft_model.state_dict()

        # Increment round once per aggregation
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        lr = self.server_lr

        for k in global_adapter["keys"]:
            # Collect client tensors (casting to FP32 for stability) and weights
            client_tensors = []
            client_weights = []
            for adapter in adapters:
                client_key = k.replace(global_adapter["name"], adapter["name"])
                if client_key not in state_dict:
                    # Skip missing keys gracefully
                    continue
                client_tensors.append(state_dict[client_key].detach().float())
                client_weights.append(self._get_client_weight(adapter))

            if not client_tensors:
                # Nothing to aggregate for this key
                continue

            # Normalize weights
            w = torch.tensor(client_weights, dtype=torch.float32, device=client_tensors[0].device)
            w = w / (w.sum() + 1e-12)

            # Weighted average of client params
            stacked = torch.stack(client_tensors, dim=0)  # [num_clients, ...]
            # Broadcast weights to param shape
            while w.ndim < stacked.ndim:
                w = w.unsqueeze(-1)
            avg_client_param = (stacked * w).sum(dim=0)

            # Current global param
            gparam = state_dict[k]
            gparam_fp32 = gparam.detach().float().to(avg_client_param.device)

            # Pseudo-gradient: global - average(client)  (aligns with gradient direction)
            g_t = gparam_fp32 - avg_client_param

            # Adam state
            self._get_state_buffers(k, g_t)
            m = self.m[k]
            v = self.v[k]

            # Moments update
            m.mul_(b1).add_(g_t, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g_t, g_t, value=1.0 - b2)

            if self.bias_correction:
                m_hat = m / (1.0 - (b1 ** self.t))
                v_hat = v / (1.0 - (b2 ** self.t))
            else:
                m_hat, v_hat = m, v

            # Adam step on the global param
            step = m_hat / (v_hat.sqrt() + eps)
            new_param = (gparam_fp32 - lr * step).to(gparam.dtype)

            # In-place update of the global parameter
            gparam.copy_(new_param)

    def reset_state(self) -> None:
        """Optional: clear Adam moments (e.g. when you change the global init)."""
        self.m.clear()
        self.v.clear()
        self.t = 0
