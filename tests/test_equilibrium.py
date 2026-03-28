import torch
import torch.nn.functional as F
from src.model.equilibrium import EquilibriumSignal, RoutingDecision, TokenEnergyBudget


def test_eq_signal_shape():
    eq = EquilibriumSignal(d_model=64)
    x = torch.randn(2, 16, 64)
    out = eq(x)
    assert out["ed"].shape == (2, 16)
    assert out["x"].shape == (2, 16, 64)


def test_eq_signal_updates_running_stats():
    eq = EquilibriumSignal(d_model=64)
    eq.train()
    x = torch.randn(2, 16, 64) * 5 + 3
    eq(x)
    assert eq.num_batches_tracked.item() == 1
    assert not torch.allclose(eq.running_mean, torch.zeros(64), atol=0.01)


def test_eq_signal_no_update_in_eval():
    eq = EquilibriumSignal(d_model=64)
    eq.eval()
    mean_before = eq.running_mean.clone()
    x = torch.randn(2, 16, 64) * 10
    eq(x)
    assert torch.allclose(eq.running_mean, mean_before)


def test_ed_higher_for_unusual_input():
    eq = EquilibriumSignal(d_model=64)
    eq.train()
    # Train on normal data
    for _ in range(100):
        eq(torch.randn(4, 16, 64))
    eq.eval()
    normal = torch.randn(1, 8, 64)
    unusual = torch.randn(1, 8, 64) * 10 + 20
    ed_normal = eq(normal)["ed"].mean()
    ed_unusual = eq(unusual)["ed"].mean()
    assert ed_unusual > ed_normal


def test_routing_decision_shape():
    rd = RoutingDecision()
    ed = torch.rand(2, 16)
    out = rd(ed)
    assert out["route"].shape == (2, 16)
    assert out["route_probs"].shape == (2, 16, 4)


def test_routing_probs_sum_to_one():
    rd = RoutingDecision()
    ed = torch.rand(2, 16) * 3
    out = rd(ed)
    sums = out["route_probs"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=0.1)


def test_energy_budget_shape():
    eb = TokenEnergyBudget()
    ed = torch.rand(2, 16)
    probs = torch.rand(2, 16, 4)
    probs = probs / probs.sum(-1, keepdim=True)
    budget = eb(ed, probs)
    assert budget.shape == (2, 16)
    assert budget.min() >= 1
    assert budget.max() <= 4


# --- Phase 0 tests ---

def test_eq_signal_warmup():
    eq = EquilibriumSignal(d_model=64, warmup_steps=5)
    eq.train()
    for i in range(5):
        out = eq(torch.randn(2, 8, 64))
        assert out["warming_up"] == (i < 4)  # step 0..3 warming, step 4 = 5th call = done
    out = eq(torch.randn(2, 8, 64))
    assert out["warming_up"] is False


def test_eq_signal_momentum_default():
    eq = EquilibriumSignal(d_model=64)
    assert eq.momentum == 0.1


def test_theta_ordering_guaranteed():
    rd = RoutingDecision()
    assert rd.theta2.item() > rd.theta1.item()
    assert rd.theta3.item() > rd.theta2.item()
    # Even after adversarial raw values
    rd.theta1_raw.data.fill_(10.0)
    rd.theta2_delta.data.fill_(-5.0)
    rd.theta3_delta.data.fill_(-5.0)
    assert rd.theta2.item() > rd.theta1.item()
    assert rd.theta3.item() > rd.theta2.item()


def test_theta_gradients_flow():
    rd = RoutingDecision()
    ed = torch.rand(2, 16, requires_grad=True)
    out = rd(ed)
    loss = out["route_probs"].sum()
    loss.backward()
    assert rd.theta1_raw.grad is not None
    assert rd.theta2_delta.grad is not None
    assert rd.theta3_delta.grad is not None
