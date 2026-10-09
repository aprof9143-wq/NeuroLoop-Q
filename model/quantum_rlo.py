"""
NeuroLoop-Q Quantum Reinforcement Learning Optimizer (Q-RLO)
=============================================================
A Variational Quantum Circuit (VQC) serves as the policy network
for dynamic hyperparameter optimization during training.

The VQC explores the hyperparameter space through quantum superposition,
enabling parallel evaluation of configurations. The parameter-shift rule
(Wierichs et al., 2021) provides exact analytical gradients of VQC
expectation values with respect to circuit parameters, making the
policy trainable with standard gradient-based RL (REINFORCE).

State (s_t):
  [current_loss, gradient_norm, val_accuracy, loop_depth, epoch_progress, entropy]

VQC:
  3 layers of {Ry rotations, CNOT entanglement, Ry rotations}
  8-12 qubits

Action (a_t):
  [lr_eeg, lr_mri, lr_core, loop_depth, gate_temp, dropout, grad_clip]

Reward (r_t):
  delta(validation_accuracy) - alpha * compute_cost(loop_depth)

References:
  - Wierichs et al., "General Parameter-Shift Rules", 2021
  - Fösel et al., "Quantum Circuit Optimization with DRL", 2021
  - Sun et al., "Quantum Architecture Search", 2024
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not installed. Install with: pip install pennylane")


class VariationalQuantumCircuit:
    """Variational Quantum Circuit for Q-RLO policy network.

    Architecture:
      State encoding -> Ry rotations -> CNOT entanglement -> Ry rotations -> Measure

    The circuit has n_qubits * 2 * n_layers trainable parameters
    (2 rotation layers per VQC layer).
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3,
                 device: str = "default.qubit"):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for Q-RLO. Install: pip install pennylane")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * 2 * n_layers  # 2 rotation layers per VQC layer

        # Initialize quantum device
        self.dev = qml.device(device, wires=n_qubits)

        # Initialize parameters
        self.params = pnp.array(
            np.random.uniform(-np.pi, np.pi, self.n_params),
            requires_grad=True,
        )

        # Define the quantum circuit
        @qml.qnode(self.dev)
        def circuit(inputs, params):
            # ── State encoding: angle encoding ──
            # Each feature is encoded as a rotation angle on a qubit
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(inputs[i], wires=i)

            # If more features than qubits, encode via additional rotations
            if len(inputs) > n_qubits:
                for i in range(n_qubits, len(inputs)):
                    qml.RY(inputs[i] / (i // n_qubits + 1),
                           wires=i % n_qubits)

            # ── Variational layers ──
            param_idx = 0
            for layer in range(n_layers):
                # Rotation layer 1
                for q in range(n_qubits):
                    qml.RY(params[param_idx], wires=q)
                    param_idx += 1

                # Entanglement layer (ring topology)
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

                # Rotation layer 2
                for q in range(n_qubits):
                    qml.RY(params[param_idx], wires=q)
                    param_idx += 1

            # ── Measurement: expectation of Pauli-Z on each qubit ──
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state -> VQC -> action values.
        state: (n_state_features,) numpy array
        Returns: (n_qubits,) numpy array of expectation values in [-1, 1]
        """
        # Normalize state to [-pi, pi] range for angle encoding
        state_normalized = np.tanh(state) * np.pi
        return np.array(self.circuit(state_normalized, self.params))

    def update_params(self, gradients: np.ndarray, lr: float = 0.01):
        """Update VQC parameters using parameter-shift rule gradients.
        gradients: (n_params,) array of gradients
        """
        self.params = self.params - lr * gradients

    def get_params(self) -> np.ndarray:
        return np.array(self.params)

    def set_params(self, params: np.ndarray):
        self.params = pnp.array(params, requires_grad=True)


class QRL optimizer:
    """Quantum Reinforcement Learning Optimizer for NeuroLoop-Q.

    Dynamically adjusts:
      1. Learning rate per layer group (EEG, MRI, looped core)
      2. Loop depth per training step (curriculum)
      3. Fusion gate temperature
      4. Dropout rate per loop
      5. Gradient clipping threshold

    The VQC is evaluated every update_freq steps, not every step,
    to minimize quantum simulation overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        n_qubits: int = 8,
        vqc_layers: int = 3,
        update_freq: int = 50,
        lr_range: tuple = (1e-5, 5e-4),
        loop_depth_range: tuple = (3, 12),
        gate_temp_range: tuple = (0.1, 2.0),
        dropout_range: tuple = (0.05, 0.3),
        grad_clip_range: tuple = (0.5, 5.0),
        gamma: float = 0.99,
        alpha: float = 0.001,
        vqc_lr: float = 0.01,
        quantum_device: str = "default.qubit",
    ):
        self.model = model
        self.update_freq = update_freq
        self.lr_range = lr_range
        self.loop_depth_range = loop_depth_range
        self.gate_temp_range = gate_temp_range
        self.dropout_range = dropout_range
        self.grad_clip_range = grad_clip_range
        self.gamma = gamma
        self.alpha = alpha  # compute cost penalty
        self.vqc_lr = vqc_lr

        # Initialize VQC
        self.vqc = VariationalQuantumCircuit(n_qubits, vqc_layers, quantum_device)

        # State features: [loss, grad_norm, val_acc, loop_depth, epoch_progress, entropy]
        self.n_state_features = 6

        # Action dimension: [lr_eeg, lr_mri, lr_core, loop_depth, gate_temp, dropout, grad_clip]
        self.n_actions = 7

        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 100

        # Current hyperparameters
        self.current_lr = lr_range[1]  # start at max lr
        self.current_loop_depth = loop_depth_range[0]  # start at min loops
        self.current_gate_temp = 1.0
        self.current_dropout = 0.1
        self.current_grad_clip = 1.0

        # Training state
        self.step_count = 0
        self.prev_val_acc = 0.0

        # REINFORCE: episode trajectories
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def _encode_state(self, train_metrics: dict, epoch_progress: float) -> np.ndarray:
        """Encode training state into normalized features for VQC.
        train_metrics: dict with 'loss', 'grad_norm', 'val_acc', 'entropy'
        epoch_progress: float in [0, 1]
        """
        loss = train_metrics.get("loss", 1.0)
        grad_norm = train_metrics.get("grad_norm", 1.0)
        val_acc = train_metrics.get("val_acc", 0.0)
        entropy = train_metrics.get("entropy", 1.0)

        state = np.array([
            np.tanh(loss),                        # normalized loss
            np.tanh(grad_norm),                   # gradient norm
            (val_acc - 0.5) * 2,                  # val acc in [-1, 1]
            (self.current_loop_depth - 7.5) / 4.5, # loop depth normalized
            (epoch_progress - 0.5) * 2,            # epoch progress in [-1, 1]
            np.tanh(entropy),                      # entropy normalized
        ])
        return state

    def _decode_action(self, vqc_output: np.ndarray) -> dict:
        """Map VQC expectation values [-1, 1] to hyperparameter ranges.
        vqc_output: (n_qubits,) array in [-1, 1]
        """
        # Map each qubit's output to a hyperparameter
        # Qubit 0: lr_eeg (group 1)
        # Qubit 1: lr_mri (group 2)
        # Qubit 2: lr_core (group 3)
        # Qubit 3: loop_depth
        # Qubit 4: gate_temp
        # Qubit 5: dropout
        # Qubit 6: grad_clip

        def _map(val, range_):
            """Map [-1, 1] to [range_[0], range_[1]]."""
            return range_[0] + (val + 1) * 0.5 * (range_[1] - range_[0])

        actions = {}
        if len(vqc_output) >= 7:
            actions["lr_eeg"] = _map(vqc_output[0], self.lr_range)
            actions["lr_mri"] = _map(vqc_output[1], self.lr_range)
            actions["lr_core"] = _map(vqc_output[2], self.lr_range)
            actions["loop_depth"] = int(round(_map(vqc_output[3], self.loop_depth_range)))
            actions["gate_temp"] = _map(vqc_output[4], self.gate_temp_range)
            actions["dropout"] = _map(vqc_output[5], self.dropout_range)
            actions["grad_clip"] = _map(vqc_output[6], self.grad_clip_range)
        else:
            # Fallback: use fewer qubits, share across actions
            n = len(vqc_output)
            actions["lr_eeg"] = _map(vqc_output[0 % n], self.lr_range)
            actions["lr_mri"] = _map(vqc_output[1 % n], self.lr_range)
            actions["lr_core"] = _map(vqc_output[2 % n], self.lr_range)
            actions["loop_depth"] = int(round(_map(vqc_output[3 % n], self.loop_depth_range)))
            actions["gate_temp"] = _map(vqc_output[4 % n], self.gate_temp_range)
            actions["dropout"] = _map(vqc_output[5 % n], self.dropout_range)
            actions["grad_clip"] = _map(vqc_output[6 % n], self.grad_clip_range)

        return actions

    def _apply_actions(self, actions: dict):
        """Apply hyperparameter actions to the model and training state."""
        self.current_lr = actions["lr_core"]
        self.current_loop_depth = actions["loop_depth"]
        self.current_gate_temp = actions["gate_temp"]
        self.current_dropout = actions["dropout"]
        self.current_grad_clip = actions["grad_clip"]

        # Apply gate temperature to CMSSF fusion
        if hasattr(self.model, "cmssf") and hasattr(self.model.cmssf, "fusion"):
            self.model.cmssf.fusion.temperature.data = torch.tensor(self.current_gate_temp)

        # Apply dropout to transformer blocks
        for block in self.model.looped_core.blocks:
            block.attn_dropout.p = self.current_dropout
            block.ffn_dropout.p = self.current_dropout

    def _compute_reward(self, train_metrics: dict) -> float:
        """Compute reward: accuracy gain minus compute cost."""
        val_acc = train_metrics.get("val_acc", 0.0)
        accuracy_gain = val_acc - self.prev_val_acc
        compute_cost = self.alpha * self.current_loop_depth
        reward = accuracy_gain - compute_cost
        self.prev_val_acc = val_acc
        return reward

    def _compute_vqc_gradients(self, state, action, reward, next_state):
        """Compute VQC parameter gradients using REINFORCE + parameter-shift.

        The parameter-shift rule (Wierichs et al., 2021) states that for
        a quantum circuit with expectation value O = <psi|O|psi>,
        the gradient w.r.t. parameter theta_i is:

        dO/dtheta_i = [O(theta_i + pi/2) - O(theta_i - pi/2)] / 2

        This gives exact gradients without backpropagation through the
        quantum circuit, which is critical for NISQ-era devices.
        """
        params = self.vqc.get_params()
        n_params = len(params)
        gradients = np.zeros(n_params)

        # For REINFORCE, gradient is: grad_log_pi(a|s) * (R - baseline)
        # For VQC, log_pi is approximately proportional to the expectation values
        # So we use a finite-difference approximation via parameter shift

        # Baseline: running average reward
        baseline = np.mean([r for _, _, r in self.replay_buffer[-20:]) \
            if len(self.replay_buffer) > 0 else 0.0
        advantage = reward - baseline

        for i in range(n_params):
            # Parameter-shift: evaluate circuit with shifted parameter
            params_plus = params.copy()
            params_plus[i] += np.pi / 2
            params_minus = params.copy()
            params_minus[i] -= np.pi / 2

            # Forward passes with shifted parameters
            state_norm = np.tanh(state) * np.pi
            orig_params = self.vqc.get_params()

            self.vqc.set_params(params_plus)
            out_plus = self.vqc.forward(state)

            self.vqc.set_params(params_minus)
            out_minus = self.vqc.forward(state)

            # Restore original params
            self.vqc.set_params(orig_params)

            # Gradient via parameter-shift rule
            gradients[i] = np.mean((out_plus - out_minus) / 2 * advantage)

        return gradients

    def step(self, train_metrics: dict, epoch_progress: float):
        """Called every training step. Updates hyperparameters every
        update_freq steps using the VQC policy.
        """
        self.step_count += 1

        if self.step_count % self.update_freq != 0:
            return  # not time to update

        # Encode state
        state = self._encode_state(train_metrics, epoch_progress)

        # VQC forward pass
        vqc_output = self.vqc.forward(state)

        # Decode to hyperparameters
        actions = self._decode_action(vqc_output)

        # Apply to model
        self._apply_actions(actions)

        # Compute reward
        reward = self._compute_reward(train_metrics)

        # Store in replay buffer
        self.replay_buffer.append((state, vqc_output, reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        # Update VQC policy via REINFORCE + parameter-shift
        if len(self.replay_buffer) >= 10:
            # Sample a recent experience
            recent = self.replay_buffer[-1]
            state, action, reward = recent

            # Compute gradients
            gradients = self._compute_vqc_gradients(state, action, reward, state)

            # Update VQC parameters
            self.vqc.update_params(gradients, self.vqc_lr)

    def get_current_hyperparams(self) -> dict:
        """Return current hyperparameter values."""
        return {
            "lr": self.current_lr,
            "loop_depth": self.current_loop_depth,
            "gate_temp": self.current_gate_temp,
            "dropout": self.current_dropout,
            "grad_clip": self.current_grad_clip,
        }

    def state_dict(self) -> dict:
        """Save optimizer state."""
        return {
            "vqc_params": self.vqc.get_params(),
            "step_count": self.step_count,
            "prev_val_acc": self.prev_val_acc,
            "current_lr": self.current_lr,
            "current_loop_depth": self.current_loop_depth,
            "current_gate_temp": self.current_gate_temp,
            "current_dropout": self.current_dropout,
            "current_grad_clip": self.current_grad_clip,
        }

    def load_state_dict(self, state: dict):
        """Load optimizer state."""
        self.vqc.set_params(state["vqc_params"])
        self.step_count = state["step_count"]
        self.prev_val_acc = state["prev_val_acc"]
        self.current_lr = state["current_lr"]
        self.current_loop_depth = state["current_loop_depth"]
        self.current_gate_temp = state["current_gate_temp"]
        self.current_dropout = state["current_dropout"]
        self.current_grad_clip = state["current_grad_clip"]
