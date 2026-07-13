from .baseline_cnn import PhysicalFeatureCNN, RealImagCNN
from .complex_no_interaction_cnn import ComplexCNNNoInteraction
from .phase_invariant_net import PhaseInvariantReceiver
from .single_invariant_net import SingleBranchPhaseInvariantReceiver


MODEL_CHOICES = [
    "real_imag_cnn",
    "physical_cnn",
    "phase_invariant",
    "complex_no_interaction",
    "single_branch",
]


def build_model(
    name,
    bits_per_symbol,
    hidden=32,
    hidden_complex=16,
    zero_complex=16,
    branch_layers=2,
    kernel_size=3,
    use_norm=True,
    gate_type="swiglu",
    single_readout_mode="low_rank",
):
    if name == "real_imag_cnn":
        return RealImagCNN(hidden=hidden, bits_per_symbol=bits_per_symbol)
    if name == "physical_cnn":
        return PhysicalFeatureCNN(
            hidden=hidden,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
        )
    if name == "phase_invariant":
        return PhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name == "complex_no_interaction":
        return ComplexCNNNoInteraction(
            hidden_complex=hidden_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name == "single_branch":
        return SingleBranchPhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            readout_mode=single_readout_mode,
        )
    raise ValueError(f"Unknown model: {name}")


def build_model_from_args(args, bits_per_symbol=2):
    return build_model(
        args.model,
        bits_per_symbol=bits_per_symbol,
        hidden=args.hidden,
        hidden_complex=args.hidden_complex,
        zero_complex=args.zero_complex,
        branch_layers=args.branch_layers,
        kernel_size=args.kernel_size,
        use_norm=not args.no_norm,
        gate_type=args.gate_type,
        single_readout_mode=args.single_readout_mode,
    )
