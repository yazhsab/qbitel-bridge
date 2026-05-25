"""
QBITEL - Fine-Tuning Pipeline Configuration

Defines training configurations for QBITEL specialist models using
QLoRA / LoRA fine-tuning on the latest open-source base models (March 2026).

Specialist Models:
    1. qbitel-protocol  — Protocol analysis, field detection, binary parsing
       Base: Qwen3-Coder-Next (80B/3B active, Apache 2.0)
    2. qbitel-security  — PQC algorithm selection, threat scoring, compliance
       Base: DeepSeek-R1-Distill-Qwen-32B (MIT)
    3. qbitel-translate  — Legacy→modern protocol translation
       Base: DeepSeek-R1-Distill-Qwen-32B (MIT)

Fine-Tuning Stack:
    - Unsloth (2x faster QLoRA, 60% less memory)
    - LLaMA-Factory (orchestration)
    - vLLM (serving)
    - MLflow (experiment tracking)
    - Weights & Biases (visualization)

Hardware Targets:
    - qbitel-protocol:  1× NVIDIA A6000 (48GB) or RTX 4090 (24GB)
    - qbitel-security:  2× NVIDIA A6000 (48GB) or 1× A100 (80GB)
    - qbitel-translate:  2× NVIDIA A6000 (48GB) or 1× A100 (80GB)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class BaseModel(Enum):
    """Supported base models for fine-tuning."""
    # MoE models (ultra-efficient)
    QWEN3_CODER_NEXT = "Qwen/Qwen3-Coder-Next"
    MIMO_V2_FLASH = "XiaomiMiMo/MiMo-V2-Flash"
    # Dense models (strong for fine-tuning)
    DEEPSEEK_R1_DISTILLED_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    DEEPSEEK_R1_DISTILLED_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    DEEPSEEK_R1_DISTILLED_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # Large reasoning models (require multi-GPU)
    QWEN3_235B = "Qwen/Qwen3-235B-A22B"
    DEEPSEEK_V3_2 = "deepseek-ai/DeepSeek-V3.2"
    # Legacy (for comparison baselines)
    LLAMA_4_SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    LLAMA_3_2_8B = "meta-llama/Llama-3.2-8B-Instruct"


class QuantizationMethod(Enum):
    """Quantization methods for memory-efficient training."""
    NONE = "none"           # Full precision (FP16/BF16)
    QLORA_4BIT = "4bit"     # QLoRA with 4-bit NormalFloat
    QLORA_8BIT = "8bit"     # QLoRA with 8-bit Integer
    FP8 = "fp8"             # Native FP8 (DeepSeek V3.2, MiMo)
    AWQ = "awq"             # Activation-aware Weight Quantization
    GPTQ = "gptq"           # Post-Training Quantization


class Optimizer(Enum):
    """Optimizers for fine-tuning."""
    ADAMW_8BIT = "adamw_8bit"       # Memory-efficient AdamW (recommended)
    ADAMW = "adamw_torch"           # Standard PyTorch AdamW
    PAGED_ADAMW = "paged_adamw_8bit"  # Paged optimizer (Unsloth)
    SGD = "sgd"                     # For comparison
    ADAFACTOR = "adafactor"         # Memory-efficient alternative


class LRScheduler(Enum):
    """Learning rate schedulers."""
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    LINEAR = "linear"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class TrainingStage(Enum):
    """Training pipeline stages."""
    SFT = "supervised_fine_tuning"      # Instruction tuning
    DPO = "direct_preference_optimization"  # Preference alignment
    GRPO = "group_relative_policy_optimization"  # DeepSeek-R1 style RL
    EVAL = "evaluation"                 # Benchmark evaluation
    MERGE = "merge_and_export"          # Merge LoRA + quantize for serving


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration."""
    r: int = 64                     # LoRA rank
    lora_alpha: int = 128           # Alpha scaling (typically 2× rank)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # MoE-specific: do NOT fine-tune the router layer
    modules_to_save: Optional[List[str]] = None
    fan_in_fan_out: bool = False
    bias: str = "none"              # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Dataset configuration for fine-tuning."""
    # Dataset paths
    train_data: str = ""
    eval_data: str = ""
    test_data: str = ""

    # Data format
    format: str = "instruction"     # "instruction", "conversation", "completion"
    instruction_field: str = "instruction"
    input_field: str = "context"
    output_field: str = "response"

    # Processing
    max_seq_length: int = 4096
    packing: bool = True            # Pack short examples into single sequences
    num_proc: int = 8               # Parallel preprocessing workers

    # Splits
    eval_split_ratio: float = 0.05
    seed: int = 42

    # Domain-specific dataset mixing ratios
    dataset_mix: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Core training configuration."""
    # Model
    base_model: BaseModel = BaseModel.QWEN3_CODER_NEXT
    quantization: QuantizationMethod = QuantizationMethod.QLORA_4BIT

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    warmup_steps: int = 0           # Overrides warmup_ratio if > 0

    # Optimizer & scheduler
    optimizer: Optimizer = Optimizer.PAGED_ADAMW
    lr_scheduler: LRScheduler = LRScheduler.COSINE

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True               # Use BFloat16 (requires Ampere+)
    fp16: bool = False
    tf32: bool = True

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["mlflow", "wandb"])

    # Distributed
    deepspeed_config: Optional[str] = None
    fsdp: Optional[str] = None      # "full_shard auto_wrap"
    local_rank: int = -1

    # Unsloth optimizations
    use_unsloth: bool = True
    unsloth_max_seq_length: int = 4096


@dataclass
class DPOConfig:
    """Direct Preference Optimization configuration."""
    beta: float = 0.1               # KL penalty coefficient
    loss_type: str = "sigmoid"      # "sigmoid" or "hinge"
    max_prompt_length: int = 1024
    max_length: int = 4096
    learning_rate: float = 5e-5     # Lower than SFT
    num_epochs: int = 1


@dataclass
class GRPOConfig:
    """
    Group Relative Policy Optimization configuration.
    (DeepSeek-R1 style reinforcement learning for reasoning.)
    """
    group_size: int = 8             # Number of responses per prompt
    kl_coeff: float = 0.04          # KL penalty
    clip_range: float = 0.2         # PPO clip range
    max_new_tokens: int = 2048
    num_generations: int = 4        # RL generation rounds
    reward_model: Optional[str] = None  # External reward model


@dataclass
class EvalConfig:
    """Evaluation benchmark configuration."""
    benchmarks: List[str] = field(default_factory=lambda: [
        "protocol_analysis",       # Custom: protocol field detection accuracy
        "pqc_selection",           # Custom: PQC algorithm recommendation accuracy
        "threat_scoring",          # Custom: quantum threat score correlation
        "compliance_check",        # Custom: regulatory compliance accuracy
        "swe_bench_verified",      # Standard: SWE-bench for general coding
        "humaneval",               # Standard: code generation
    ])
    num_samples_per_benchmark: int = 200
    temperature: float = 0.0        # Deterministic for eval


@dataclass
class ExportConfig:
    """Model export configuration for vLLM deployment."""
    merge_lora: bool = True
    output_format: str = "safetensors"
    quantize_merged: Optional[QuantizationMethod] = QuantizationMethod.FP8
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    vllm_ready: bool = True         # Ensure vLLM-compatible output


@dataclass
class FineTuneJob:
    """Complete fine-tuning job specification."""
    job_name: str
    description: str

    # Pipeline stages
    stages: List[TrainingStage] = field(default_factory=lambda: [
        TrainingStage.SFT,
        TrainingStage.EVAL,
        TrainingStage.MERGE,
    ])

    # Configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dpo: Optional[DPOConfig] = None
    grpo: Optional[GRPOConfig] = None
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Paths
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "qbitel-finetune"
    wandb_project: str = "qbitel-finetune"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for config files."""
        import dataclasses
        return dataclasses.asdict(self)


# =============================================================================
# Pre-defined Job Configurations for QBITEL Specialist Models
# =============================================================================

def create_protocol_job(
    datasets_dir: str = "./datasets",
    output_dir: str = "./output/qbitel-protocol",
) -> FineTuneJob:
    """
    Create fine-tuning job for QBITEL Protocol Analyst model.

    Base: Qwen3-Coder-Next (80B/3B active)
    Task: Protocol analysis, field detection, binary parsing, anomaly detection
    Hardware: 1× A6000 (48GB) or RTX 4090 (24GB)

    Training Data Mix:
        40% - Protocol samples (HL7, ISO-8583, Modbus, SIP, GOOSE, ACARS, V2X)
        30% - Field detection (IOB-tagged protocol fields)
        20% - Protocol translation pairs (legacy→modern)
        10% - Anomaly detection (normal vs malicious)
    """
    return FineTuneJob(
        job_name="qbitel-protocol-v1",
        description="Protocol analysis specialist on Qwen3-Coder-Next",
        stages=[TrainingStage.SFT, TrainingStage.DPO, TrainingStage.EVAL, TrainingStage.MERGE],
        training=TrainingConfig(
            base_model=BaseModel.QWEN3_CODER_NEXT,
            quantization=QuantizationMethod.QLORA_4BIT,
            lora=LoRAConfig(
                r=64,
                lora_alpha=128,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                # Do NOT tune the MoE router for Qwen3-Coder-Next
            ),
            num_epochs=3,
            per_device_train_batch_size=8,  # Small active params = larger batch
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            use_unsloth=True,
            unsloth_max_seq_length=8192,
        ),
        data=DataConfig(
            train_data=f"{datasets_dir}/protocol_finetune/train.jsonl",
            eval_data=f"{datasets_dir}/protocol_finetune/eval.jsonl",
            max_seq_length=8192,
            packing=True,
            dataset_mix={
                "protocol_samples": 0.40,
                "field_detection": 0.30,
                "protocol_translation": 0.20,
                "anomaly_detection": 0.10,
            },
        ),
        dpo=DPOConfig(
            beta=0.1,
            learning_rate=5e-5,
            num_epochs=1,
        ),
        eval_config=EvalConfig(
            benchmarks=[
                "protocol_analysis",
                "field_detection_accuracy",
                "protocol_translation_bleu",
                "anomaly_detection_f1",
                "humaneval",
            ],
        ),
        export=ExportConfig(
            merge_lora=True,
            quantize_merged=QuantizationMethod.FP8,
            vllm_ready=True,
        ),
        output_dir=output_dir,
        mlflow_experiment_name="qbitel-protocol",
        wandb_project="qbitel-protocol",
    )


def create_security_job(
    datasets_dir: str = "./datasets",
    output_dir: str = "./output/qbitel-security",
) -> FineTuneJob:
    """
    Create fine-tuning job for QBITEL Security Analyst model.

    Base: DeepSeek-R1-Distill-Qwen-32B (dense, strong CoT reasoning)
    Task: PQC algorithm selection, quantum threat scoring, compliance analysis
    Hardware: 2× A6000 (48GB) or 1× A100 (80GB)

    Training Data Mix:
        35% - PQC algorithm selection (domain-specific recommendations)
        25% - Quantum threat assessment (Mosca's theorem, risk scoring)
        20% - Compliance analysis (CNSA 2.0, HIPAA, PCI-DSS, IEC 62443)
        10% - Migration planning (hybrid mode, certificate rollover)
        10% - Protocol security analysis (TLS, SSH, IKEv2 PQC integration)
    """
    return FineTuneJob(
        job_name="qbitel-security-v1",
        description="PQC security analyst on DeepSeek-R1-Distilled-32B",
        stages=[
            TrainingStage.SFT,
            TrainingStage.GRPO,  # Use GRPO for reasoning quality
            TrainingStage.EVAL,
            TrainingStage.MERGE,
        ],
        training=TrainingConfig(
            base_model=BaseModel.DEEPSEEK_R1_DISTILLED_32B,
            quantization=QuantizationMethod.QLORA_4BIT,
            lora=LoRAConfig(
                r=64,
                lora_alpha=128,
            ),
            num_epochs=3,
            per_device_train_batch_size=2,  # 32B dense = smaller batch
            gradient_accumulation_steps=8,
            learning_rate=1e-4,  # Slightly lower for dense model
            use_unsloth=True,
            unsloth_max_seq_length=4096,
        ),
        data=DataConfig(
            train_data=f"{datasets_dir}/security_finetune/train.jsonl",
            eval_data=f"{datasets_dir}/security_finetune/eval.jsonl",
            max_seq_length=4096,
            packing=True,
            dataset_mix={
                "pqc_algorithm_selection": 0.35,
                "quantum_threat_assessment": 0.25,
                "compliance_analysis": 0.20,
                "migration_planning": 0.10,
                "protocol_security": 0.10,
            },
        ),
        grpo=GRPOConfig(
            group_size=4,
            kl_coeff=0.04,
            num_generations=2,
            max_new_tokens=2048,
        ),
        eval_config=EvalConfig(
            benchmarks=[
                "pqc_selection",
                "threat_scoring",
                "compliance_check",
                "migration_planning_quality",
                "humaneval",
            ],
        ),
        export=ExportConfig(
            merge_lora=True,
            quantize_merged=QuantizationMethod.FP8,
            vllm_ready=True,
        ),
        output_dir=output_dir,
        mlflow_experiment_name="qbitel-security",
        wandb_project="qbitel-security",
    )


def create_translate_job(
    datasets_dir: str = "./datasets",
    output_dir: str = "./output/qbitel-translate",
) -> FineTuneJob:
    """
    Create fine-tuning job for QBITEL Protocol Translator model.

    Base: DeepSeek-R1-Distill-Qwen-32B (strong at code translation)
    Task: Legacy→modern protocol translation with PQC integration
    Hardware: 2× A6000 (48GB) or 1× A100 (80GB)

    Translation Pairs:
        - TN3270e → REST API
        - HL7 v2.x → FHIR R4
        - ISO 8583 → ISO 20022 (pacs.008)
        - GOOSE → OPC-UA
        - ACARS → AeroMACS
        - SIP → WebRTC
        - Modbus → OPC-UA
    """
    return FineTuneJob(
        job_name="qbitel-translate-v1",
        description="Protocol translator on DeepSeek-R1-Distilled-32B",
        stages=[TrainingStage.SFT, TrainingStage.DPO, TrainingStage.EVAL, TrainingStage.MERGE],
        training=TrainingConfig(
            base_model=BaseModel.DEEPSEEK_R1_DISTILLED_32B,
            quantization=QuantizationMethod.QLORA_4BIT,
            lora=LoRAConfig(
                r=64,
                lora_alpha=128,
            ),
            num_epochs=5,  # More epochs for translation accuracy
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            use_unsloth=True,
            unsloth_max_seq_length=8192,  # Longer for translation pairs
        ),
        data=DataConfig(
            train_data=f"{datasets_dir}/translation_finetune/train.jsonl",
            eval_data=f"{datasets_dir}/translation_finetune/eval.jsonl",
            max_seq_length=8192,
            packing=False,  # Don't pack translation pairs
            dataset_mix={
                "tn3270e_to_rest": 0.15,
                "hl7v2_to_fhir": 0.20,
                "iso8583_to_iso20022": 0.20,
                "goose_to_opcua": 0.15,
                "acars_to_aeromacs": 0.10,
                "sip_to_webrtc": 0.10,
                "modbus_to_opcua": 0.10,
            },
        ),
        dpo=DPOConfig(
            beta=0.1,
            learning_rate=5e-5,
            num_epochs=1,
        ),
        eval_config=EvalConfig(
            benchmarks=[
                "protocol_translation_bleu",
                "translation_correctness",
                "pqc_preservation",  # Verify PQC properties survive translation
                "humaneval",
            ],
        ),
        export=ExportConfig(
            merge_lora=True,
            quantize_merged=QuantizationMethod.FP8,
            vllm_ready=True,
        ),
        output_dir=output_dir,
        mlflow_experiment_name="qbitel-translate",
        wandb_project="qbitel-translate",
    )


# =============================================================================
# Dataset Assembly Utilities
# =============================================================================


@dataclass
class DatasetMixer:
    """
    Mix multiple dataset sources according to specified ratios.

    Reads JSONL files from multiple directories and produces a single
    shuffled JSONL with the specified mixing proportions.
    """
    sources: Dict[str, str]         # name → path mapping
    ratios: Dict[str, float]        # name → proportion (must sum to 1.0)
    output_path: str
    seed: int = 42
    max_total_samples: Optional[int] = None

    def mix(self) -> Dict[str, int]:
        """
        Mix datasets and write to output_path.

        Returns dict mapping source name → number of samples used.
        """
        import json
        import random

        random.seed(self.seed)
        all_samples = []
        counts = {}

        # Calculate total budget
        total = self.max_total_samples or self._count_total_available()

        for name, path in self.sources.items():
            ratio = self.ratios.get(name, 0.0)
            target_count = int(total * ratio)

            samples = self._load_jsonl(path)
            if len(samples) > target_count:
                samples = random.sample(samples, target_count)
            elif len(samples) < target_count:
                logger.warning(
                    f"Dataset '{name}' has {len(samples)} samples but "
                    f"requested {target_count}. Using all available."
                )

            # Tag source
            for s in samples:
                s["_source"] = name

            all_samples.extend(samples)
            counts[name] = len(samples)

        # Shuffle
        random.shuffle(all_samples)

        # Write output
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(
            f"Mixed dataset: {len(all_samples)} samples from "
            f"{len(counts)} sources → {self.output_path}"
        )

        return counts

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file."""
        import json

        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _count_total_available(self) -> int:
        """Count total available samples across all sources."""
        total = 0
        for path in self.sources.values():
            try:
                with open(path, "r") as f:
                    total += sum(1 for line in f if line.strip())
            except FileNotFoundError:
                pass
        return total


# =============================================================================
# Helper: Generate Preference Pairs for DPO
# =============================================================================


def generate_preference_pairs(
    instruction_data_path: str,
    output_path: str,
    num_pairs: int = 5000,
    seed: int = 42,
) -> int:
    """
    Generate preference pairs for DPO training from instruction data.

    For each instruction, creates:
    - chosen: The original high-quality response (from instruction data)
    - rejected: A degraded response with common failure modes

    Failure modes simulated:
    1. Wrong algorithm recommendation (e.g., RSA instead of ML-KEM)
    2. Missing compliance consideration
    3. Ignoring domain constraints (e.g., memory limits for medical devices)
    4. Insecure hybrid mode configuration
    5. Outdated algorithm reference (pre-NIST standardization names)

    Returns number of pairs generated.
    """
    import json
    import random

    random.seed(seed)

    degradation_strategies = [
        _degrade_wrong_algorithm,
        _degrade_missing_compliance,
        _degrade_ignore_constraints,
        _degrade_insecure_config,
        _degrade_outdated_names,
    ]

    pairs = []
    with open(instruction_data_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for sample in samples[:num_pairs]:
        strategy = random.choice(degradation_strategies)
        rejected = strategy(sample)

        pair = {
            "prompt": sample.get("instruction", ""),
            "chosen": sample.get("response", ""),
            "rejected": rejected,
        }
        pairs.append(pair)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(pairs)} preference pairs → {output_path}")
    return len(pairs)


def _degrade_wrong_algorithm(sample: Dict) -> str:
    """Replace PQC algorithms with classical/wrong ones."""
    response = sample.get("response", "")
    replacements = {
        "ML-KEM-768": "RSA-2048",
        "ML-KEM-1024": "ECDH-P256",
        "ML-DSA-65": "ECDSA-P256",
        "ML-DSA-87": "RSA-3072",
        "Falcon-512": "Ed25519",
        "SLH-DSA": "SHA-256",
        "XMSS": "HMAC-SHA256",
    }
    for pqc, classical in replacements.items():
        response = response.replace(pqc, classical)
    return response


def _degrade_missing_compliance(sample: Dict) -> str:
    """Remove compliance-related content."""
    response = sample.get("response", "")
    compliance_terms = [
        "CNSA 2.0", "FIPS 203", "FIPS 204", "FIPS 205",
        "NIST SP 800-208", "PCI-DSS", "HIPAA", "IEC 62443",
        "compliance", "regulatory", "audit",
    ]
    lines = response.split("\n")
    filtered = [
        line for line in lines
        if not any(term.lower() in line.lower() for term in compliance_terms)
    ]
    return "\n".join(filtered)


def _degrade_ignore_constraints(sample: Dict) -> str:
    """Remove domain constraint considerations."""
    response = sample.get("response", "")
    constraint_terms = [
        "memory", "battery", "latency", "bandwidth", "constrained",
        "embedded", "real-time", "sub-10ms", "lightweight",
    ]
    lines = response.split("\n")
    filtered = [
        line for line in lines
        if not any(term.lower() in line.lower() for term in constraint_terms)
    ]
    return "\n".join(filtered)


def _degrade_insecure_config(sample: Dict) -> str:
    """Introduce insecure configuration suggestions."""
    response = sample.get("response", "")
    insecure_additions = [
        "\n\nNote: For faster deployment, you can skip hybrid mode and go directly to PQC-only.",
        "\n\nTip: Disable certificate validation during migration for smoother transition.",
        "\n\nOptional: Use NIST Level 1 security for all applications to minimize overhead.",
    ]
    import random
    return response + random.choice(insecure_additions)


def _degrade_outdated_names(sample: Dict) -> str:
    """Use pre-standardization / outdated algorithm names."""
    response = sample.get("response", "")
    outdated_names = {
        "ML-KEM": "CRYSTALS-Kyber",
        "ML-DSA": "CRYSTALS-Dilithium",
        "SLH-DSA": "SPHINCS+",
        "FIPS 203": "Round 3 KEM",
        "FIPS 204": "Round 3 DSA",
    }
    for current, outdated in outdated_names.items():
        response = response.replace(current, outdated)
    return response


# =============================================================================
# CLI Entry Point
# =============================================================================


def print_job_summary(job: FineTuneJob) -> None:
    """Print a human-readable summary of a fine-tuning job."""
    print(f"\n{'='*60}")
    print(f"Fine-Tuning Job: {job.job_name}")
    print(f"{'='*60}")
    print(f"Description:    {job.description}")
    print(f"Base Model:     {job.training.base_model.value}")
    print(f"Quantization:   {job.training.quantization.value}")
    print(f"LoRA Rank:      {job.training.lora.r}")
    print(f"Epochs:         {job.training.num_epochs}")
    print(f"Batch Size:     {job.training.per_device_train_batch_size}")
    print(f"Learning Rate:  {job.training.learning_rate}")
    print(f"Optimizer:      {job.training.optimizer.value}")
    print(f"Scheduler:      {job.training.lr_scheduler.value}")
    print(f"Stages:         {' → '.join(s.value for s in job.stages)}")
    print(f"Output:         {job.output_dir}")
    print()
    print("Dataset Mix:")
    for source, ratio in job.data.dataset_mix.items():
        print(f"  {source:30s} {ratio:5.0%}")
    print()
    print("Evaluation Benchmarks:")
    for bench in job.eval_config.benchmarks:
        print(f"  - {bench}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("QBITEL Fine-Tuning Pipeline - Job Configurations\n")

    protocol_job = create_protocol_job()
    security_job = create_security_job()
    translate_job = create_translate_job()

    for job in [protocol_job, security_job, translate_job]:
        print_job_summary(job)
