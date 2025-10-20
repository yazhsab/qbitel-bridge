"""
Global mocks for optional dependencies that aren't installed.
Import this at the TOP of conftest.py BEFORE anything else.
"""

import sys
from unittest.mock import MagicMock

# Mock mlflow completely
mock_mlflow = MagicMock()
mock_mlflow.tracking = MagicMock()
mock_mlflow.tracking.MlflowClient = MagicMock()
mock_mlflow.pytorch = MagicMock()
mock_mlflow.entities = MagicMock()
mock_mlflow.entities.model_registry = MagicMock()
mock_mlflow.entities.model_registry.ModelVersion = MagicMock()

sys.modules["mlflow"] = mock_mlflow
sys.modules["mlflow.tracking"] = mock_mlflow.tracking
sys.modules["mlflow.pytorch"] = mock_mlflow.pytorch
sys.modules["mlflow.entities"] = mock_mlflow.entities
sys.modules["mlflow.entities.model_registry"] = mock_mlflow.entities.model_registry

# Mock optuna completely
mock_optuna = MagicMock()
mock_optuna.integration = MagicMock()
mock_optuna.integration.mlflow = MagicMock()
mock_optuna.integration.mlflow.MLflowCallback = MagicMock()
mock_optuna.trial = MagicMock()
mock_optuna.samplers = MagicMock()

sys.modules["optuna"] = mock_optuna
sys.modules["optuna.integration"] = mock_optuna.integration
sys.modules["optuna.integration.mlflow"] = mock_optuna.integration.mlflow
sys.modules["optuna.trial"] = mock_optuna.trial
sys.modules["optuna.samplers"] = mock_optuna.samplers

# Mock wandb
sys.modules["wandb"] = MagicMock()

# Mock boto3 and botocore
mock_boto3 = MagicMock()
mock_botocore = MagicMock()
mock_botocore.exceptions = MagicMock()
mock_botocore.exceptions.ClientError = Exception  # Use real Exception class

sys.modules["boto3"] = mock_boto3
sys.modules["botocore"] = mock_botocore
sys.modules["botocore.exceptions"] = mock_botocore.exceptions

# Mock pyotp and qrcode
sys.modules["pyotp"] = MagicMock()
sys.modules["qrcode"] = MagicMock()
sys.modules["qrcode.image"] = MagicMock()
sys.modules["qrcode.image.svg"] = MagicMock()

# Mock etcd3 to avoid protobuf issues
mock_etcd3 = MagicMock()
mock_etcd3_client = MagicMock()
mock_etcd3.client = MagicMock(return_value=mock_etcd3_client)
mock_etcd3.Etcd3Client = MagicMock(return_value=mock_etcd3_client)

sys.modules["etcd3"] = mock_etcd3
sys.modules["etcd3.etcdrpc"] = MagicMock()
sys.modules["etcd3.etcdrpc.rpc_pb2"] = MagicMock()
sys.modules["etcd3.etcdrpc.kv_pb2"] = MagicMock()
sys.modules["etcd3.etcdrpc.auth_pb2"] = MagicMock()

print("✓ All optional dependencies mocked successfully")
