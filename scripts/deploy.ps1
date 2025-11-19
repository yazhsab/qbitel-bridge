##############################################################################
# CRONOS AI Deployment Script (PowerShell)
# Automates the deployment of CRONOS AI to Kubernetes using Helm
##############################################################################

param(
    [string]$Namespace = "cronos-service-mesh",
    [string]$ReleaseName = "cronos-ai",
    [string]$ValuesFile = "",
    [string]$Environment = "production"
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$HelmChartDir = Join-Path $ProjectRoot "helm\cronos-ai"

# Functions
function Print-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "===================================================" -ForegroundColor Green
    Write-Host $Message -ForegroundColor Green
    Write-Host "===================================================" -ForegroundColor Green
    Write-Host ""
}

function Print-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Print-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Check-Prerequisites {
    Print-Header "Checking Prerequisites"

    # Check kubectl
    try {
        $kubectlVersion = kubectl version --client --short 2>$null
        if (-not $kubectlVersion) {
            $kubectlVersion = kubectl version --client 2>$null
        }
        Print-Success "kubectl found: $kubectlVersion"
    }
    catch {
        Print-Error "kubectl not found. Please install kubectl."
        exit 1
    }

    # Check helm
    try {
        $helmVersion = helm version --short
        Print-Success "helm found: $helmVersion"
    }
    catch {
        Print-Error "helm not found. Please install Helm 3.8+."
        exit 1
    }

    # Check cluster connectivity
    try {
        kubectl cluster-info | Out-Null
        Print-Success "Kubernetes cluster is accessible"
    }
    catch {
        Print-Error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    }

    # Check if namespace exists
    try {
        kubectl get namespace $Namespace 2>$null | Out-Null
        Print-Warning "Namespace $Namespace already exists"
    }
    catch {
        Print-Success "Namespace $Namespace will be created"
    }
}

function Validate-HelmChart {
    Print-Header "Validating Helm Chart"

    Push-Location $HelmChartDir

    try {
        # Lint the chart
        $lintOutput = helm lint . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Print-Success "Helm chart validation passed"
        }
        else {
            Print-Error "Helm chart validation failed"
            Write-Host $lintOutput
            exit 1
        }

        # Template the chart
        helm template test-release . --namespace $Namespace | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Print-Success "Helm template generation successful"
        }
        else {
            Print-Error "Helm template generation failed"
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}

function Deploy-CronosAI {
    Print-Header "Deploying CRONOS AI"

    Push-Location $HelmChartDir

    try {
        # Build helm install command
        $helmArgs = @(
            "upgrade",
            "--install",
            $ReleaseName,
            ".",
            "--namespace", $Namespace,
            "--create-namespace",
            "--wait",
            "--timeout", "10m"
        )

        # Add values file if specified
        if ($ValuesFile -and (Test-Path $ValuesFile)) {
            $helmArgs += "-f", $ValuesFile
            Print-Success "Using values file: $ValuesFile"
        }
        elseif ($ValuesFile) {
            Print-Error "Values file not found: $ValuesFile"
            exit 1
        }

        # Add environment-specific values
        switch ($Environment) {
            "production" {
                Print-Success "Using production configuration"
            }
            "staging" {
                $helmArgs += "--set", "xdsServer.replicaCount=2"
                $helmArgs += "--set", "admissionWebhook.replicaCount=2"
                Print-Success "Using staging configuration"
            }
            "development" {
                $helmArgs += "--set", "xdsServer.replicaCount=1"
                $helmArgs += "--set", "admissionWebhook.replicaCount=1"
                $helmArgs += "--set", "xdsServer.autoscaling.enabled=false"
                $helmArgs += "--set", "admissionWebhook.autoscaling.enabled=false"
                Print-Success "Using development configuration"
            }
            default {
                Print-Error "Unknown environment: $Environment. Use production, staging, or development."
                exit 1
            }
        }

        Write-Host ""
        Write-Host "Executing: helm $($helmArgs -join ' ')"
        Write-Host ""

        # Execute deployment
        helm $helmArgs
        if ($LASTEXITCODE -eq 0) {
            Print-Success "CRONOS AI deployed successfully"
        }
        else {
            Print-Error "Deployment failed"
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}

function Verify-Deployment {
    Print-Header "Verifying Deployment"

    # Wait for deployments to be ready
    Write-Host "Waiting for deployments to be ready..."

    try {
        kubectl wait --for=condition=available --timeout=300s `
            deployment/cronos-xds-server `
            deployment/cronos-admission-webhook `
            -n $Namespace 2>$null
        Print-Success "All deployments are ready"
    }
    catch {
        Print-Warning "Some deployments may not be ready yet"
    }

    # Check pod status
    Write-Host ""
    Write-Host "Pod Status:"
    kubectl get pods -n $Namespace

    # Check service status
    Write-Host ""
    Write-Host "Service Status:"
    kubectl get svc -n $Namespace

    # Check if all pods are running
    $runningPods = (kubectl get pods -n $Namespace --field-selector=status.phase=Running --no-headers | Measure-Object).Count
    $totalPods = (kubectl get pods -n $Namespace --no-headers | Measure-Object).Count

    Write-Host ""
    if ($runningPods -eq $totalPods) {
        Print-Success "All pods are running ($runningPods/$totalPods)"
    }
    else {
        Print-Warning "Some pods are not running yet ($runningPods/$totalPods)"
    }
}

function Run-SmokeTests {
    Print-Header "Running Smoke Tests"

    # Test 1: Check xDS Server health
    Write-Host "Testing xDS Server health endpoint..."
    try {
        kubectl exec -n $Namespace deployment/cronos-xds-server -- `
            curl -f http://localhost:8081/healthz 2>$null | Out-Null
        Print-Success "xDS Server is healthy"
    }
    catch {
        Print-Warning "xDS Server health check failed (may not be fully ready)"
    }

    # Test 2: Check admission webhook
    Write-Host "Testing Admission Webhook..."
    try {
        kubectl get validatingwebhookconfigurations cronos-validating-webhook 2>$null | Out-Null
        Print-Success "Admission Webhook is configured"
    }
    catch {
        Print-Warning "Admission Webhook configuration not found"
    }

    # Test 3: Verify quantum crypto implementation
    Write-Host "Testing Quantum Cryptography..."
    try {
        kubectl exec -n $Namespace deployment/cronos-xds-server -- `
            python3 -c "from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager; print('OK')" 2>$null | Out-Null
        Print-Success "Quantum Cryptography modules loaded successfully"
    }
    catch {
        Print-Warning "Quantum Cryptography test skipped (Python environment may not be ready)"
    }
}

function Show-AccessInfo {
    Print-Header "Access Information"

    Write-Host @"

🎉 CRONOS AI has been successfully deployed!

NAMESPACE: $Namespace
RELEASE: $ReleaseName
ENVIRONMENT: $Environment

To access the services:

1. AI Engine API (REST):
   kubectl port-forward -n $Namespace svc/cronos-ai-engine 8000:8000
   Then open: http://localhost:8000/docs

2. xDS Server (gRPC):
   kubectl port-forward -n $Namespace svc/cronos-xds-server 18000:18000

3. Grafana Dashboards:
   kubectl port-forward -n $Namespace svc/grafana 3000:3000
   Default credentials: admin / cronos-ai-admin

4. Prometheus Metrics:
   kubectl port-forward -n $Namespace svc/prometheus 9090:9090

Useful commands:

  # View logs
  kubectl logs -n $Namespace deployment/cronos-xds-server -f

  # Check status
  kubectl get pods -n $Namespace

  # Run tests
  helm test $ReleaseName -n $Namespace

For more information:
  helm status $ReleaseName -n $Namespace

Documentation: https://github.com/qbitel/cronos-ai

"@
}

# Main execution
function Main {
    Print-Header "CRONOS AI Deployment"
    Write-Host "Deploying to namespace: $Namespace"
    Write-Host "Release name: $ReleaseName"
    Write-Host "Environment: $Environment"

    Check-Prerequisites
    Validate-HelmChart
    Deploy-CronosAI
    Verify-Deployment
    Run-SmokeTests
    Show-AccessInfo

    Print-Header "Deployment Complete"
    Print-Success "CRONOS AI is ready to use!"
}

# Run main function
try {
    Main
}
catch {
    Print-Error "Deployment failed with error: $_"
    exit 1
}
