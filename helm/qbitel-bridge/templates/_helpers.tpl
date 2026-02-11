{{/*
Expand the name of the chart.
*/}}
{{- define "qbitel-bridge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "qbitel-bridge.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "qbitel-bridge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "qbitel-bridge.labels" -}}
helm.sh/chart: {{ include "qbitel-bridge.chart" . }}
{{ include "qbitel-bridge.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
qbitel.com/quantum-safe: "true"
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "qbitel-bridge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "qbitel-bridge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
xDS Server labels
*/}}
{{- define "qbitel-bridge.xdsServer.labels" -}}
{{ include "qbitel-bridge.labels" . }}
app.kubernetes.io/component: xds-server
{{- end }}

{{/*
Admission Webhook labels
*/}}
{{- define "qbitel-bridge.admissionWebhook.labels" -}}
{{ include "qbitel-bridge.labels" . }}
app.kubernetes.io/component: admission-webhook
{{- end }}

{{/*
AI Engine labels
*/}}
{{- define "qbitel-bridge.aiEngine.labels" -}}
{{ include "qbitel-bridge.labels" . }}
app.kubernetes.io/component: ai-engine
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "qbitel-bridge.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "qbitel-bridge.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the appropriate namespace
*/}}
{{- define "qbitel-bridge.namespace" -}}
{{- if .Values.namespaceOverride }}
{{- .Values.namespaceOverride }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the proper image name for xDS Server
*/}}
{{- define "qbitel-bridge.xdsServer.image" -}}
{{- $tag := .Values.xdsServer.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.xdsServer.image.repository $tag }}
{{- end }}

{{/*
Return the proper image name for Admission Webhook
*/}}
{{- define "qbitel-bridge.admissionWebhook.image" -}}
{{- $tag := .Values.admissionWebhook.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.admissionWebhook.image.repository $tag }}
{{- end }}

{{/*
Return the proper image name for AI Engine
*/}}
{{- define "qbitel-bridge.aiEngine.image" -}}
{{- $tag := .Values.aiEngine.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.aiEngine.image.repository $tag }}
{{- end }}
