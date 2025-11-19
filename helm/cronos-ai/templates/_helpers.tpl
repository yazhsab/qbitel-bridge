{{/*
Expand the name of the chart.
*/}}
{{- define "cronos-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cronos-ai.fullname" -}}
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
{{- define "cronos-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cronos-ai.labels" -}}
helm.sh/chart: {{ include "cronos-ai.chart" . }}
{{ include "cronos-ai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
cronos.ai/quantum-safe: "true"
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cronos-ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cronos-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
xDS Server labels
*/}}
{{- define "cronos-ai.xdsServer.labels" -}}
{{ include "cronos-ai.labels" . }}
app.kubernetes.io/component: xds-server
{{- end }}

{{/*
Admission Webhook labels
*/}}
{{- define "cronos-ai.admissionWebhook.labels" -}}
{{ include "cronos-ai.labels" . }}
app.kubernetes.io/component: admission-webhook
{{- end }}

{{/*
AI Engine labels
*/}}
{{- define "cronos-ai.aiEngine.labels" -}}
{{ include "cronos-ai.labels" . }}
app.kubernetes.io/component: ai-engine
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cronos-ai.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "cronos-ai.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the appropriate namespace
*/}}
{{- define "cronos-ai.namespace" -}}
{{- if .Values.namespaceOverride }}
{{- .Values.namespaceOverride }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the proper image name for xDS Server
*/}}
{{- define "cronos-ai.xdsServer.image" -}}
{{- $tag := .Values.xdsServer.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.xdsServer.image.repository $tag }}
{{- end }}

{{/*
Return the proper image name for Admission Webhook
*/}}
{{- define "cronos-ai.admissionWebhook.image" -}}
{{- $tag := .Values.admissionWebhook.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.admissionWebhook.image.repository $tag }}
{{- end }}

{{/*
Return the proper image name for AI Engine
*/}}
{{- define "cronos-ai.aiEngine.image" -}}
{{- $tag := .Values.aiEngine.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.aiEngine.image.repository $tag }}
{{- end }}
