{{- define "evidently.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "evidently.namespace" -}}
{{- default .Release.Namespace .Values.namespace.name -}}
{{- end -}}

{{- define "evidently.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "evidently.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "evidently.uiName" -}}
{{- printf "%s-ui" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.postgresName" -}}
{{- printf "%s-postgres" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.postgresHeadlessName" -}}
{{- printf "%s-postgres-headless" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.postgresSecretName" -}}
{{- printf "%s-postgres" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.configSecretName" -}}
{{- printf "%s-config" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.workspacePvcName" -}}
{{- printf "%s-workspace" (include "evidently.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evidently.databaseUrl" -}}
{{- printf "postgresql://%s:%s@%s.%s.svc.cluster.local:%v/%s" .Values.postgres.auth.username .Values.postgres.auth.password (include "evidently.postgresName" .) (include "evidently.namespace" .) .Values.postgres.service.port .Values.postgres.auth.database -}}
{{- end -}}
