import {
  Device,
  DeviceFilters,
  EnrollmentRequest,
  EnrollmentSession,
  AttestationData,
  DeviceCertificate,
  DevicePolicy,
  DeviceMetrics,
  DeviceActivity,
  ComplianceViolation,
  DeviceAlert,
  ListDevicesResponse,
  DeviceDetailsResponse,
  DeviceMetricsResponse,
  DeviceConfigurationForm,
  SuspendDeviceForm,
  DecommissionDeviceForm,
} from '../types/device';

// Base API configuration
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'https://api.qslb.local';
const API_VERSION = 'v1';

interface ApiError {
  error: string;
  message: string;
  details?: Record<string, any>;
  trace_id?: string;
}

class DeviceApiError extends Error {
  public readonly status: number;
  public readonly details?: Record<string, any>;
  public readonly traceId?: string;

  constructor(message: string, status: number, details?: Record<string, any>, traceId?: string) {
    super(message);
    this.name = 'DeviceApiError';
    this.status = status;
    this.details = details;
    this.traceId = traceId;
  }
}

class DeviceApiClient {
  private baseUrl: string;
  private getAuthToken: () => Promise<string | null>;

  constructor(getAuthToken: () => Promise<string | null>) {
    this.baseUrl = `${API_BASE_URL}/${API_VERSION}`;
    this.getAuthToken = getAuthToken;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const token = await this.getAuthToken();
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Request-ID': crypto.randomUUID(),
      ...(options.headers as Record<string, string>),
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorData: ApiError;
      try {
        errorData = await response.json();
      } catch {
        errorData = {
          error: 'HTTP_ERROR',
          message: `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      throw new DeviceApiError(
        errorData.message,
        response.status,
        errorData.details,
        errorData.trace_id
      );
    }

    return response.json();
  }

  // Device Management
  async listDevices(
    page = 1,
    pageSize = 50,
    filters?: DeviceFilters,
    search?: string,
    sortBy?: string,
    sortOrder: 'asc' | 'desc' = 'desc'
  ): Promise<ListDevicesResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      sort_by: sortBy || 'updated_at',
      sort_order: sortOrder,
    });

    if (search) {
      params.append('search', search);
    }

    if (filters) {
      if (filters.status?.length) {
        filters.status.forEach(status => params.append('status', status));
      }
      if (filters.device_type?.length) {
        filters.device_type.forEach(type => params.append('device_type', type));
      }
      if (filters.manufacturer?.length) {
        filters.manufacturer.forEach(mfg => params.append('manufacturer', mfg));
      }
      if (filters.tags?.length) {
        filters.tags.forEach(tag => params.append('tags', tag));
      }
      if (filters.compliance_status?.length) {
        filters.compliance_status.forEach(status => params.append('compliance_status', status));
      }
      if (filters.health_status?.length) {
        filters.health_status.forEach(status => params.append('health_status', status));
      }
    }

    return this.makeRequest<ListDevicesResponse>(`/devices?${params}`);
  }

  async getDevice(deviceId: string): Promise<DeviceDetailsResponse> {
    return this.makeRequest<DeviceDetailsResponse>(`/devices/${deviceId}`);
  }

  async updateDevice(
    deviceId: string,
    updates: DeviceConfigurationForm
  ): Promise<Device> {
    return this.makeRequest<Device>(`/devices/${deviceId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  async suspendDevice(
    deviceId: string,
    suspendData: SuspendDeviceForm
  ): Promise<Device> {
    return this.makeRequest<Device>(`/devices/${deviceId}/suspend`, {
      method: 'POST',
      body: JSON.stringify(suspendData),
    });
  }

  async resumeDevice(deviceId: string): Promise<Device> {
    return this.makeRequest<Device>(`/devices/${deviceId}/resume`, {
      method: 'POST',
    });
  }

  async decommissionDevice(
    deviceId: string,
    decommissionData: DecommissionDeviceForm
  ): Promise<void> {
    await this.makeRequest<void>(`/devices/${deviceId}/decommission`, {
      method: 'POST',
      body: JSON.stringify(decommissionData),
    });
  }

  // Device Enrollment
  async createEnrollmentSession(
    request: EnrollmentRequest
  ): Promise<EnrollmentSession> {
    return this.makeRequest<EnrollmentSession>('/devices/enrollment/sessions', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getEnrollmentSession(sessionId: string): Promise<EnrollmentSession> {
    return this.makeRequest<EnrollmentSession>(`/devices/enrollment/sessions/${sessionId}`);
  }

  async submitAttestation(
    sessionId: string,
    attestationData: AttestationData
  ): Promise<EnrollmentSession> {
    return this.makeRequest<EnrollmentSession>(
      `/devices/enrollment/sessions/${sessionId}/attestation`,
      {
        method: 'POST',
        body: JSON.stringify(attestationData),
      }
    );
  }

  async approveEnrollment(sessionId: string): Promise<Device> {
    return this.makeRequest<Device>(
      `/devices/enrollment/sessions/${sessionId}/approve`,
      {
        method: 'POST',
      }
    );
  }

  async rejectEnrollment(sessionId: string, reason: string): Promise<void> {
    await this.makeRequest<void>(
      `/devices/enrollment/sessions/${sessionId}/reject`,
      {
        method: 'POST',
        body: JSON.stringify({ reason }),
      }
    );
  }

  // Certificate Management
  async getDeviceCertificate(deviceId: string): Promise<DeviceCertificate> {
    return this.makeRequest<DeviceCertificate>(`/devices/${deviceId}/certificate`);
  }

  async renewDeviceCertificate(deviceId: string): Promise<DeviceCertificate> {
    return this.makeRequest<DeviceCertificate>(`/devices/${deviceId}/certificate/renew`, {
      method: 'POST',
    });
  }

  async revokeDeviceCertificate(
    deviceId: string,
    reason: string
  ): Promise<void> {
    await this.makeRequest<void>(`/devices/${deviceId}/certificate/revoke`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  }

  // Policy Management
  async getDevicePolicies(deviceId: string): Promise<DevicePolicy[]> {
    return this.makeRequest<DevicePolicy[]>(`/devices/${deviceId}/policies`);
  }

  async updateDevicePolicy(
    deviceId: string,
    policySet: string,
    policies: Record<string, any>
  ): Promise<DevicePolicy> {
    return this.makeRequest<DevicePolicy>(`/devices/${deviceId}/policies`, {
      method: 'PUT',
      body: JSON.stringify({ policy_set: policySet, policies }),
    });
  }

  async deployPolicyToDevice(deviceId: string, policyId: string): Promise<void> {
    await this.makeRequest<void>(`/devices/${deviceId}/policies/${policyId}/deploy`, {
      method: 'POST',
    });
  }

  // Compliance and Health
  async triggerComplianceCheck(deviceId: string): Promise<void> {
    await this.makeRequest<void>(`/devices/${deviceId}/compliance/check`, {
      method: 'POST',
    });
  }

  async getComplianceViolations(deviceId: string): Promise<ComplianceViolation[]> {
    return this.makeRequest<ComplianceViolation[]>(`/devices/${deviceId}/compliance/violations`);
  }

  async resolveComplianceViolation(
    deviceId: string,
    violationId: string,
    resolutionNotes: string
  ): Promise<void> {
    await this.makeRequest<void>(
      `/devices/${deviceId}/compliance/violations/${violationId}/resolve`,
      {
        method: 'POST',
        body: JSON.stringify({ resolution_notes: resolutionNotes }),
      }
    );
  }

  async triggerHealthCheck(deviceId: string): Promise<void> {
    await this.makeRequest<void>(`/devices/${deviceId}/health/check`, {
      method: 'POST',
    });
  }

  // Alerts and Activity
  async getDeviceAlerts(
    deviceId?: string,
    severity?: string[],
    alertType?: string[],
    resolved?: boolean
  ): Promise<DeviceAlert[]> {
    const params = new URLSearchParams();
    
    if (deviceId) {
      params.append('device_id', deviceId);
    }
    if (severity?.length) {
      severity.forEach(s => params.append('severity', s));
    }
    if (alertType?.length) {
      alertType.forEach(t => params.append('alert_type', t));
    }
    if (resolved !== undefined) {
      params.append('resolved', resolved.toString());
    }

    return this.makeRequest<DeviceAlert[]>(`/devices/alerts?${params}`);
  }

  async acknowledgeAlert(alertId: string): Promise<DeviceAlert> {
    return this.makeRequest<DeviceAlert>(`/devices/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  }

  async resolveAlert(alertId: string, resolutionNotes?: string): Promise<DeviceAlert> {
    return this.makeRequest<DeviceAlert>(`/devices/alerts/${alertId}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ resolution_notes: resolutionNotes }),
    });
  }

  async getDeviceActivity(
    deviceId?: string,
    activityType?: string[],
    limit = 100
  ): Promise<DeviceActivity[]> {
    const params = new URLSearchParams({
      limit: limit.toString(),
    });

    if (deviceId) {
      params.append('device_id', deviceId);
    }
    if (activityType?.length) {
      activityType.forEach(type => params.append('activity_type', type));
    }

    return this.makeRequest<DeviceActivity[]>(`/devices/activity?${params}`);
  }

  // Metrics and Analytics
  async getDeviceMetrics(): Promise<DeviceMetricsResponse> {
    return this.makeRequest<DeviceMetricsResponse>('/devices/metrics');
  }

  async getDeviceMetricsHistory(
    startTime: string,
    endTime: string,
    granularity: 'hour' | 'day' | 'week' = 'hour'
  ): Promise<DeviceMetricsResponse[]> {
    const params = new URLSearchParams({
      start_time: startTime,
      end_time: endTime,
      granularity,
    });

    return this.makeRequest<DeviceMetricsResponse[]>(`/devices/metrics/history?${params}`);
  }

  // Bulk Operations
  async bulkUpdateDevices(
    deviceIds: string[],
    updates: Partial<DeviceConfigurationForm>
  ): Promise<{ success: string[]; failed: { device_id: string; error: string }[] }> {
    return this.makeRequest<{ success: string[]; failed: { device_id: string; error: string }[] }>(
      '/devices/bulk/update',
      {
        method: 'POST',
        body: JSON.stringify({ device_ids: deviceIds, updates }),
      }
    );
  }

  async bulkSuspendDevices(
    deviceIds: string[],
    suspendData: SuspendDeviceForm
  ): Promise<{ success: string[]; failed: { device_id: string; error: string }[] }> {
    return this.makeRequest<{ success: string[]; failed: { device_id: string; error: string }[] }>(
      '/devices/bulk/suspend',
      {
        method: 'POST',
        body: JSON.stringify({ device_ids: deviceIds, ...suspendData }),
      }
    );
  }

  async bulkResumeDevices(
    deviceIds: string[]
  ): Promise<{ success: string[]; failed: { device_id: string; error: string }[] }> {
    return this.makeRequest<{ success: string[]; failed: { device_id: string; error: string }[] }>(
      '/devices/bulk/resume',
      {
        method: 'POST',
        body: JSON.stringify({ device_ids: deviceIds }),
      }
    );
  }

  // Export and Import
  async exportDevices(
    filters?: DeviceFilters,
    format: 'csv' | 'json' = 'csv'
  ): Promise<Blob> {
    const params = new URLSearchParams({ format });

    if (filters) {
      if (filters.status?.length) {
        filters.status.forEach(status => params.append('status', status));
      }
      if (filters.device_type?.length) {
        filters.device_type.forEach(type => params.append('device_type', type));
      }
      // Add other filters as needed
    }

    const response = await fetch(`${this.baseUrl}/devices/export?${params}`, {
      headers: {
        'Authorization': `Bearer ${await this.getAuthToken()}`,
        'Accept': format === 'csv' ? 'text/csv' : 'application/json',
      },
    });

    if (!response.ok) {
      throw new DeviceApiError(
        `Export failed: ${response.statusText}`,
        response.status
      );
    }

    return response.blob();
  }
}

export { DeviceApiClient, DeviceApiError };
export type { ApiError };