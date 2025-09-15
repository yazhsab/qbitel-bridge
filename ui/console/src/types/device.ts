export type DeviceStatus = 
  | 'pending'
  | 'enrolling'
  | 'active'
  | 'inactive'
  | 'suspended'
  | 'decommissioned'
  | 'error';

export type DeviceType = 
  | 'gateway'
  | 'endpoint'
  | 'sensor'
  | 'actuator'
  | 'controller';

export type ComplianceStatus = 
  | 'compliant'
  | 'non_compliant'
  | 'unknown'
  | 'checking';

export type HealthStatus = 
  | 'healthy'
  | 'unhealthy'
  | 'unknown'
  | 'checking';

export interface Device {
  id: string;
  name: string;
  organization_id: string;
  device_type: DeviceType;
  status: DeviceStatus;
  manufacturer: string;
  model: string;
  serial_number: string;
  firmware_version: string;
  hardware_version: string;
  public_key?: string;
  certificate_id?: string;
  tpm_endorsement_key?: string;
  tpm_attestation_key?: string;
  enrolled_at: string;
  last_seen: string;
  last_health_check: string;
  last_policy_update: string;
  last_compliance_check: string;
  configuration: Record<string, any>;
  policy_set: string;
  tags: string[];
  compliance_status: ComplianceStatus;
  health_status: HealthStatus;
  capabilities: string[];
  metadata: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface DeviceFilters {
  status?: DeviceStatus[];
  device_type?: DeviceType[];
  manufacturer?: string[];
  tags?: string[];
  compliance_status?: ComplianceStatus[];
  health_status?: HealthStatus[];
}

export interface EnrollmentRequest {
  device_id: string;
  device_name: string;
  organization_id: string;
  device_type: DeviceType;
  manufacturer: string;
  model: string;
  serial_number: string;
  firmware_version: string;
  hardware_version: string;
  configuration: Record<string, any>;
  tags: string[];
  capabilities: string[];
  metadata: Record<string, string>;
}

export interface EnrollmentSession {
  id: string;
  device_id: string;
  challenge: string;
  status: 'pending' | 'challenged' | 'verifying' | 'approved' | 'rejected' | 'expired';
  created_at: string;
  expires_at: string;
  metadata: Record<string, string>;
}

export interface AttestationData {
  quote: string;
  signature: string;
  pcr_values: Record<number, string>;
  event_log: string;
  ek_cert: string;
  ak_cert: string;
  nonce: string;
  timestamp: string;
}

export interface DeviceCertificate {
  id: string;
  device_id: string;
  certificate: string;
  serial_number: string;
  subject: string;
  issuer: string;
  not_before: string;
  not_after: string;
  key_usage: string[];
  status: 'active' | 'expired' | 'revoked' | 'suspended';
  created_at: string;
  revoked_at?: string;
  revocation_reason?: string;
}

export interface DevicePolicy {
  id: string;
  device_id: string;
  policy_set: string;
  version: string;
  policies: Record<string, any>;
  signature: string;
  created_at: string;
  updated_at: string;
  applied_at?: string;
}

export interface DeviceMetrics {
  total_devices: number;
  active_devices: number;
  inactive_devices: number;
  suspended_devices: number;
  decommissioned_devices: number;
  compliant_devices: number;
  non_compliant_devices: number;
  healthy_devices: number;
  unhealthy_devices: number;
  devices_by_type: Record<DeviceType, number>;
  devices_by_manufacturer: Record<string, number>;
  enrollment_rate_24h: number;
  compliance_rate: number;
  health_rate: number;
}

export interface DeviceActivity {
  device_id: string;
  activity_type: 'enrollment' | 'status_change' | 'policy_update' | 'compliance_check' | 'health_check';
  timestamp: string;
  details: Record<string, any>;
  user_id?: string;
  user_email?: string;
}

export interface ComplianceViolation {
  device_id: string;
  violation_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  detected_at: string;
  resolved_at?: string;
  resolution_notes?: string;
}

export interface DeviceAlert {
  id: string;
  device_id: string;
  alert_type: 'health' | 'compliance' | 'security' | 'performance';
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  description: string;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  acknowledged_by?: string;
  resolved_by?: string;
}

// API Response types
export interface ListDevicesResponse {
  devices: Device[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface DeviceDetailsResponse {
  device: Device;
  certificate?: DeviceCertificate;
  policies: DevicePolicy[];
  recent_activity: DeviceActivity[];
  alerts: DeviceAlert[];
  compliance_violations: ComplianceViolation[];
}

export interface DeviceMetricsResponse {
  metrics: DeviceMetrics;
  timestamp: string;
}

// Form types
export interface DeviceConfigurationForm {
  name: string;
  tags: string[];
  configuration: Record<string, any>;
  policy_set: string;
}

export interface SuspendDeviceForm {
  reason: string;
  duration?: number; // in hours, optional for indefinite suspension
  notify_device: boolean;
}

export interface DecommissionDeviceForm {
  reason: string;
  wipe_data: boolean;
  revoke_certificates: boolean;
  notify_device: boolean;
}

// Utility types
export type DeviceStatusColor = 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning';

export const DEVICE_STATUS_COLORS: Record<DeviceStatus, DeviceStatusColor> = {
  pending: 'warning',
  enrolling: 'info',
  active: 'success',
  inactive: 'warning',
  suspended: 'error',
  decommissioned: 'default',
  error: 'error',
};

export const COMPLIANCE_STATUS_COLORS: Record<ComplianceStatus, DeviceStatusColor> = {
  compliant: 'success',
  non_compliant: 'error',
  unknown: 'warning',
  checking: 'info',
};

export const HEALTH_STATUS_COLORS: Record<HealthStatus, DeviceStatusColor> = {
  healthy: 'success',
  unhealthy: 'error',
  unknown: 'warning',
  checking: 'info',
};

// Constants
export const DEVICE_TYPES: { value: DeviceType; label: string }[] = [
  { value: 'gateway', label: 'Gateway' },
  { value: 'endpoint', label: 'Endpoint' },
  { value: 'sensor', label: 'Sensor' },
  { value: 'actuator', label: 'Actuator' },
  { value: 'controller', label: 'Controller' },
];

export const DEVICE_STATUSES: { value: DeviceStatus; label: string }[] = [
  { value: 'pending', label: 'Pending' },
  { value: 'enrolling', label: 'Enrolling' },
  { value: 'active', label: 'Active' },
  { value: 'inactive', label: 'Inactive' },
  { value: 'suspended', label: 'Suspended' },
  { value: 'decommissioned', label: 'Decommissioned' },
  { value: 'error', label: 'Error' },
];

export const COMPLIANCE_STATUSES: { value: ComplianceStatus; label: string }[] = [
  { value: 'compliant', label: 'Compliant' },
  { value: 'non_compliant', label: 'Non-Compliant' },
  { value: 'unknown', label: 'Unknown' },
  { value: 'checking', label: 'Checking' },
];

export const HEALTH_STATUSES: { value: HealthStatus; label: string }[] = [
  { value: 'healthy', label: 'Healthy' },
  { value: 'unhealthy', label: 'Unhealthy' },
  { value: 'unknown', label: 'Unknown' },
  { value: 'checking', label: 'Checking' },
];