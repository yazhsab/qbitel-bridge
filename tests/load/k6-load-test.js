/**
 * QBITEL - K6 Load Testing Script
 * 
 * Comprehensive load testing for API endpoints and system performance.
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiResponseTime = new Trend('api_response_time');
const successfulRequests = new Counter('successful_requests');
const failedRequests = new Counter('failed_requests');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 users
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '5m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95% of requests under 500ms, 99% under 1s
    'http_req_failed': ['rate<0.01'],  // Error rate under 1%
    'errors': ['rate<0.05'],           // Custom error rate under 5%
    'api_response_time': ['p(95)<600'],
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'qbitel_test_key';

// Helper function to create auth headers
function getAuthHeaders() {
  return {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json',
  };
}

// Helper function to encode data to base64
function toBase64(str) {
  return encoding.b64encode(str);
}

// Test data
const testPacketData = 'R0VUIC9hcGkvdjEvdGVzdCBIVFRQLzEuMQpIb3N0OiBleGFtcGxlLmNvbQoK'; // Base64 encoded HTTP request

export default function() {
  // Health check test
  group('Health Checks', function() {
    const healthRes = http.get(`${BASE_URL}/health`);
    
    check(healthRes, {
      'health check status is 200': (r) => r.status === 200,
      'health check has status field': (r) => JSON.parse(r.body).status !== undefined,
    });
    
    apiResponseTime.add(healthRes.timings.duration);
    errorRate.add(healthRes.status !== 200);
    
    if (healthRes.status === 200) {
      successfulRequests.add(1);
    } else {
      failedRequests.add(1);
    }
  });
  
  sleep(1);
  
  // Kubernetes health probes
  group('K8s Health Probes', function() {
    // Liveness probe
    const livenessRes = http.get(`${BASE_URL}/health/live`);
    check(livenessRes, {
      'liveness probe returns 200 or 503': (r) => [200, 503].includes(r.status),
    });
    
    // Readiness probe
    const readinessRes = http.get(`${BASE_URL}/health/ready`);
    check(readinessRes, {
      'readiness probe returns 200 or 503': (r) => [200, 503].includes(r.status),
    });
    
    // Startup probe
    const startupRes = http.get(`${BASE_URL}/health/startup`);
    check(startupRes, {
      'startup probe returns 200 or 503': (r) => [200, 503].includes(r.status),
    });
  });
  
  sleep(1);
  
  // Protocol discovery test
  group('Protocol Discovery', function() {
    const payload = JSON.stringify({
      packet_data: testPacketData,
      metadata: {
        source: 'load_test',
        timestamp: new Date().toISOString(),
      },
      enable_llm_analysis: false,
    });
    
    const res = http.post(
      `${BASE_URL}/api/v1/discover`,
      payload,
      { headers: getAuthHeaders() }
    );
    
    const success = check(res, {
      'protocol discovery status is 200': (r) => r.status === 200,
      'protocol discovery has protocol_type': (r) => {
        try {
          return JSON.parse(r.body).protocol_type !== undefined;
        } catch (e) {
          return false;
        }
      },
      'protocol discovery response time < 1s': (r) => r.timings.duration < 1000,
    });
    
    apiResponseTime.add(res.timings.duration);
    errorRate.add(!success);
    
    if (success) {
      successfulRequests.add(1);
    } else {
      failedRequests.add(1);
    }
  });
  
  sleep(1);
  
  // Field detection test
  group('Field Detection', function() {
    const payload = JSON.stringify({
      message_data: testPacketData,
      protocol_type: 'http',
      enable_llm_analysis: false,
    });
    
    const res = http.post(
      `${BASE_URL}/api/v1/detect-fields`,
      payload,
      { headers: getAuthHeaders() }
    );
    
    const success = check(res, {
      'field detection status is 200': (r) => r.status === 200,
      'field detection has detected_fields': (r) => {
        try {
          return JSON.parse(r.body).detected_fields !== undefined;
        } catch (e) {
          return false;
        }
      },
    });
    
    apiResponseTime.add(res.timings.duration);
    errorRate.add(!success);
    
    if (success) {
      successfulRequests.add(1);
    } else {
      failedRequests.add(1);
    }
  });
  
  sleep(2);
}

// Setup function - runs once before the test
export function setup() {
  console.log('Starting load test...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`API Key: ${API_KEY ? 'Configured' : 'Not configured'}`);
  
  // Verify service is available
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`Service not available. Health check returned: ${healthCheck.status}`);
  }
  
  return { startTime: new Date() };
}

// Teardown function - runs once after the test
export function teardown(data) {
  const endTime = new Date();
  const duration = (endTime - data.startTime) / 1000;
  console.log(`Load test completed in ${duration} seconds`);
}