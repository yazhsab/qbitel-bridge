import { DeviceApiClient, DeviceApiError } from './devices';
import { config, CacheManager, ErrorTracker, PerformanceMonitor } from '../config/production';

export interface RealTimeSubscription {
  topic: string;
  callback: (data: any) => void;
  active: boolean;
}

export interface ApiResponse<T> {
  data: T;
  cached: boolean;
  timestamp: number;
}

export class EnhancedApiClient extends DeviceApiClient {
  private wsConnections: Map<string, WebSocket> = new Map();
  private subscriptions: Map<string, RealTimeSubscription[]> = new Map();
  private requestQueue: Array<() => Promise<any>> = [];
  private processing = false;

  constructor(getAuthToken: () => Promise<string | null>) {
    super(getAuthToken);
  }

  // Enhanced request method with caching and performance monitoring
  async makeEnhancedRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    cacheKey?: string,
    cacheTTL?: number
  ): Promise<ApiResponse<T>> {
    const endTimer = PerformanceMonitor.startTimer(`API.${endpoint}`);
    
    try {
      // Check cache first
      if (cacheKey && CacheManager.has(cacheKey)) {
        endTimer();
        const cachedData = CacheManager.get(cacheKey);
        return {
          data: cachedData,
          cached: true,
          timestamp: Date.now(),
        };
      }

      // Add request to queue if rate limiting is needed
      if (this.shouldQueue(endpoint)) {
        return this.queueRequest(() => this.makeRequest<T>(endpoint, options));
      }

      const response = await this.makeRequest<T>(endpoint, options);

      // Cache successful responses
      if (cacheKey && response) {
        CacheManager.set(cacheKey, response, cacheTTL);
      }

      endTimer();
      return {
        data: response,
        cached: false,
        timestamp: Date.now(),
      };

    } catch (error) {
      endTimer();
      ErrorTracker.captureError(error as Error, { endpoint, options });
      throw error;
    }
  }

  // Real-time WebSocket management
  async establishWebSocketConnection(
    endpoint: string,
    protocols?: string[]
  ): Promise<WebSocket> {
    const wsUrl = `${config.websocket.url}${endpoint}`;
    const token = await this.getAuthToken();

    const ws = new WebSocket(wsUrl, protocols);
    
    // Add authentication
    ws.onopen = () => {
      if (token) {
        ws.send(JSON.stringify({
          type: 'auth',
          token: token,
        }));
      }
    };

    // Handle connection errors
    ws.onerror = (error) => {
      ErrorTracker.captureError(new Error('WebSocket error'), { endpoint, error });
    };

    // Handle reconnection
    ws.onclose = (event) => {
      if (event.code !== 1000) { // Not a normal closure
        setTimeout(() => {
          this.establishWebSocketConnection(endpoint, protocols);
        }, config.websocket.reconnectDelay);
      }
    };

    this.wsConnections.set(endpoint, ws);
    return ws;
  }

  // Subscribe to real-time updates
  async subscribeToRealTimeUpdates<T>(
    topic: string,
    callback: (data: T) => void,
    endpoint: string = '/dashboard'
  ): Promise<void> {
    let ws = this.wsConnections.get(endpoint);
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      ws = await this.establishWebSocketConnection(endpoint);
    }

    const subscription: RealTimeSubscription = {
      topic,
      callback,
      active: true,
    };

    if (!this.subscriptions.has(endpoint)) {
      this.subscriptions.set(endpoint, []);
    }
    
    this.subscriptions.get(endpoint)!.push(subscription);

    // Set up message handler
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.topic === topic) {
          const activeSubscriptions = this.subscriptions.get(endpoint)?.filter(s => 
            s.topic === topic && s.active
          ) || [];
          
          activeSubscriptions.forEach(sub => {
            try {
              sub.callback(message.data);
            } catch (error) {
              ErrorTracker.captureError(error as Error, { topic, subscription });
            }
          });
        }
      } catch (error) {
        ErrorTracker.captureError(error as Error, { endpoint, event });
      }
    };

    // Send subscription message
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'subscribe',
        topic,
      }));
    }
  }

  // Unsubscribe from real-time updates
  unsubscribeFromRealTimeUpdates(topic: string, endpoint: string = '/dashboard'): void {
    const subscriptions = this.subscriptions.get(endpoint);
    
    if (subscriptions) {
      subscriptions.forEach(sub => {
        if (sub.topic === topic) {
          sub.active = false;
        }
      });
      
      // Clean up inactive subscriptions
      const activeSubscriptions = subscriptions.filter(s => s.active);
      this.subscriptions.set(endpoint, activeSubscriptions);
    }

    const ws = this.wsConnections.get(endpoint);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'unsubscribe',
        topic,
      }));
    }
  }

  // Enhanced dashboard metrics with caching
  async getDashboardMetrics(useCache = true): Promise<ApiResponse<any>> {
    const cacheKey = 'dashboard.metrics';
    const cacheTTL = 30000; // 30 seconds

    return this.makeEnhancedRequest(
      '/dashboard/metrics',
      { method: 'GET' },
      useCache ? cacheKey : undefined,
      cacheTTL
    );
  }

  // Protocol analytics with real-time capabilities
  async getProtocolAnalytics(
    timeRange: string = '24h',
    useCache = true
  ): Promise<ApiResponse<any>> {
    const cacheKey = `protocol.analytics.${timeRange}`;
    const cacheTTL = timeRange === '1h' ? 60000 : 300000; // 1 min or 5 min

    return this.makeEnhancedRequest(
      `/protocols/analytics?range=${timeRange}`,
      { method: 'GET' },
      useCache ? cacheKey : undefined,
      cacheTTL
    );
  }

  // AI model metrics
  async getAIModelMetrics(useCache = true): Promise<ApiResponse<any>> {
    const cacheKey = 'ai.models.metrics';
    const cacheTTL = 60000; // 1 minute

    return this.makeEnhancedRequest(
      '/ai/models/metrics',
      { method: 'GET' },
      useCache ? cacheKey : undefined,
      cacheTTL
    );
  }

  // Threat intelligence data
  async getThreatIntelligence(
    severity?: string[],
    useCache = true
  ): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (severity?.length) {
      severity.forEach(s => params.append('severity', s));
    }

    const cacheKey = `threat.intelligence.${params.toString()}`;
    const cacheTTL = 120000; // 2 minutes

    return this.makeEnhancedRequest(
      `/threat-intelligence?${params}`,
      { method: 'GET' },
      useCache ? cacheKey : undefined,
      cacheTTL
    );
  }

  // Advanced analytics with predictive data
  async getAdvancedAnalytics(
    metric: string,
    timeRange: string = '24h',
    includePredictions = true,
    useCache = true
  ): Promise<ApiResponse<any>> {
    const params = new URLSearchParams({
      metric,
      range: timeRange,
      predictions: includePredictions.toString(),
    });

    const cacheKey = `analytics.${metric}.${timeRange}.${includePredictions}`;
    const cacheTTL = 180000; // 3 minutes

    return this.makeEnhancedRequest(
      `/analytics?${params}`,
      { method: 'GET' },
      useCache ? cacheKey : undefined,
      cacheTTL
    );
  }

  // Batch operations with progress tracking
  async performBatchOperation<T>(
    operation: string,
    items: string[],
    batchSize = 10,
    onProgress?: (completed: number, total: number) => void
  ): Promise<T[]> {
    const results: T[] = [];
    const total = items.length;
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      
      try {
        const batchResults = await this.makeRequest<T[]>(
          `/batch/${operation}`,
          {
            method: 'POST',
            body: JSON.stringify({ items: batch }),
          }
        );
        
        results.push(...batchResults);
        
        if (onProgress) {
          onProgress(Math.min(i + batchSize, total), total);
        }
      } catch (error) {
        ErrorTracker.captureError(error as Error, { operation, batch });
        throw error;
      }
    }
    
    return results;
  }

  // Health check with retry logic
  async healthCheck(retries = 3): Promise<boolean> {
    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        await this.makeRequest('/health');
        return true;
      } catch (error) {
        if (attempt === retries - 1) {
          ErrorTracker.captureError(error as Error, { context: 'healthCheck' });
          return false;
        }
        
        // Wait before retry
        await new Promise(resolve => 
          setTimeout(resolve, Math.pow(2, attempt) * 1000)
        );
      }
    }
    
    return false;
  }

  // Request queuing for rate limiting
  private shouldQueue(endpoint: string): boolean {
    // Implement rate limiting logic based on endpoint
    const rateLimitedEndpoints = ['/analytics', '/threat-intelligence'];
    return rateLimitedEndpoints.some(pattern => endpoint.includes(pattern));
  }

  private async queueRequest<T>(request: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await request();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      
      this.processQueue();
    });
  }

  private async processQueue(): Promise<void> {
    if (this.processing || this.requestQueue.length === 0) {
      return;
    }
    
    this.processing = true;
    
    while (this.requestQueue.length > 0) {
      const request = this.requestQueue.shift()!;
      await request();
      
      // Add delay between requests
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    this.processing = false;
  }

  // Cleanup method
  cleanup(): void {
    // Close all WebSocket connections
    this.wsConnections.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Client cleanup');
      }
    });
    
    this.wsConnections.clear();
    this.subscriptions.clear();
    this.requestQueue.length = 0;
  }

  // Get connection status
  getConnectionStatus(): Record<string, string> {
    const status: Record<string, string> = {};
    
    this.wsConnections.forEach((ws, endpoint) => {
      switch (ws.readyState) {
        case WebSocket.CONNECTING:
          status[endpoint] = 'connecting';
          break;
        case WebSocket.OPEN:
          status[endpoint] = 'connected';
          break;
        case WebSocket.CLOSING:
          status[endpoint] = 'closing';
          break;
        case WebSocket.CLOSED:
          status[endpoint] = 'closed';
          break;
        default:
          status[endpoint] = 'unknown';
      }
    });
    
    return status;
  }

  // Get performance metrics
  getPerformanceMetrics(): Record<string, any> {
    return PerformanceMonitor.getAllMetrics();
  }

  // Get error summary
  getErrorSummary(): Array<any> {
    return ErrorTracker.getErrors();
  }
}

// Factory function to create enhanced API client
export function createEnhancedApiClient(
  getAuthToken: () => Promise<string | null>
): EnhancedApiClient {
  return new EnhancedApiClient(getAuthToken);
}

export default EnhancedApiClient;