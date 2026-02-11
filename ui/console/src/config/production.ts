// Production Configuration System
export interface ProductionConfig {
  api: {
    baseUrl: string;
    timeout: number;
    retryAttempts: number;
    retryDelay: number;
  };
  websocket: {
    url: string;
    reconnectAttempts: number;
    reconnectDelay: number;
    heartbeatInterval: number;
  };
  cache: {
    defaultTTL: number;
    maxSize: number;
    enableServiceWorker: boolean;
  };
  performance: {
    enableVirtualization: boolean;
    chunkSize: number;
    debounceDelay: number;
    enableLazyLoading: boolean;
  };
  monitoring: {
    enableErrorTracking: boolean;
    enablePerformanceMonitoring: boolean;
    sampleRate: number;
  };
  security: {
    enableCSP: boolean;
    enableSRI: boolean;
    sessionTimeout: number;
  };
}

const production: ProductionConfig = {
  api: {
    baseUrl: process.env.REACT_APP_API_BASE_URL || 'https://api.qbitelai.local',
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
  },
  websocket: {
    url: process.env.REACT_APP_WS_URL || 'wss://api.qbitelai.local/ws',
    reconnectAttempts: 5,
    reconnectDelay: 1000,
    heartbeatInterval: 30000,
  },
  cache: {
    defaultTTL: 300000, // 5 minutes
    maxSize: 100,
    enableServiceWorker: true,
  },
  performance: {
    enableVirtualization: true,
    chunkSize: 50,
    debounceDelay: 300,
    enableLazyLoading: true,
  },
  monitoring: {
    enableErrorTracking: true,
    enablePerformanceMonitoring: true,
    sampleRate: 0.1,
  },
  security: {
    enableCSP: true,
    enableSRI: true,
    sessionTimeout: 3600000, // 1 hour
  },
};

const development: ProductionConfig = {
  ...production,
  api: {
    ...production.api,
    baseUrl: 'http://localhost:8080',
  },
  websocket: {
    ...production.websocket,
    url: 'ws://localhost:8080/ws',
  },
  cache: {
    ...production.cache,
    enableServiceWorker: false,
  },
  monitoring: {
    ...production.monitoring,
    enableErrorTracking: false,
    sampleRate: 1.0,
  },
};

export const config: ProductionConfig = 
  process.env.NODE_ENV === 'production' ? production : development;

// Performance monitoring utilities
export class PerformanceMonitor {
  private static metrics: Map<string, number[]> = new Map();

  static startTimer(name: string): () => void {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      if (!this.metrics.has(name)) {
        this.metrics.set(name, []);
      }
      
      this.metrics.get(name)!.push(duration);
      
      // Keep only last 100 measurements
      const measurements = this.metrics.get(name)!;
      if (measurements.length > 100) {
        measurements.shift();
      }
      
      if (config.monitoring.enablePerformanceMonitoring) {
        console.debug(`Performance: ${name} took ${duration.toFixed(2)}ms`);
      }
    };
  }

  static getMetrics(name: string) {
    const measurements = this.metrics.get(name) || [];
    if (measurements.length === 0) return null;

    const sum = measurements.reduce((a, b) => a + b, 0);
    const avg = sum / measurements.length;
    const min = Math.min(...measurements);
    const max = Math.max(...measurements);

    return { avg, min, max, count: measurements.length };
  }

  static getAllMetrics() {
    const result: Record<string, any> = {};
    for (const [name, _] of this.metrics) {
      result[name] = this.getMetrics(name);
    }
    return result;
  }
}

// Error tracking utilities
export class ErrorTracker {
  private static errors: Array<{
    message: string;
    stack?: string;
    timestamp: Date;
    context?: any;
  }> = [];

  static captureError(error: Error, context?: any) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date(),
      context,
    };

    this.errors.push(errorInfo);

    // Keep only last 50 errors
    if (this.errors.length > 50) {
      this.errors.shift();
    }

    if (config.monitoring.enableErrorTracking) {
      console.error('Captured error:', errorInfo);
      
      // Send to monitoring service in production
      if (process.env.NODE_ENV === 'production') {
        this.sendToMonitoringService(errorInfo);
      }
    }
  }

  private static async sendToMonitoringService(errorInfo: any) {
    try {
      // Implementation would depend on monitoring service (e.g., Sentry, DataDog)
      await fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorInfo),
      });
    } catch (e) {
      console.warn('Failed to send error to monitoring service:', e);
    }
  }

  static getErrors() {
    return this.errors;
  }

  static clearErrors() {
    this.errors = [];
  }
}

// Cache management system
export class CacheManager {
  private static cache: Map<string, {
    data: any;
    timestamp: number;
    ttl: number;
  }> = new Map();

  static set(key: string, data: any, ttl: number = config.cache.defaultTTL) {
    // Check if cache is at max size
    if (this.cache.size >= config.cache.maxSize) {
      // Remove oldest entry
      const oldestKey = Array.from(this.cache.keys())[0];
      this.cache.delete(oldestKey);
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  static get(key: string): any | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }

    // Check if entry has expired
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  static has(key: string): boolean {
    return this.get(key) !== null;
  }

  static clear() {
    this.cache.clear();
  }

  static getStats() {
    return {
      size: this.cache.size,
      maxSize: config.cache.maxSize,
    };
  }
}

// Debounce utility for performance
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number = config.performance.debounceDelay
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
}

// Throttle utility for performance
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastExecTime = 0;

  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastExecTime >= delay) {
      func.apply(null, args);
      lastExecTime = now;
    }
  };
}

export default config;