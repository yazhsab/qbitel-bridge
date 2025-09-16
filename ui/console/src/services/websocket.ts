import { useEffect, useRef, useCallback, useState } from 'react';

export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: number;
  source?: string;
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export enum WebSocketReadyState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3,
}

export class EnterpriseWebSocketClient {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private subscriptions = new Map<string, Set<(message: WebSocketMessage) => void>>();
  private isManualClose = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectAttempts: 10,
      reconnectInterval: 3000,
      heartbeatInterval: 30000,
      ...config,
    };
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      this.ws = new WebSocket(this.config.url, this.config.protocols);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.handleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = (event) => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.isManualClose = false;
      this.startHeartbeat();
      this.flushMessageQueue();
      this.config.onOpen?.(event);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      this.stopHeartbeat();
      this.config.onClose?.(event);
      
      if (!this.isManualClose && this.shouldReconnect()) {
        this.handleReconnect();
      }
    };

    this.ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      this.config.onError?.(event);
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
  }

  private handleMessage(message: WebSocketMessage): void {
    // Handle heartbeat responses
    if (message.type === 'pong') {
      return;
    }

    // Dispatch to global handler
    this.config.onMessage?.(message);

    // Dispatch to type-specific subscribers
    const subscribers = this.subscriptions.get(message.type);
    if (subscribers) {
      subscribers.forEach(callback => callback(message));
    }

    // Dispatch to wildcard subscribers
    const wildcardSubscribers = this.subscriptions.get('*');
    if (wildcardSubscribers) {
      wildcardSubscribers.forEach(callback => callback(message));
    }
  }

  private shouldReconnect(): boolean {
    return this.reconnectAttempts < (this.config.reconnectAttempts || 10);
  }

  private handleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnectInterval! * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    if (this.config.heartbeatInterval && this.config.heartbeatInterval > 0) {
      this.heartbeatTimer = setInterval(() => {
        this.send({ type: 'ping', payload: {}, timestamp: Date.now() });
      }, this.config.heartbeatInterval);
    }
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  send(message: Omit<WebSocketMessage, 'timestamp'>): void {
    const fullMessage: WebSocketMessage = {
      ...message,
      timestamp: Date.now(),
    };

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.sendMessage(fullMessage);
    } else {
      // Queue message for later sending
      this.messageQueue.push(fullMessage);
      if (this.messageQueue.length > 100) {
        this.messageQueue.shift(); // Remove oldest message
      }
    }
  }

  private sendMessage(message: WebSocketMessage): void {
    try {
      this.ws?.send(JSON.stringify(message));
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
    }
  }

  subscribe(messageType: string, callback: (message: WebSocketMessage) => void): () => void {
    if (!this.subscriptions.has(messageType)) {
      this.subscriptions.set(messageType, new Set());
    }
    
    this.subscriptions.get(messageType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const subscribers = this.subscriptions.get(messageType);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          this.subscriptions.delete(messageType);
        }
      }
    };
  }

  unsubscribe(messageType: string, callback?: (message: WebSocketMessage) => void): void {
    if (callback) {
      this.subscriptions.get(messageType)?.delete(callback);
    } else {
      this.subscriptions.delete(messageType);
    }
  }

  getReadyState(): WebSocketReadyState {
    return this.ws?.readyState ?? WebSocketReadyState.CLOSED;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  close(code?: number, reason?: string): void {
    this.isManualClose = true;
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close(code, reason);
    }
  }

  destroy(): void {
    this.close();
    this.subscriptions.clear();
    this.messageQueue.length = 0;
  }
}

// React Hook for WebSocket
export interface UseWebSocketOptions extends Omit<WebSocketConfig, 'url'> {
  enabled?: boolean;
}

export interface UseWebSocketReturn {
  sendMessage: (message: Omit<WebSocketMessage, 'timestamp'>) => void;
  subscribe: (messageType: string, callback: (message: WebSocketMessage) => void) => () => void;
  readyState: WebSocketReadyState;
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const { enabled = true, ...config } = options;
  const clientRef = useRef<EnterpriseWebSocketClient | null>(null);
  const [readyState, setReadyState] = useState<WebSocketReadyState>(WebSocketReadyState.CLOSED);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');

  const updateReadyState = useCallback(() => {
    if (clientRef.current) {
      const state = clientRef.current.getReadyState();
      setReadyState(state);
      
      switch (state) {
        case WebSocketReadyState.CONNECTING:
          setConnectionStatus('connecting');
          break;
        case WebSocketReadyState.OPEN:
          setConnectionStatus('connected');
          break;
        case WebSocketReadyState.CLOSED:
        case WebSocketReadyState.CLOSING:
          setConnectionStatus('disconnected');
          break;
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const client = new EnterpriseWebSocketClient({
      url,
      ...config,
      onOpen: (event) => {
        updateReadyState();
        config.onOpen?.(event);
      },
      onClose: (event) => {
        updateReadyState();
        config.onClose?.(event);
      },
      onError: (event) => {
        setConnectionStatus('error');
        config.onError?.(event);
      },
      onMessage: (message) => {
        setLastMessage(message);
        config.onMessage?.(message);
      },
    });

    clientRef.current = client;
    client.connect();

    // Check connection status periodically
    const statusInterval = setInterval(updateReadyState, 1000);

    return () => {
      clearInterval(statusInterval);
      client.destroy();
      clientRef.current = null;
    };
  }, [url, enabled, updateReadyState]);

  const sendMessage = useCallback((message: Omit<WebSocketMessage, 'timestamp'>) => {
    clientRef.current?.send(message);
  }, []);

  const subscribe = useCallback((messageType: string, callback: (message: WebSocketMessage) => void) => {
    return clientRef.current?.subscribe(messageType, callback) || (() => {});
  }, []);

  return {
    sendMessage,
    subscribe,
    readyState,
    isConnected: readyState === WebSocketReadyState.OPEN,
    lastMessage,
    connectionStatus,
  };
}

// Global WebSocket Manager for the entire application
export class GlobalWebSocketManager {
  private static instance: GlobalWebSocketManager;
  private clients = new Map<string, EnterpriseWebSocketClient>();
  private defaultClient: EnterpriseWebSocketClient | null = null;

  private constructor() {}

  static getInstance(): GlobalWebSocketManager {
    if (!GlobalWebSocketManager.instance) {
      GlobalWebSocketManager.instance = new GlobalWebSocketManager();
    }
    return GlobalWebSocketManager.instance;
  }

  createClient(name: string, config: WebSocketConfig): EnterpriseWebSocketClient {
    const client = new EnterpriseWebSocketClient(config);
    this.clients.set(name, client);
    
    if (!this.defaultClient || name === 'default') {
      this.defaultClient = client;
    }
    
    return client;
  }

  getClient(name?: string): EnterpriseWebSocketClient | null {
    if (name) {
      return this.clients.get(name) || null;
    }
    return this.defaultClient;
  }

  removeClient(name: string): void {
    const client = this.clients.get(name);
    if (client) {
      client.destroy();
      this.clients.delete(name);
      
      if (this.defaultClient === client) {
        this.defaultClient = this.clients.values().next().value || null;
      }
    }
  }

  broadcast(message: Omit<WebSocketMessage, 'timestamp'>): void {
    this.clients.forEach(client => {
      if (client.isConnected()) {
        client.send(message);
      }
    });
  }

  destroy(): void {
    this.clients.forEach(client => client.destroy());
    this.clients.clear();
    this.defaultClient = null;
  }
}

// Message type constants for different components
export const MessageTypes = {
  // Protocol Discovery
  PROTOCOL_DISCOVERED: 'protocol_discovered',
  PROTOCOL_UPDATED: 'protocol_updated',
  PROTOCOL_REMOVED: 'protocol_removed',
  PROTOCOL_METRICS: 'protocol_metrics',
  
  // AI Model Monitoring
  MODEL_METRICS: 'model_metrics',
  MODEL_ALERT: 'model_alert',
  MODEL_TRAINING_COMPLETE: 'model_training_complete',
  MODEL_DEPLOYED: 'model_deployed',
  
  // Threat Intelligence
  THREAT_DETECTED: 'threat_detected',
  THREAT_RESOLVED: 'threat_resolved',
  IOC_UPDATED: 'ioc_updated',
  SECURITY_ALERT: 'security_alert',
  
  // Analytics
  ANALYTICS_UPDATE: 'analytics_update',
  ANOMALY_DETECTED: 'anomaly_detected',
  PREDICTION_UPDATED: 'prediction_updated',
  
  // System
  HEARTBEAT: 'heartbeat',
  SYSTEM_STATUS: 'system_status',
  USER_ACTION: 'user_action',
  
  // General
  NOTIFICATION: 'notification',
  ERROR: 'error',
  STATUS_UPDATE: 'status_update',
} as const;

// Utility functions
export function createMessage(
  type: string, 
  payload: any, 
  source?: string
): Omit<WebSocketMessage, 'timestamp'> {
  return {
    type,
    payload,
    source,
  };
}

export function isMessageType(message: WebSocketMessage, type: string): boolean {
  return message.type === type;
}

export function filterMessagesByType(messages: WebSocketMessage[], type: string): WebSocketMessage[] {
  return messages.filter(msg => msg.type === type);
}

// Default WebSocket configuration
export const defaultWebSocketConfig: Partial<WebSocketConfig> = {
  reconnectAttempts: 10,
  reconnectInterval: 3000,
  heartbeatInterval: 30000,
};