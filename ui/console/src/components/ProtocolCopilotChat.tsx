import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  TextField,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Chip,
  CircularProgress,
  Alert,
  Tooltip,
  Collapse,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Send as SendIcon,
  Psychology as CopilotIcon,
  Person as UserIcon,
  ContentCopy as CopyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Upload as UploadIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';

// Types
interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  queryType?: string;
  suggestions?: string[];
  sourceData?: Array<{ type: string; name?: string; confidence?: number }>;
  visualizations?: Array<{ type: string; title: string; data: any }>;
  metadata?: Record<string, any>;
  isTyping?: boolean;
}

interface CopilotChatProps {
  sessionId?: string;
  onSessionChange?: (sessionId: string) => void;
  apiBaseUrl?: string;
  wsBaseUrl?: string;
  maxHeight?: string;
  embedded?: boolean;
}

interface WebSocketMessage {
  type: string;
  data: Record<string, any>;
  timestamp?: string;
  correlation_id?: string;
}

const ProtocolCopilotChat: React.FC<CopilotChatProps> = ({
  sessionId: initialSessionId,
  onSessionChange,
  apiBaseUrl = '/api/v1/copilot',
  wsBaseUrl = 'ws://localhost:8000/api/v1/copilot/ws',
  maxHeight = '600px',
  embedded = false,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State management
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState(initialSessionId || '');
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSourceData, setShowSourceData] = useState<Record<string, boolean>>({});
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [packetData, setPacketData] = useState<string>('');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      const token = localStorage.getItem('auth_token'); // Adjust based on your auth implementation
      const wsUrl = `${wsBaseUrl}?token=${token}`;
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setWsConnected(true);
        setError(null);
        console.log('WebSocket connected to Protocol Copilot');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      wsRef.current.onclose = () => {
        setWsConnected(false);
        setIsTyping(false);
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error. Retrying...');
      };
      
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setError('Failed to connect to Protocol Copilot');
    }
  }, [wsBaseUrl]);
  
  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'response':
        setIsTyping(false);
        setIsLoading(false);
        
        const responseMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'assistant',
          content: message.data.response,
          timestamp: new Date(),
          confidence: message.data.confidence,
          queryType: message.data.query_type,
          suggestions: message.data.suggestions || [],
          sourceData: message.data.source_data || [],
          visualizations: message.data.visualizations || [],
          metadata: message.data.metadata || {}
        };
        
        setMessages(prev => [...prev, responseMessage]);
        
        // Update session ID if provided
        if (message.data.session_id && message.data.session_id !== sessionId) {
          setSessionId(message.data.session_id);
          onSessionChange?.(message.data.session_id);
        }
        break;
        
      case 'typing':
        setIsTyping(true);
        break;
        
      case 'error':
        setIsTyping(false);
        setIsLoading(false);
        setError(message.data.error || 'An error occurred');
        break;
        
      case 'pong':
        // Health check response
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }, [sessionId, onSessionChange]);
  
  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);
  
  // Send message via WebSocket
  const sendMessage = useCallback(async () => {
    if (!currentMessage.trim() || !wsConnected) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);
    setError(null);
    
    // Send via WebSocket
    const wsMessage: WebSocketMessage = {
      type: 'query',
      data: {
        query: currentMessage,
        session_id: sessionId,
        context: {},
        packet_data: packetData || undefined
      },
      correlation_id: userMessage.id
    };
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(wsMessage));
    } else {
      setError('Connection lost. Reconnecting...');
      setIsLoading(false);
      connectWebSocket();
    }
  }, [currentMessage, sessionId, packetData, wsConnected, connectWebSocket]);
  
  // Handle suggestion click
  const handleSuggestionClick = useCallback((suggestion: string) => {
    setCurrentMessage(suggestion);
    inputRef.current?.focus();
  }, []);
  
  // Handle source data toggle
  const toggleSourceData = useCallback((messageId: string) => {
    setShowSourceData(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  }, []);
  
  // Handle file upload for packet analysis
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result;
        if (typeof result === 'string') {
          // Convert to base64 if it's binary data
          const base64Data = btoa(result);
          setPacketData(base64Data);
          setUploadDialogOpen(false);
        }
      };
      reader.readAsBinaryString(file);
    }
  }, []);
  
  // Copy message content
  const copyToClipboard = useCallback((content: string) => {
    navigator.clipboard.writeText(content);
    // You could add a toast notification here
  }, []);
  
  // Render message
  const renderMessage = (message: ChatMessage) => (
    <motion.div
      key={message.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <ListItem alignItems="flex-start" sx={{ py: 1 }}>
        <Avatar
          sx={{
            bgcolor: message.type === 'user' ? 'primary.main' : 'secondary.main',
            mr: 2
          }}
        >
          {message.type === 'user' ? <UserIcon /> : <CopilotIcon />}
        </Avatar>
        
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <Typography variant="subtitle2" color="textSecondary">
              {message.type === 'user' ? 'You' : 'Protocol Copilot'}
            </Typography>
            <Typography variant="caption" color="textSecondary">
              {format(message.timestamp, 'HH:mm')}
            </Typography>
            {message.confidence && (
              <Chip
                label={`${Math.round(message.confidence * 100)}% confident`}
                size="small"
                color={message.confidence > 0.8 ? 'success' : 'warning'}
                variant="outlined"
              />
            )}
            {message.queryType && (
              <Chip
                label={message.queryType.replace('_', ' ')}
                size="small"
                variant="outlined"
              />
            )}
          </Box>
          
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', mb: 1 }}>
            {message.content}
          </Typography>
          
          {/* Action buttons */}
          <Box display="flex" gap={1} mb={1}>
            <Tooltip title="Copy message">
              <IconButton size="small" onClick={() => copyToClipboard(message.content)}>
                <CopyIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            {message.type === 'assistant' && (
              <>
                <Tooltip title="Helpful">
                  <IconButton size="small">
                    <ThumbUpIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Not helpful">
                  <IconButton size="small">
                    <ThumbDownIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </>
            )}
            {message.sourceData && message.sourceData.length > 0 && (
              <Button
                size="small"
                startIcon={showSourceData[message.id] ? <CollapseIcon /> : <ExpandIcon />}
                onClick={() => toggleSourceData(message.id)}
              >
                Sources ({message.sourceData.length})
              </Button>
            )}
          </Box>
          
          {/* Source data */}
          <Collapse in={showSourceData[message.id]}>
            <Box sx={{ pl: 2, borderLeft: '2px solid', borderColor: 'divider', mb: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Source Data:
              </Typography>
              {message.sourceData?.map((source, index) => (
                <Box key={index} display="flex" alignItems="center" gap={1} mb={0.5}>
                  <Chip
                    label={source.type}
                    size="small"
                    variant="outlined"
                  />
                  {source.name && (
                    <Typography variant="body2">{source.name}</Typography>
                  )}
                  {source.confidence && (
                    <Typography variant="caption" color="textSecondary">
                      ({Math.round(source.confidence * 100)}%)
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          </Collapse>
          
          {/* Suggestions */}
          {message.suggestions && message.suggestions.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="textSecondary" gutterBottom>
                Suggestions:
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {message.suggestions.map((suggestion, index) => (
                  <Chip
                    key={index}
                    label={suggestion}
                    size="small"
                    clickable
                    onClick={() => handleSuggestionClick(suggestion)}
                  />
                ))}
              </Box>
            </Box>
          )}
          
          {/* Visualizations placeholder */}
          {message.visualizations && message.visualizations.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="textSecondary">
                Visualizations available (would render charts here)
              </Typography>
            </Box>
          )}
        </Box>
      </ListItem>
    </motion.div>
  );
  
  return (
    <>
      <Card sx={{ height: embedded ? '100%' : maxHeight, display: 'flex', flexDirection: 'column' }}>
        <CardHeader
          title={
            <Box display="flex" alignItems="center" gap={1}>
              <CopilotIcon color="primary" />
              <Typography variant="h6">Protocol Intelligence Copilot</Typography>
              <Chip
                label={wsConnected ? 'Connected' : 'Connecting...'}
                size="small"
                color={wsConnected ? 'success' : 'warning'}
                variant="outlined"
              />
            </Box>
          }
          action={
            <Box display="flex" gap={1}>
              <Tooltip title="Upload packet data">
                <IconButton onClick={() => setUploadDialogOpen(true)}>
                  <UploadIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh connection">
                <IconButton onClick={connectWebSocket}>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Settings">
                <IconButton>
                  <SettingsIcon />
                </IconButton>
              </Tooltip>
            </Box>
          }
          sx={{ pb: 1 }}
        />
        
        {error && (
          <Alert severity="error" sx={{ mx: 2, mb: 1 }}>
            {error}
          </Alert>
        )}
        
        <CardContent sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', p: 0 }}>
          {/* Messages */}
          <List sx={{ flex: 1, overflow: 'auto', px: 2 }}>
            <AnimatePresence>
              {messages.map(renderMessage)}
              
              {/* Typing indicator */}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <ListItem>
                    <Avatar sx={{ bgcolor: 'secondary.main', mr: 2 }}>
                      <CopilotIcon />
                    </Avatar>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2" color="textSecondary">
                        Protocol Copilot is typing
                      </Typography>
                      <CircularProgress size={16} />
                    </Box>
                  </ListItem>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </List>
          
          {/* Input area */}
          <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            {packetData && (
              <Alert severity="info" sx={{ mb: 1 }}>
                Packet data uploaded ({packetData.length} bytes)
                <Button size="small" onClick={() => setPacketData('')} sx={{ ml: 1 }}>
                  Clear
                </Button>
              </Alert>
            )}
            
            <Box display="flex" gap={1}>
              <TextField
                ref={inputRef}
                fullWidth
                variant="outlined"
                placeholder="Ask me about protocols, security, compliance..."
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                disabled={!wsConnected || isLoading}
                multiline
                maxRows={3}
                size={isMobile ? 'small' : 'medium'}
              />
              <IconButton
                onClick={sendMessage}
                disabled={!currentMessage.trim() || !wsConnected || isLoading}
                color="primary"
                size={isMobile ? 'small' : 'medium'}
              >
                {isLoading ? <CircularProgress size={20} /> : <SendIcon />}
              </IconButton>
            </Box>
          </Box>
        </CardContent>
      </Card>
      
      {/* Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
        <DialogTitle>Upload Packet Data</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Upload packet capture files (PCAP, binary data) for analysis
          </Typography>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileUpload}
            style={{ marginTop: 16 }}
            accept=".pcap,.cap,.dump,*"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ProtocolCopilotChat;