use crate::parser::{Iso8583Message, Iso8583Parser};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error, info};

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("no route found for message: {0}")]
    NoRoute(String),
    #[error("invalid routing configuration: {0}")]
    InvalidConfig(String),
    #[error("routing rule error: {0}")]
    RuleError(String),
    #[error("destination unreachable: {0}")]
    DestinationUnreachable(String),
}

/// Message routing destination
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingDestination {
    /// Destination identifier
    pub id: String,
    /// Destination address (IP:port, hostname, etc.)
    pub address: String,
    /// Destination type (upstream, downstream, internal)
    pub destination_type: DestinationType,
    /// Load balancing weight
    pub weight: u32,
    /// Health status
    pub healthy: bool,
    /// Connection timeout in milliseconds
    pub timeout_ms: u32,
}

/// Destination types
#[derive(Debug, Clone, PartialEq)]
pub enum DestinationType {
    /// Upstream system (bank, processor)
    Upstream,
    /// Downstream system (ATM, POS)
    Downstream,
    /// Internal system (fraud detection, logging)
    Internal,
    /// Load balancer
    LoadBalancer,
}

/// Routing rule based on message content
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Rule identifier
    pub id: String,
    /// Rule priority (lower number = higher priority)
    pub priority: u32,
    /// MTI pattern to match
    pub mti_pattern: Option<String>,
    /// Processing code pattern to match
    pub processing_code_pattern: Option<String>,
    /// Acquiring institution ID pattern
    pub acquirer_pattern: Option<String>,
    /// Card acceptor ID pattern
    pub card_acceptor_pattern: Option<String>,
    /// Field-based conditions
    pub field_conditions: HashMap<u8, String>,
    /// Target destination
    pub destination: RoutingDestination,
    /// Rule enabled flag
    pub enabled: bool,
}

/// Load balancing strategy
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Hash-based (consistent hashing)
    HashBased,
    /// Random selection
    Random,
}

/// Load balancer state
#[derive(Debug)]
struct LoadBalancerState {
    /// Current round-robin index
    round_robin_index: usize,
    /// Connection counts per destination
    connection_counts: HashMap<String, u32>,
    /// Last hash values for consistent hashing
    hash_ring: Vec<(u64, String)>,
}

/// Message router for ISO-8583 messages
pub struct MessageRouter {
    /// Routing rules (sorted by priority)
    rules: Vec<RoutingRule>,
    /// Default destination
    default_destination: Option<RoutingDestination>,
    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
    /// Load balancer state
    lb_state: LoadBalancerState,
    /// Route cache for performance
    route_cache: HashMap<String, RoutingDestination>,
    /// Cache TTL in seconds
    cache_ttl: u64,
}

impl MessageRouter {
    /// Create a new message router
    pub fn new(load_balancing: LoadBalancingStrategy) -> Self {
        Self {
            rules: Vec::new(),
            default_destination: None,
            load_balancing,
            lb_state: LoadBalancerState {
                round_robin_index: 0,
                connection_counts: HashMap::new(),
                hash_ring: Vec::new(),
            },
            route_cache: HashMap::new(),
            cache_ttl: 300, // 5 minutes
        }
    }
    
    /// Add a routing rule
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
        // Sort rules by priority (lower number = higher priority)
        self.rules.sort_by_key(|r| r.priority);
        
        // Clear cache when rules change
        self.route_cache.clear();
        
        info!(rule_id = %self.rules.last().unwrap().id, "added routing rule");
    }
    
    /// Set default destination
    pub fn set_default_destination(&mut self, destination: RoutingDestination) {
        self.default_destination = Some(destination);
    }
    
    /// Route a message to appropriate destination
    pub fn route_message(
        &mut self,
        message: &Iso8583Message,
        parser: &Iso8583Parser,
    ) -> Result<RoutingDestination, RoutingError> {
        // Generate cache key
        let cache_key = self.generate_cache_key(message);
        
        // Check cache first
        if let Some(cached_dest) = self.route_cache.get(&cache_key) {
            debug!(cache_key = %cache_key, "using cached route");
            return Ok(cached_dest.clone());
        }
        
        // Find matching rule
        let destination = self.find_matching_rule(message, parser)?;
        
        // Apply load balancing if multiple destinations available
        let final_destination = self.apply_load_balancing(&destination)?;
        
        // Cache the result
        self.route_cache.insert(cache_key, final_destination.clone());
        
        info!(
            mti = %message.mti,
            destination = %final_destination.id,
            "routed message"
        );
        
        Ok(final_destination)
    }
    
    /// Generate cache key for a message
    fn generate_cache_key(&self, message: &Iso8583Message) -> String {
        let mut key = message.mti.clone();
        
        if let Some(proc_code) = message.get_processing_code() {
            key.push_str(":");
            key.push_str(proc_code);
        }
        
        if let Some(acquirer) = message.get_field(32) {
            key.push_str(":");
            key.push_str(acquirer);
        }
        
        key
    }
    
    /// Find the first matching routing rule
    fn find_matching_rule(
        &self,
        message: &Iso8583Message,
        parser: &Iso8583Parser,
    ) -> Result<RoutingDestination, RoutingError> {
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            
            if self.rule_matches(rule, message)? {
                debug!(rule_id = %rule.id, "matched routing rule");
                return Ok(rule.destination.clone());
            }
        }
        
        // No rule matched, use default destination
        self.default_destination.clone()
            .ok_or_else(|| RoutingError::NoRoute(format!(
                "no route found for MTI {} and no default destination configured",
                message.mti
            )))
    }
    
    /// Check if a rule matches a message
    fn rule_matches(&self, rule: &RoutingRule, message: &Iso8583Message) -> Result<bool, RoutingError> {
        // Check MTI pattern
        if let Some(mti_pattern) = &rule.mti_pattern {
            if !self.pattern_matches(mti_pattern, &message.mti) {
                return Ok(false);
            }
        }
        
        // Check processing code pattern
        if let Some(proc_pattern) = &rule.processing_code_pattern {
            if let Some(proc_code) = message.get_processing_code() {
                if !self.pattern_matches(proc_pattern, proc_code) {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Required field not present
            }
        }
        
        // Check acquirer pattern
        if let Some(acq_pattern) = &rule.acquirer_pattern {
            if let Some(acquirer) = message.get_field(32) {
                if !self.pattern_matches(acq_pattern, acquirer) {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Required field not present
            }
        }
        
        // Check card acceptor pattern
        if let Some(ca_pattern) = &rule.card_acceptor_pattern {
            if let Some(card_acceptor) = message.get_field(42) {
                if !self.pattern_matches(ca_pattern, card_acceptor) {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Required field not present
            }
        }
        
        // Check field conditions
        for (field_num, expected_value) in &rule.field_conditions {
            if let Some(field_value) = message.get_field(*field_num) {
                if !self.pattern_matches(expected_value, field_value) {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Required field not present
            }
        }
        
        Ok(true)
    }
    
    /// Check if a pattern matches a value (supports wildcards)
    fn pattern_matches(&self, pattern: &str, value: &str) -> bool {
        if pattern == "*" {
            return true; // Wildcard matches everything
        }
        
        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return value.starts_with(prefix) && value.ends_with(suffix);
            }
        }
        
        // Exact match
        pattern == value
    }
    
    /// Apply load balancing strategy
    fn apply_load_balancing(
        &mut self,
        destination: &RoutingDestination,
    ) -> Result<RoutingDestination, RoutingError> {
        // For now, just return the destination as-is
        // In a real implementation, this would handle multiple destinations
        // and apply the configured load balancing strategy
        
        if !destination.healthy {
            return Err(RoutingError::DestinationUnreachable(
                format!("destination {} is unhealthy", destination.id)
            ));
        }
        
        Ok(destination.clone())
    }
    
    /// Update destination health status
    pub fn update_destination_health(&mut self, destination_id: &str, healthy: bool) {
        for rule in &mut self.rules {
            if rule.destination.id == destination_id {
                rule.destination.healthy = healthy;
                info!(
                    destination = destination_id,
                    healthy = healthy,
                    "updated destination health"
                );
            }
        }
        
        if let Some(default_dest) = &mut self.default_destination {
            if default_dest.id == destination_id {
                default_dest.healthy = healthy;
            }
        }
        
        // Clear cache when health status changes
        self.route_cache.clear();
    }
    
    /// Get routing statistics
    pub fn get_stats(&self) -> RoutingStats {
        let total_rules = self.rules.len();
        let enabled_rules = self.rules.iter().filter(|r| r.enabled).count();
        let healthy_destinations = self.rules.iter()
            .filter(|r| r.destination.healthy)
            .count();
        
        RoutingStats {
            total_rules,
            enabled_rules,
            healthy_destinations,
            cache_size: self.route_cache.len(),
            cache_ttl: self.cache_ttl,
        }
    }
    
    /// Clear routing cache
    pub fn clear_cache(&mut self) {
        self.route_cache.clear();
        info!("cleared routing cache");
    }
    
    /// Remove a routing rule
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        let initial_len = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        
        if self.rules.len() < initial_len {
            self.route_cache.clear();
            info!(rule_id = rule_id, "removed routing rule");
            true
        } else {
            false
        }
    }
    
    /// Enable or disable a routing rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) -> bool {
        for rule in &mut self.rules {
            if rule.id == rule_id {
                rule.enabled = enabled;
                self.route_cache.clear();
                info!(rule_id = rule_id, enabled = enabled, "updated rule status");
                return true;
            }
        }
        false
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub total_rules: usize,
    pub enabled_rules: usize,
    pub healthy_destinations: usize,
    pub cache_size: usize,
    pub cache_ttl: u64,
}

/// Builder for routing rules
pub struct RoutingRuleBuilder {
    rule: RoutingRule,
}

impl RoutingRuleBuilder {
    /// Create a new rule builder
    pub fn new(id: String, destination: RoutingDestination) -> Self {
        Self {
            rule: RoutingRule {
                id,
                priority: 100,
                mti_pattern: None,
                processing_code_pattern: None,
                acquirer_pattern: None,
                card_acceptor_pattern: None,
                field_conditions: HashMap::new(),
                destination,
                enabled: true,
            },
        }
    }
    
    /// Set rule priority
    pub fn priority(mut self, priority: u32) -> Self {
        self.rule.priority = priority;
        self
    }
    
    /// Set MTI pattern
    pub fn mti_pattern(mut self, pattern: String) -> Self {
        self.rule.mti_pattern = Some(pattern);
        self
    }
    
    /// Set processing code pattern
    pub fn processing_code_pattern(mut self, pattern: String) -> Self {
        self.rule.processing_code_pattern = Some(pattern);
        self
    }
    
    /// Set acquirer pattern
    pub fn acquirer_pattern(mut self, pattern: String) -> Self {
        self.rule.acquirer_pattern = Some(pattern);
        self
    }
    
    /// Set card acceptor pattern
    pub fn card_acceptor_pattern(mut self, pattern: String) -> Self {
        self.rule.card_acceptor_pattern = Some(pattern);
        self
    }
    
    /// Add field condition
    pub fn field_condition(mut self, field_num: u8, value: String) -> Self {
        self.rule.field_conditions.insert(field_num, value);
        self
    }
    
    /// Set enabled status
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.rule.enabled = enabled;
        self
    }
    
    /// Build the routing rule
    pub fn build(self) -> RoutingRule {
        self.rule
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Iso8583Parser, MessageEncoding};
    use std::collections::HashMap;
    
    fn create_test_message() -> Iso8583Message {
        let mut fields = HashMap::new();
        fields.insert(3, "000000".to_string()); // Processing code
        fields.insert(32, "123456".to_string()); // Acquirer
        fields.insert(42, "TESTMERCHANT".to_string()); // Card acceptor
        
        Iso8583Message {
            mti: "0200".to_string(),
            bitmap: vec![false; 64],
            fields,
            encoding: MessageEncoding::Ascii,
        }
    }
    
    fn create_test_destination() -> RoutingDestination {
        RoutingDestination {
            id: "test-dest".to_string(),
            address: "127.0.0.1:8080".to_string(),
            destination_type: DestinationType::Upstream,
            weight: 100,
            healthy: true,
            timeout_ms: 5000,
        }
    }
    
    #[test]
    fn test_pattern_matching() {
        let router = MessageRouter::new(LoadBalancingStrategy::RoundRobin);
        
        // Exact match
        assert!(router.pattern_matches("0200", "0200"));
        assert!(!router.pattern_matches("0200", "0100"));
        
        // Wildcard match
        assert!(router.pattern_matches("*", "anything"));
        assert!(router.pattern_matches("02*", "0200"));
        assert!(router.pattern_matches("*00", "0200"));
        assert!(!router.pattern_matches("01*", "0200"));
    }
    
    #[test]
    fn test_routing_rule_builder() {
        let destination = create_test_destination();
        
        let rule = RoutingRuleBuilder::new("test-rule".to_string(), destination)
            .priority(10)
            .mti_pattern("02*".to_string())
            .processing_code_pattern("000000".to_string())
            .field_condition(32, "123456".to_string())
            .build();
        
        assert_eq!(rule.id, "test-rule");
        assert_eq!(rule.priority, 10);
        assert_eq!(rule.mti_pattern, Some("02*".to_string()));
        assert_eq!(rule.processing_code_pattern, Some("000000".to_string()));
        assert_eq!(rule.field_conditions.get(&32), Some(&"123456".to_string()));
    }
    
    #[test]
    fn test_message_routing() {
        let mut router = MessageRouter::new(LoadBalancingStrategy::RoundRobin);
        let parser = Iso8583Parser::new(MessageEncoding::Ascii);
        let message = create_test_message();
        let destination = create_test_destination();
        
        // Add a matching rule
        let rule = RoutingRuleBuilder::new("test-rule".to_string(), destination.clone())
            .mti_pattern("02*".to_string())
            .build();
        
        router.add_rule(rule);
        
        // Route the message
        let result = router.route_message(&message, &parser).unwrap();
        assert_eq!(result.id, "test-dest");
    }
    
    #[test]
    fn test_no_matching_rule() {
        let mut router = MessageRouter::new(LoadBalancingStrategy::RoundRobin);
        let parser = Iso8583Parser::new(MessageEncoding::Ascii);
        let message = create_test_message();
        
        // No rules added, should fail
        let result = router.route_message(&message, &parser);
        assert!(result.is_err());
        
        // Add default destination
        let default_dest = create_test_destination();
        router.set_default_destination(default_dest.clone());
        
        // Should now use default destination
        let result = router.route_message(&message, &parser).unwrap();
        assert_eq!(result.id, "test-dest");
    }
    
    #[test]
    fn test_unhealthy_destination() {
        let mut router = MessageRouter::new(LoadBalancingStrategy::RoundRobin);
        let parser = Iso8583Parser::new(MessageEncoding::Ascii);
        let message = create_test_message();
        
        let mut destination = create_test_destination();
        destination.healthy = false;
        
        let rule = RoutingRuleBuilder::new("test-rule".to_string(), destination)
            .mti_pattern("02*".to_string())
            .build();
        
        router.add_rule(rule);
        
        // Should fail due to unhealthy destination
        let result = router.route_message(&message, &parser);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_cache_key_generation() {
        let router = MessageRouter::new(LoadBalancingStrategy::RoundRobin);
        let message = create_test_message();
        
        let cache_key = router.generate_cache_key(&message);
        assert!(cache_key.contains("0200")); // MTI
        assert!(cache_key.contains("000000")); // Processing code
        assert!(cache_key.contains("123456")); // Acquirer
    }
}