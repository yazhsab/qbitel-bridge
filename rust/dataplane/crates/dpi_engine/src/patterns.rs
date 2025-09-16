//! Pattern Matching for DPI Classification
//!
//! High-performance pattern matching using various algorithms including
//! Aho-Corasick, Boyer-Moore, and Hyperscan for deep packet inspection.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

#[cfg(feature = "hardware-acceleration")]
use hyperscan::*;

use crate::DpiError;

/// Pattern matching errors
#[derive(Error, Debug)]
pub enum PatternError {
    #[error("Pattern compilation failed: {0}")]
    CompilationFailed(String),
    
    #[error("Pattern matching failed: {0}")]
    MatchingFailed(String),
    
    #[error("Pattern database loading failed: {0}")]
    DatabaseLoadFailed(String),
    
    #[error("Hyperscan error: {0}")]
    HyperscanError(String),
}

type Result<T> = std::result::Result<T, PatternError>;

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: u32,
    pub pattern_name: String,
    pub offset: usize,
    pub length: usize,
    pub matched_data: String,
    pub confidence: f32,
    pub category: PatternCategory,
}

/// Pattern categories for organization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternCategory {
    Protocol,
    Application,
    Security,
    Malware,
    FileType,
    Custom,
}

/// Pattern database for efficient matching
pub struct PatternDatabase {
    patterns: Vec<Pattern>,
    compiled_database: Option<CompiledDatabase>,
    aho_corasick: Option<aho_corasick::AhoCorasick>,
}

/// Individual pattern definition
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: u32,
    pub name: String,
    pub pattern: String,
    pub flags: PatternFlags,
    pub category: PatternCategory,
    pub confidence: f32,
}

/// Pattern flags for matching behavior
#[derive(Debug, Clone, Default)]
pub struct PatternFlags {
    pub case_insensitive: bool,
    pub multiline: bool,
    pub dotall: bool,
    pub utf8: bool,
}

/// Compiled database wrapper for different engines
enum CompiledDatabase {
    #[cfg(feature = "hardware-acceleration")]
    Hyperscan(hyperscan::Database),
    AhoCorasick(aho_corasick::AhoCorasick),
    Regex(Vec<regex::Regex>),
}

/// High-performance pattern matcher
pub struct PatternMatcher {
    database: Arc<RwLock<PatternDatabase>>,
    config: PatternMatcherConfig,
    statistics: Arc<RwLock<PatternMatcherStats>>,
}

/// Pattern matcher configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatcherConfig {
    pub enable_hyperscan: bool,
    pub max_pattern_length: usize,
    pub case_sensitive: bool,
    pub enable_regex: bool,
    pub max_matches_per_packet: usize,
    pub pattern_cache_size: usize,
}

impl Default for PatternMatcherConfig {
    fn default() -> Self {
        Self {
            enable_hyperscan: cfg!(feature = "hardware-acceleration"),
            max_pattern_length: 1000,
            case_sensitive: true,
            enable_regex: true,
            max_matches_per_packet: 100,
            pattern_cache_size: 1000,
        }
    }
}

/// Pattern matcher statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMatcherStats {
    pub patterns_loaded: u32,
    pub total_matches: u64,
    pub packets_scanned: u64,
    pub average_scan_time_us: f64,
    pub database_compilation_time_ms: u64,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub async fn new(config: &crate::config::DpiConfig) -> Result<Self> {
        info!("Initializing Pattern Matcher");
        
        let matcher_config = PatternMatcherConfig::default();
        let database = Arc::new(RwLock::new(PatternDatabase::new()));
        
        let matcher = Self {
            database,
            config: matcher_config,
            statistics: Arc::new(RwLock::new(PatternMatcherStats::default())),
        };
        
        // Load default patterns
        matcher.load_default_patterns().await?;
        
        info!("Pattern Matcher initialized successfully");
        Ok(matcher)
    }
    
    /// Match patterns against packet data
    #[instrument(skip(self, data))]
    pub async fn match_patterns(&self, data: &Bytes) -> Result<Vec<PatternMatch>> {
        let start_time = std::time::Instant::now();
        let database = self.database.read();
        
        let matches = if let Some(ref compiled_db) = database.compiled_database {
            match compiled_db {
                #[cfg(feature = "hardware-acceleration")]
                CompiledDatabase::Hyperscan(db) => {
                    self.hyperscan_match(db, data)?
                },
                CompiledDatabase::AhoCorasick(ac) => {
                    self.aho_corasick_match(ac, data)?
                },
                CompiledDatabase::Regex(regexes) => {
                    self.regex_match(regexes, data)?
                },
            }
        } else {
            // Fallback to simple string matching
            self.simple_string_match(&database.patterns, data)?
        };
        
        // Update statistics
        let scan_time = start_time.elapsed().as_micros() as f64;
        let mut stats = self.statistics.write();
        stats.packets_scanned += 1;
        stats.total_matches += matches.len() as u64;
        stats.average_scan_time_us = 
            (stats.average_scan_time_us * (stats.packets_scanned - 1) as f64 + scan_time) / 
            stats.packets_scanned as f64;
        
        debug!("Pattern matching found {} matches in {:.2}μs", matches.len(), scan_time);
        Ok(matches)
    }
    
    /// Load default pattern set
    async fn load_default_patterns(&self) -> Result<()> {
        let default_patterns = vec![
            Pattern {
                id: 1,
                name: "HTTP_GET".to_string(),
                pattern: "GET ".to_string(),
                flags: PatternFlags::default(),
                category: PatternCategory::Protocol,
                confidence: 0.9,
            },
            Pattern {
                id: 2,
                name: "HTTP_POST".to_string(),
                pattern: "POST ".to_string(),
                flags: PatternFlags::default(),
                category: PatternCategory::Protocol,
                confidence: 0.9,
            },
            Pattern {
                id: 3,
                name: "TLS_HANDSHAKE".to_string(),
                pattern: "\x16\x03".to_string(),
                flags: PatternFlags::default(),
                category: PatternCategory::Protocol,
                confidence: 0.95,
            },
        ];
        
        let mut database = self.database.write();
        database.patterns = default_patterns;
        
        // Compile patterns
        database.compile_patterns(&self.config)?;
        
        let mut stats = self.statistics.write();
        stats.patterns_loaded = database.patterns.len() as u32;
        
        info!("Loaded {} default patterns", database.patterns.len());
        Ok(())
    }
    
    #[cfg(feature = "hardware-acceleration")]
    fn hyperscan_match(&self, _db: &hyperscan::Database, _data: &Bytes) -> Result<Vec<PatternMatch>> {
        // Placeholder for Hyperscan implementation
        Ok(Vec::new())
    }
    
    fn aho_corasick_match(&self, ac: &aho_corasick::AhoCorasick, data: &Bytes) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        
        for mat in ac.find_iter(data) {
            matches.push(PatternMatch {
                pattern_id: mat.pattern().as_u32(),
                pattern_name: format!("Pattern_{}", mat.pattern().as_usize()),
                offset: mat.start(),
                length: mat.len(),
                matched_data: String::from_utf8_lossy(&data[mat.start()..mat.end()]).to_string(),
                confidence: 0.8,
                category: PatternCategory::Protocol,
            });
        }
        
        Ok(matches)
    }
    
    fn regex_match(&self, _regexes: &[regex::Regex], _data: &Bytes) -> Result<Vec<PatternMatch>> {
        // Placeholder for regex matching
        Ok(Vec::new())
    }
    
    fn simple_string_match(&self, patterns: &[Pattern], data: &Bytes) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let data_str = String::from_utf8_lossy(data);
        
        for pattern in patterns {
            if let Some(pos) = data_str.find(&pattern.pattern) {
                matches.push(PatternMatch {
                    pattern_id: pattern.id,
                    pattern_name: pattern.name.clone(),
                    offset: pos,
                    length: pattern.pattern.len(),
                    matched_data: pattern.pattern.clone(),
                    confidence: pattern.confidence,
                    category: pattern.category.clone(),
                });
            }
        }
        
        Ok(matches)
    }
    
    /// Get pattern matcher statistics
    pub fn get_statistics(&self) -> PatternMatcherStats {
        self.statistics.read().clone()
    }
}

impl PatternDatabase {
    fn new() -> Self {
        Self {
            patterns: Vec::new(),
            compiled_database: None,
            aho_corasick: None,
        }
    }
    
    fn compile_patterns(&mut self, config: &PatternMatcherConfig) -> Result<()> {
        info!("Compiling pattern database");
        let start_time = std::time::Instant::now();
        
        // Try Hyperscan first if available
        #[cfg(feature = "hardware-acceleration")]
        if config.enable_hyperscan {
            match self.compile_hyperscan() {
                Ok(db) => {
                    self.compiled_database = Some(CompiledDatabase::Hyperscan(db));
                    info!("Compiled patterns using Hyperscan");
                    return Ok(());
                },
                Err(e) => {
                    warn!("Hyperscan compilation failed, falling back: {}", e);
                }
            }
        }
        
        // Fallback to Aho-Corasick
        match self.compile_aho_corasick() {
            Ok(ac) => {
                self.compiled_database = Some(CompiledDatabase::AhoCorasick(ac));
                info!("Compiled patterns using Aho-Corasick");
            },
            Err(e) => {
                warn!("Aho-Corasick compilation failed: {}", e);
                return Err(e);
            }
        }
        
        let compilation_time = start_time.elapsed().as_millis();
        info!("Pattern database compiled in {}ms", compilation_time);
        
        Ok(())
    }
    
    #[cfg(feature = "hardware-acceleration")]
    fn compile_hyperscan(&self) -> Result<hyperscan::Database> {
        // Placeholder for Hyperscan compilation
        Err(PatternError::CompilationFailed("Hyperscan not available".to_string()))
    }
    
    fn compile_aho_corasick(&self) -> Result<aho_corasick::AhoCorasick> {
        let patterns: Vec<&str> = self.patterns.iter()
            .map(|p| p.pattern.as_str())
            .collect();
        
        aho_corasick::AhoCorasick::new(&patterns)
            .map_err(|e| PatternError::CompilationFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pattern_matcher_creation() {
        let config = crate::config::DpiConfig::default();
        let result = PatternMatcher::new(&config).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_simple_pattern_matching() {
        let config = crate::config::DpiConfig::default();
        let matcher = PatternMatcher::new(&config).await.unwrap();
        
        let test_data = Bytes::from("GET /index.html HTTP/1.1\r\n");
        let matches = matcher.match_patterns(&test_data).await.unwrap();
        
        assert!(!matches.is_empty());
        assert_eq!(matches[0].pattern_name, "HTTP_GET");
    }
}