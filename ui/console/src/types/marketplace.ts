// Marketplace types and interfaces

export interface MarketplaceUser {
  user_id: string;
  email: string;
  username: string;
  full_name?: string;
  user_type: 'individual' | 'vendor' | 'enterprise';
  organization?: string;
  is_verified: boolean;
  verification_date?: string;
  reputation_score: number;
  total_contributions: number;
  total_downloads: number;
  stripe_account_id?: string;
  payout_enabled: boolean;
  bio?: string;
  website_url?: string;
  avatar_url?: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface ProtocolAuthor {
  user_id: string;
  username: string;
  full_name?: string;
  organization?: string;
  is_verified: boolean;
  reputation_score: number;
  total_contributions: number;
}

export interface TechnicalSpecs {
  spec_format: string;
  spec_file_url: string;
  parser_code_url?: string;
  test_data_url?: string;
  documentation_url?: string;
}

export interface Licensing {
  license_type: 'free' | 'paid' | 'enterprise';
  price_model?: 'one_time' | 'subscription' | 'usage_based';
  base_price?: number;
  currency?: string;
  refund_policy?: string;
}

export interface QualityMetrics {
  certification_status: 'pending' | 'certified' | 'rejected' | 'in_review';
  certification_date?: string;
  average_rating: number;
  total_ratings: number;
  download_count: number;
  active_installations: number;
}

export interface Compatibility {
  min_qbitel_version: string;
  supported_qbitel_versions: string[];
  dependencies: string[];
}

export interface ReviewsSummary {
  5_star: number;
  4_star: number;
  3_star: number;
  2_star: number;
  1_star: number;
  recent_reviews: Review[];
}

export interface Protocol {
  protocol_id: string;
  protocol_name: string;
  display_name: string;
  short_description: string;
  long_description?: string;
  category: string;
  subcategory?: string;
  tags: string[];
  version: string;
  protocol_type: 'binary' | 'text' | 'xml' | 'json';
  industry?: string;
  technical_specs?: TechnicalSpecs;
  author: ProtocolAuthor;
  licensing: Licensing;
  quality_metrics: QualityMetrics;
  compatibility?: Compatibility;
  reviews_summary?: ReviewsSummary;
  is_featured: boolean;
  is_official: boolean;
  created_at: string;
  published_at?: string;
  updated_at: string;
}

export interface Review {
  review_id: string;
  protocol_id: string;
  customer_id: string;
  rating: number;
  title?: string;
  review_text?: string;
  helpful_count: number;
  unhelpful_count: number;
  is_verified_purchase: boolean;
  status: string;
  created_at: string;
  updated_at: string;
  reviewer_name?: string;
}

export interface Installation {
  installation_id: string;
  protocol_id: string;
  customer_id: string;
  installed_version: string;
  installation_date: string;
  last_updated: string;
  license_key: string;
  license_type: string;
  expires_at?: string;
  total_packets_processed: number;
  last_used_at?: string;
  status: 'active' | 'suspended' | 'expired';
  protocol?: Protocol;
}

export interface Transaction {
  transaction_id: string;
  protocol_id: string;
  customer_id: string;
  installation_id?: string;
  transaction_type: 'purchase' | 'subscription' | 'renewal' | 'refund';
  amount: number;
  currency: string;
  platform_fee: number;
  creator_revenue: number;
  stripe_payment_intent_id?: string;
  payment_method?: string;
  status: 'pending' | 'completed' | 'failed' | 'refunded';
  created_at: string;
}

export interface ValidationStep {
  step: 'syntax_validation' | 'parser_testing' | 'security_scan' | 'performance_benchmark' | 'manual_review';
  status: 'pending' | 'in_progress' | 'passed' | 'failed';
  message: string;
  score?: number;
  errors?: string[];
  warnings?: string[];
}

export interface ValidationStatus {
  protocol_id: string;
  validation_status: 'pending' | 'in_progress' | 'passed' | 'failed' | 'needs_review';
  steps: ValidationStep[];
  estimated_completion?: string;
}

// API Request/Response types

export interface SearchProtocolsParams {
  q?: string;
  category?: string;
  license_type?: 'free' | 'paid' | 'enterprise';
  min_rating?: number;
  sort?: 'rating' | 'downloads' | 'recent' | 'alphabetical';
  page?: number;
  limit?: number;
}

export interface SearchProtocolsResponse {
  protocols: Protocol[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    pages: number;
  };
  facets: {
    categories: Record<string, number>;
    license_types: Record<string, number>;
  };
}

export interface SubmitProtocolRequest {
  protocol_name: string;
  display_name: string;
  short_description: string;
  long_description?: string;
  category: string;
  subcategory?: string;
  tags: string[];
  version: string;
  protocol_type: 'binary' | 'text' | 'xml' | 'json';
  industry?: string;
  spec_format: string;
  spec_file: string; // base64 encoded
  parser_code?: string; // base64 encoded
  test_data?: string; // base64 encoded
  license_type: 'free' | 'paid' | 'enterprise';
  price_model?: 'one_time' | 'subscription' | 'usage_based';
  base_price?: number;
  min_qbitel_version: string;
}

export interface SubmitProtocolResponse {
  protocol_id: string;
  status: string;
  message: string;
  estimated_review_time: string;
  validation_url: string;
}

export interface PurchaseProtocolRequest {
  license_type: 'production' | 'development' | 'enterprise';
  payment_method_id: string;
  billing_email: string;
}

export interface PurchaseProtocolResponse {
  installation_id: string;
  license_key: string;
  license_type: string;
  expires_at?: string;
  status: string;
  download_urls: {
    spec: string;
    parser?: string;
    docs?: string;
  };
  installation_instructions: string;
}

export interface SubmitReviewRequest {
  rating: number;
  title?: string;
  review_text?: string;
}

// UI State types

export interface MarketplaceFilters {
  searchQuery: string;
  category: string;
  licenseType: string;
  minRating: number;
  sortBy: 'rating' | 'downloads' | 'recent' | 'alphabetical';
}

export interface PurchaseFlowState {
  step: 'select_plan' | 'payment' | 'confirm' | 'success';
  selectedPlan?: 'monthly' | 'annual' | 'one_time';
  licenseType: string;
  paymentMethodId?: string;
  billingEmail: string;
  processing: boolean;
  error?: string;
}

export interface SubmissionFlowState {
  step: number; // 1-5
  formData: Partial<SubmitProtocolRequest>;
  files: {
    spec?: File;
    parser?: File;
    testData?: File;
    docs?: File;
  };
  validationErrors: Record<string, string>;
  submitting: boolean;
}

// Constants

export const PROTOCOL_CATEGORIES = [
  { value: 'finance', label: 'Financial', icon: '💰' },
  { value: 'healthcare', label: 'Healthcare', icon: '🏥' },
  { value: 'industrial', label: 'Industrial', icon: '🏭' },
  { value: 'iot', label: 'IoT', icon: '📡' },
  { value: 'telecom', label: 'Telecom', icon: '📞' },
  { value: 'legacy', label: 'Legacy Systems', icon: '🖥️' },
  { value: 'custom', label: 'Custom', icon: '⚙️' },
];

export const LICENSE_TYPES = [
  { value: 'free', label: 'Free', description: 'Open source, community supported' },
  { value: 'paid', label: 'Commercial', description: 'Paid license with support' },
  { value: 'enterprise', label: 'Enterprise', description: 'Custom pricing and SLA' },
];

export const CERTIFICATION_BADGES = {
  certified: { label: 'Certified', color: '#00C851', icon: '✓' },
  pending: { label: 'Pending Review', color: '#FFB84D', icon: '⏳' },
  rejected: { label: 'Rejected', color: '#FF4444', icon: '✗' },
  in_review: { label: 'In Review', color: '#0066CC', icon: '👁️' },
};
