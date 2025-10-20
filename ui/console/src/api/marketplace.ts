import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  Protocol,
  SearchProtocolsParams,
  SearchProtocolsResponse,
  SubmitProtocolRequest,
  SubmitProtocolResponse,
  ValidationStatus,
  PurchaseProtocolRequest,
  PurchaseProtocolResponse,
  SubmitReviewRequest,
  Review,
  Installation,
} from '../types/marketplace';

export class MarketplaceApiClient {
  private axios: AxiosInstance;
  private getAuthToken: () => Promise<string | null>;

  constructor(
    getAuthToken: () => Promise<string | null>,
    baseURL: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  ) {
    this.getAuthToken = getAuthToken;
    this.axios = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor to add auth token
    this.axios.interceptors.request.use(
      async (config) => {
        const token = await this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.axios.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized - could trigger re-auth
          console.error('Unauthorized access to marketplace API');
        } else if (error.response?.status === 404) {
          console.error('Marketplace endpoint not found:', error.config?.url);
        } else if (error.response?.status === 500) {
          console.error('Marketplace server error:', error.message);
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * Search and filter protocols
   */
  async searchProtocols(params: SearchProtocolsParams = {}): Promise<SearchProtocolsResponse> {
    const response = await this.axios.get<SearchProtocolsResponse>('/api/v1/marketplace/protocols/search', {
      params,
    });
    return response.data;
  }

  /**
   * Get detailed protocol information
   */
  async getProtocol(protocolId: string): Promise<Protocol> {
    const response = await this.axios.get<Protocol>(`/api/v1/marketplace/protocols/${protocolId}`);
    return response.data;
  }

  /**
   * Submit a new protocol for validation
   */
  async submitProtocol(data: SubmitProtocolRequest): Promise<SubmitProtocolResponse> {
    const response = await this.axios.post<SubmitProtocolResponse>(
      '/api/v1/marketplace/protocols',
      data
    );
    return response.data;
  }

  /**
   * Check protocol validation status
   */
  async getValidationStatus(protocolId: string): Promise<ValidationStatus> {
    const response = await this.axios.get<ValidationStatus>(
      `/api/v1/marketplace/protocols/${protocolId}/validation`
    );
    return response.data;
  }

  /**
   * Purchase or subscribe to a protocol
   */
  async purchaseProtocol(
    protocolId: string,
    data: PurchaseProtocolRequest
  ): Promise<PurchaseProtocolResponse> {
    const response = await this.axios.post<PurchaseProtocolResponse>(
      `/api/v1/marketplace/protocols/${protocolId}/purchase`,
      data
    );
    return response.data;
  }

  /**
   * Submit a protocol review
   */
  async submitReview(
    protocolId: string,
    data: SubmitReviewRequest
  ): Promise<Review> {
    const response = await this.axios.post<Review>(
      `/api/v1/marketplace/protocols/${protocolId}/reviews`,
      data
    );
    return response.data;
  }

  /**
   * Get protocol reviews
   */
  async getProtocolReviews(
    protocolId: string,
    page: number = 1,
    limit: number = 10
  ): Promise<{ reviews: Review[]; total: number }> {
    const response = await this.axios.get<{ reviews: Review[]; total: number }>(
      `/api/v1/marketplace/protocols/${protocolId}/reviews`,
      { params: { page, limit } }
    );
    return response.data;
  }

  /**
   * Get user's submitted protocols
   */
  async getMyProtocols(): Promise<Protocol[]> {
    const response = await this.axios.get<Protocol[]>('/api/v1/marketplace/my/protocols');
    return response.data;
  }

  /**
   * Get user's installed protocols
   */
  async getMyInstallations(): Promise<Installation[]> {
    const response = await this.axios.get<Installation[]>('/api/v1/marketplace/my/installations');
    return response.data;
  }

  /**
   * Update protocol information (for creators)
   */
  async updateProtocol(
    protocolId: string,
    data: Partial<SubmitProtocolRequest>
  ): Promise<Protocol> {
    const response = await this.axios.put<Protocol>(
      `/api/v1/marketplace/protocols/${protocolId}`,
      data
    );
    return response.data;
  }

  /**
   * Delete protocol (for creators)
   */
  async deleteProtocol(protocolId: string): Promise<void> {
    await this.axios.delete(`/api/v1/marketplace/protocols/${protocolId}`);
  }

  /**
   * Get featured protocols
   */
  async getFeaturedProtocols(limit: number = 6): Promise<Protocol[]> {
    const response = await this.axios.get<SearchProtocolsResponse>(
      '/api/v1/marketplace/protocols/search',
      {
        params: {
          sort: 'rating',
          limit,
        },
      }
    );
    return response.data.protocols.filter(p => p.is_featured);
  }

  /**
   * Get popular protocols
   */
  async getPopularProtocols(limit: number = 10): Promise<Protocol[]> {
    const response = await this.axios.get<SearchProtocolsResponse>(
      '/api/v1/marketplace/protocols/search',
      {
        params: {
          sort: 'downloads',
          limit,
        },
      }
    );
    return response.data.protocols;
  }

  /**
   * Get recent protocols
   */
  async getRecentProtocols(limit: number = 10): Promise<Protocol[]> {
    const response = await this.axios.get<SearchProtocolsResponse>(
      '/api/v1/marketplace/protocols/search',
      {
        params: {
          sort: 'recent',
          limit,
        },
      }
    );
    return response.data.protocols;
  }

  /**
   * Get protocols by category
   */
  async getProtocolsByCategory(
    category: string,
    page: number = 1,
    limit: number = 20
  ): Promise<SearchProtocolsResponse> {
    return this.searchProtocols({
      category,
      page,
      limit,
    });
  }

  /**
   * Download protocol files
   */
  async downloadProtocolFile(url: string): Promise<Blob> {
    const response = await this.axios.get<Blob>(url, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Upload protocol file (multipart)
   */
  async uploadProtocolFile(file: File, type: 'spec' | 'parser' | 'test_data' | 'docs'): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    const response = await this.axios.post<{ url: string }>(
      '/api/v1/marketplace/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data.url;
  }

  /**
   * Get marketplace statistics
   */
  async getMarketplaceStats(): Promise<{
    total_protocols: number;
    total_downloads: number;
    total_creators: number;
    categories: Record<string, number>;
  }> {
    const response = await this.axios.get('/api/v1/marketplace/stats');
    return response.data;
  }

  /**
   * Mark review as helpful
   */
  async markReviewHelpful(reviewId: string, helpful: boolean): Promise<void> {
    await this.axios.post(`/api/v1/marketplace/reviews/${reviewId}/helpful`, {
      helpful,
    });
  }

  /**
   * Report a protocol issue
   */
  async reportProtocol(
    protocolId: string,
    reason: string,
    description: string
  ): Promise<void> {
    await this.axios.post(`/api/v1/marketplace/protocols/${protocolId}/report`, {
      reason,
      description,
    });
  }

  /**
   * Get creator analytics
   */
  async getCreatorAnalytics(): Promise<{
    total_revenue: number;
    total_downloads: number;
    active_installations: number;
    revenue_by_protocol: Record<string, number>;
    downloads_trend: Array<{ date: string; count: number }>;
  }> {
    const response = await this.axios.get('/api/v1/marketplace/creator/analytics');
    return response.data;
  }
}

// Factory function
export function createMarketplaceApiClient(
  getAuthToken: () => Promise<string | null>
): MarketplaceApiClient {
  return new MarketplaceApiClient(getAuthToken);
}

export default MarketplaceApiClient;
