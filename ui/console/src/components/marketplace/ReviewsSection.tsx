import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Rating,
  Button,
  TextField,
  Grid,
  Avatar,
  Divider,
  LinearProgress,
  IconButton,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Verified as VerifiedIcon,
} from '@mui/icons-material';
import { Review, SubmitReviewRequest } from '../../types/marketplace';
import { MarketplaceApiClient } from '../../api/marketplace';

interface ReviewsSectionProps {
  protocolId: string;
  apiClient: MarketplaceApiClient;
  initialReviews: Review[];
}

const ReviewsSection: React.FC<ReviewsSectionProps> = ({
  protocolId,
  apiClient,
  initialReviews,
}) => {
  const [reviews, setReviews] = useState<Review[]>(initialReviews);
  const [reviewDialogOpen, setReviewDialogOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newReview, setNewReview] = useState<SubmitReviewRequest>({
    rating: 5,
    title: '',
    review_text: '',
  });

  // Calculate rating distribution
  const ratingDistribution = {
    5: reviews.filter(r => r.rating === 5).length,
    4: reviews.filter(r => r.rating === 4).length,
    3: reviews.filter(r => r.rating === 3).length,
    2: reviews.filter(r => r.rating === 2).length,
    1: reviews.filter(r => r.rating === 1).length,
  };

  const totalReviews = reviews.length;
  const averageRating = totalReviews > 0
    ? reviews.reduce((sum, r) => sum + r.rating, 0) / totalReviews
    : 0;

  const handleSubmitReview = async () => {
    try {
      setSubmitting(true);
      setError(null);

      const review = await apiClient.submitReview(protocolId, newReview);
      setReviews([review, ...reviews]);
      setReviewDialogOpen(false);
      setNewReview({ rating: 5, title: '', review_text: '' });
    } catch (err: any) {
      setError(err.message || 'Failed to submit review');
    } finally {
      setSubmitting(false);
    }
  };

  const handleHelpful = async (reviewId: string, helpful: boolean) => {
    try {
      await apiClient.markReviewHelpful(reviewId, helpful);
      // Update local state
      setReviews(reviews.map(r =>
        r.review_id === reviewId
          ? {
              ...r,
              helpful_count: helpful ? r.helpful_count + 1 : r.helpful_count,
              unhelpful_count: !helpful ? r.unhelpful_count + 1 : r.unhelpful_count,
            }
          : r
      ));
    } catch (err) {
      console.error('Failed to mark review as helpful:', err);
    }
  };

  return (
    <Box>
      {/* Rating Overview */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box textAlign="center">
              <Typography variant="h2" fontWeight="bold">
                {averageRating.toFixed(1)}
              </Typography>
              <Rating value={averageRating} precision={0.1} size="large" readOnly />
              <Typography variant="body2" color="text.secondary">
                Based on {totalReviews} reviews
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={8}>
            {[5, 4, 3, 2, 1].map((star) => {
              const count = ratingDistribution[star as keyof typeof ratingDistribution];
              const percentage = totalReviews > 0 ? (count / totalReviews) * 100 : 0;

              return (
                <Box key={star} display="flex" alignItems="center" mb={1}>
                  <Typography variant="body2" sx={{ width: 60 }}>
                    {star} star
                  </Typography>
                  <Box flexGrow={1} mx={2}>
                    <LinearProgress
                      variant="determinate"
                      value={percentage}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                  <Typography variant="body2" sx={{ width: 40, textAlign: 'right' }}>
                    {count}
                  </Typography>
                </Box>
              );
            })}
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        <Button
          variant="outlined"
          fullWidth
          onClick={() => setReviewDialogOpen(true)}
        >
          Write a Review
        </Button>
      </Paper>

      {/* Reviews List */}
      <Typography variant="h6" gutterBottom fontWeight="600">
        Customer Reviews
      </Typography>

      {reviews.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No reviews yet. Be the first to review this protocol!
          </Typography>
        </Paper>
      ) : (
        <Box>
          {reviews.map((review) => (
            <Paper key={review.review_id} sx={{ p: 3, mb: 2 }}>
              <Box display="flex" alignItems="flex-start">
                <Avatar sx={{ mr: 2, bgcolor: 'primary.light' }}>
                  {review.reviewer_name?.charAt(0) || 'U'}
                </Avatar>
                <Box flexGrow={1}>
                  <Box display="flex" alignItems="center" mb={1}>
                    <Typography variant="subtitle1" fontWeight="600">
                      {review.reviewer_name || 'Anonymous'}
                    </Typography>
                    {review.is_verified_purchase && (
                      <VerifiedIcon
                        color="success"
                        sx={{ ml: 1, fontSize: 18 }}
                        titleAccess="Verified Purchase"
                      />
                    )}
                  </Box>
                  <Box display="flex" alignItems="center" gap={2} mb={1}>
                    <Rating value={review.rating} size="small" readOnly />
                    <Typography variant="caption" color="text.secondary">
                      {new Date(review.created_at).toLocaleDateString()}
                    </Typography>
                  </Box>
                  {review.title && (
                    <Typography variant="subtitle2" fontWeight="600" mb={1}>
                      {review.title}
                    </Typography>
                  )}
                  <Typography variant="body2" paragraph>
                    {review.review_text}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="caption" color="text.secondary">
                      Was this helpful?
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={() => handleHelpful(review.review_id, true)}
                    >
                      <ThumbUpIcon fontSize="small" />
                    </IconButton>
                    <Typography variant="caption">{review.helpful_count}</Typography>
                    <IconButton
                      size="small"
                      onClick={() => handleHelpful(review.review_id, false)}
                    >
                      <ThumbDownIcon fontSize="small" />
                    </IconButton>
                    <Typography variant="caption">{review.unhelpful_count}</Typography>
                  </Box>
                </Box>
              </Box>
            </Paper>
          ))}
        </Box>
      )}

      {/* Write Review Dialog */}
      <Dialog
        open={reviewDialogOpen}
        onClose={() => setReviewDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Write a Review</DialogTitle>
        <DialogContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Your Rating *
            </Typography>
            <Rating
              value={newReview.rating}
              onChange={(e, value) =>
                setNewReview({ ...newReview, rating: value || 5 })
              }
              size="large"
            />
          </Box>

          <TextField
            fullWidth
            label="Review Title (optional)"
            value={newReview.title}
            onChange={(e) =>
              setNewReview({ ...newReview, title: e.target.value })
            }
            sx={{ mb: 2 }}
            placeholder="Summarize your experience"
          />

          <TextField
            fullWidth
            label="Your Review (optional)"
            value={newReview.review_text}
            onChange={(e) =>
              setNewReview({ ...newReview, review_text: e.target.value })
            }
            multiline
            rows={4}
            placeholder="Tell others about your experience with this protocol"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmitReview}
            disabled={submitting}
          >
            {submitting ? 'Submitting...' : 'Submit Review'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ReviewsSection;
