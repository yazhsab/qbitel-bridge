import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Chip,
  Rating,
  Button,
  Avatar,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Verified as VerifiedIcon,
  Download as DownloadIcon,
  Star as StarIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { Protocol, CERTIFICATION_BADGES } from '../../types/marketplace';
import { useNavigate } from 'react-router-dom';

interface ProtocolCardProps {
  protocol: Protocol;
  onViewDetails?: (protocol: Protocol) => void;
}

const ProtocolCard: React.FC<ProtocolCardProps> = ({ protocol, onViewDetails }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onViewDetails) {
      onViewDetails(protocol);
    } else {
      navigate(`/marketplace/protocols/${protocol.protocol_id}`);
    }
  };

  const certificationBadge = CERTIFICATION_BADGES[protocol.quality_metrics.certification_status];

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        cursor: 'pointer',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 6,
        },
        position: 'relative',
        border: '1px solid',
        borderColor: 'divider',
      }}
      onClick={handleClick}
    >
      {/* Featured/Official Badge */}
      {(protocol.is_featured || protocol.is_official) && (
        <Box
          sx={{
            position: 'absolute',
            top: 12,
            right: 12,
            zIndex: 1,
          }}
        >
          {protocol.is_official && (
            <Chip
              label="Official"
              size="small"
              sx={{
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                fontWeight: 600,
                mr: 0.5,
              }}
            />
          )}
          {protocol.is_featured && (
            <Chip
              label="Featured"
              size="small"
              icon={<StarIcon />}
              sx={{
                bgcolor: 'warning.main',
                color: 'warning.contrastText',
                fontWeight: 600,
              }}
            />
          )}
        </Box>
      )}

      <CardContent sx={{ flexGrow: 1, pt: protocol.is_featured || protocol.is_official ? 5 : 2 }}>
        {/* Protocol Icon and Name */}
        <Box display="flex" alignItems="center" mb={2}>
          <Avatar
            sx={{
              width: 56,
              height: 56,
              bgcolor: 'primary.light',
              mr: 2,
            }}
          >
            {protocol.display_name.substring(0, 2).toUpperCase()}
          </Avatar>
          <Box flexGrow={1}>
            <Typography variant="h6" component="div" gutterBottom sx={{ fontWeight: 600 }}>
              {protocol.display_name}
              {protocol.author.is_verified && (
                <Tooltip title="Verified Creator">
                  <VerifiedIcon
                    color="primary"
                    sx={{ ml: 0.5, fontSize: 18, verticalAlign: 'middle' }}
                  />
                </Tooltip>
              )}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              by {protocol.author.organization || protocol.author.username}
            </Typography>
          </Box>
        </Box>

        {/* Description */}
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            mb: 2,
            height: '3em',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
          }}
        >
          {protocol.short_description}
        </Typography>

        {/* Tags */}
        <Box display="flex" flexWrap="wrap" gap={0.5} mb={2}>
          {protocol.tags.slice(0, 3).map((tag) => (
            <Chip
              key={tag}
              label={tag}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
          ))}
          {protocol.tags.length > 3 && (
            <Chip
              label={`+${protocol.tags.length - 3}`}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
          )}
        </Box>

        {/* Metrics */}
        <Box display="flex" alignItems="center" gap={2} mb={1}>
          <Box display="flex" alignItems="center">
            <Rating
              value={protocol.quality_metrics.average_rating}
              precision={0.1}
              size="small"
              readOnly
            />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
              {protocol.quality_metrics.average_rating.toFixed(1)}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            •
          </Typography>
          <Box display="flex" alignItems="center">
            <DownloadIcon sx={{ fontSize: 16, mr: 0.5 }} />
            <Typography variant="body2" color="text.secondary">
              {protocol.quality_metrics.download_count.toLocaleString()}
            </Typography>
          </Box>
        </Box>

        {/* Certification Status */}
        <Chip
          label={certificationBadge.label}
          size="small"
          sx={{
            bgcolor: `${certificationBadge.color}20`,
            color: certificationBadge.color,
            fontWeight: 600,
            borderColor: certificationBadge.color,
            border: '1px solid',
          }}
        />
      </CardContent>

      <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
        {/* Price */}
        <Box>
          {protocol.licensing.license_type === 'free' ? (
            <Typography variant="h6" color="success.main" fontWeight="bold">
              FREE
            </Typography>
          ) : (
            <Box>
              <Typography variant="h6" color="primary.main" fontWeight="bold">
                ${protocol.licensing.base_price?.toFixed(2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {protocol.licensing.price_model === 'subscription' ? '/month' : 'one-time'}
              </Typography>
            </Box>
          )}
        </Box>

        {/* View Details Button */}
        <Button
          variant="contained"
          size="small"
          endIcon={<InfoIcon />}
          onClick={handleClick}
        >
          View Details
        </Button>
      </CardActions>
    </Card>
  );
};

export default ProtocolCard;
