package main

import (
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/yazhsab/qbitel-bridge/go/mgmtapi/internal/devices"
)

var startTime = time.Now()

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	cfg := devices.DefaultLifecycleConfig()
	dlm, err := devices.NewDeviceLifecycleManager(logger, cfg)
	if err != nil {
		logger.Fatal("failed to initialize device lifecycle manager", zap.Error(err))
	}

	router := gin.Default()

	api := router.Group("/v1")
	{
		api.GET("/health", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"status":    "healthy",
				"timestamp": time.Now().Format(time.RFC3339),
			})
		})

		api.GET("/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"service":   "mgmt-api",
				"status":    "running",
				"timestamp": time.Now().Format(time.RFC3339),
				"uptime":    time.Since(startTime).String(),
			})
		})

		// Device collection
        api.GET("/devices", func(c *gin.Context) {
            page, pageSize := parsePagination(c)
            filters := parseDeviceFilters(c)

            orgID := strings.TrimSpace(c.Query("organization_id"))
            var devicesList []*devices.Device
            var err error
            if orgID != "" {
                devicesList, err = dlm.ListDevices(orgID, filters)
            } else {
                devicesList, err = dlm.ListAllDevices(filters)
            }
            if err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }

            devicesList = applySearchFilter(c.Query("search"), devicesList)
			total := len(devicesList)
			start := (page - 1) * pageSize
			if start > total {
				start = total
			}
			end := start + pageSize
			if end > total {
				end = total
			}

			c.JSON(http.StatusOK, gin.H{
				"devices":     devicesList[start:end],
				"total_count": total,
				"page":        page,
				"page_size":   pageSize,
			})
		})

        api.GET("/devices/:device_id", func(c *gin.Context) {
            deviceID := c.Param("device_id")
            device, err := dlm.GetDevice(deviceID)
            if err != nil {
                respondError(c, http.StatusNotFound, err)
                return
            }

            cert, _ := dlm.GetDeviceCertificate(deviceID)
            response := gin.H{
                "device": device,
                "policies": []any{},
                "recent_activity": []any{},
                "alerts": []any{},
                "compliance_violations": []any{},
            }
            if cert != nil {
                response["certificate"] = certificateResponse(cert)
            }

            c.JSON(http.StatusOK, response)
        })

        api.PATCH("/devices/:device_id", func(c *gin.Context) {
            deviceID := c.Param("device_id")
            var updates map[string]interface{}
            if err := c.ShouldBindJSON(&updates); err != nil {
                respondError(c, http.StatusBadRequest, err)
                return
            }
            if err := dlm.UpdateDeviceConfiguration(c.Request.Context(), deviceID, updates); err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }
            device, err := dlm.GetDevice(deviceID)
            if err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"device": device})
        })

        api.POST("/devices/:device_id/suspend", func(c *gin.Context) {
            deviceID := c.Param("device_id")
            var req struct {
                Reason string `json:"reason"`
            }
            if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			if err := dlm.SuspendDevice(c.Request.Context(), deviceID, req.Reason); err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
            device, err := dlm.GetDevice(deviceID)
            if err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"device": device})
        })

		api.POST("/devices/:device_id/resume", func(c *gin.Context) {
			deviceID := c.Param("device_id")
			if err := dlm.ReactivateDevice(c.Request.Context(), deviceID); err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
            device, err := dlm.GetDevice(deviceID)
            if err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"device": device})
        })

		api.POST("/devices/:device_id/decommission", func(c *gin.Context) {
			deviceID := c.Param("device_id")
			var req struct {
				Reason string `json:"reason"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			if err := dlm.DecommissionDevice(c.Request.Context(), deviceID, req.Reason); err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
            device, err := dlm.GetDevice(deviceID)
            if err != nil {
                respondError(c, http.StatusInternalServerError, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"device": device})
        })

		// Enrollment lifecycle
		api.POST("/devices/enrollment/sessions", func(c *gin.Context) {
			var req devices.EnrollmentRequest
			if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			session, err := dlm.StartEnrollment(c.Request.Context(), &req)
			if err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
			c.JSON(http.StatusCreated, session)
		})

		api.GET("/devices/enrollment/sessions/:session_id", func(c *gin.Context) {
			session, err := dlm.GetEnrollmentSession(c.Param("session_id"))
			if err != nil {
				respondError(c, http.StatusNotFound, err)
				return
			}
			c.JSON(http.StatusOK, session)
		})

		api.POST("/devices/enrollment/sessions/:session_id/attestation", func(c *gin.Context) {
			sessionID := c.Param("session_id")
			var attestation devices.AttestationData
			if err := c.ShouldBindJSON(&attestation); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			if err := dlm.SubmitAttestation(c.Request.Context(), sessionID, &attestation); err != nil {
				respondError(c, http.StatusUnauthorized, err)
				return
			}
			session, _ := dlm.GetEnrollmentSession(sessionID)
			c.JSON(http.StatusOK, session)
		})

		api.POST("/devices/enrollment/sessions/:session_id/approve", func(c *gin.Context) {
			device, err := dlm.ApproveEnrollment(c.Request.Context(), c.Param("session_id"))
			if err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			c.JSON(http.StatusOK, device)
		})

		api.POST("/devices/enrollment/sessions/:session_id/reject", func(c *gin.Context) {
			var req struct {
				Reason string `json:"reason"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			if err := dlm.RejectEnrollment(c.Request.Context(), c.Param("session_id"), req.Reason); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			c.Status(http.StatusNoContent)
		})

		api.POST("/devices/enrollment/sessions/:session_id/complete", func(c *gin.Context) {
			sessionID := c.Param("session_id")
			var req struct {
				CSR string `json:"csr"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			csr, err := parseCertificateSigningRequest(req.CSR)
			if err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			cert, err := dlm.CompleteEnrollment(c.Request.Context(), sessionID, csr)
			if err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
			c.JSON(http.StatusOK, certificateResponse(cert))
		})

		// Certificate operations
		api.GET("/devices/:device_id/certificate", func(c *gin.Context) {
			cert, err := dlm.GetDeviceCertificate(c.Param("device_id"))
			if err != nil {
				respondError(c, http.StatusNotFound, err)
				return
			}
			c.JSON(http.StatusOK, certificateResponse(cert))
		})

		api.POST("/devices/:device_id/certificate/renew", func(c *gin.Context) {
			var req struct {
				CSR string `json:"csr"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			csr, err := parseCertificateSigningRequest(req.CSR)
			if err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			cert, err := dlm.RenewDeviceCertificate(c.Request.Context(), c.Param("device_id"), csr)
			if err != nil {
				respondError(c, http.StatusBadRequest, err)
				return
			}
			c.JSON(http.StatusOK, certificateResponse(cert))
		})

        api.POST("/devices/:device_id/certificate/revoke", func(c *gin.Context) {
            var req struct {
                Reason string `json:"reason"`
            }
            if err := c.ShouldBindJSON(&req); err != nil {
                respondError(c, http.StatusBadRequest, err)
                return
            }
            if err := dlm.RevokeDeviceCertificate(c.Request.Context(), c.Param("device_id"), req.Reason); err != nil {
                respondError(c, http.StatusBadRequest, err)
                return
            }
            c.Status(http.StatusNoContent)
        })

        // Placeholder implementations for advanced features
        api.GET("/devices/:device_id/policies", func(c *gin.Context) {
            c.JSON(http.StatusOK, []any{})
        })

        api.PUT("/devices/:device_id/policies", func(c *gin.Context) {
            c.Status(http.StatusAccepted)
        })

        api.POST("/devices/:device_id/policies/:policy_id/deploy", func(c *gin.Context) {
            c.Status(http.StatusAccepted)
        })

        api.POST("/devices/:device_id/compliance/check", func(c *gin.Context) {
            c.Status(http.StatusAccepted)
        })

        api.GET("/devices/:device_id/compliance/violations", func(c *gin.Context) {
            c.JSON(http.StatusOK, []any{})
        })

        api.POST("/devices/:device_id/compliance/violations/:violation_id/resolve", func(c *gin.Context) {
            c.Status(http.StatusNoContent)
        })

        api.POST("/devices/:device_id/health/check", func(c *gin.Context) {
            c.Status(http.StatusAccepted)
        })

        api.GET("/devices/alerts", func(c *gin.Context) {
            c.JSON(http.StatusOK, []any{})
        })

        api.POST("/devices/alerts/:alert_id/acknowledge", func(c *gin.Context) {
            c.Status(http.StatusNoContent)
        })

        api.POST("/devices/alerts/:alert_id/resolve", func(c *gin.Context) {
            c.Status(http.StatusNoContent)
        })

        api.GET("/devices/activity", func(c *gin.Context) {
            c.JSON(http.StatusOK, []any{})
        })

        api.GET("/devices/metrics", func(c *gin.Context) {
            c.JSON(http.StatusOK, gin.H{
                "metrics": gin.H{},
                "timestamp": time.Now().Format(time.RFC3339),
            })
        })

        api.GET("/devices/metrics/history", func(c *gin.Context) {
            c.JSON(http.StatusOK, []any{})
        })

        api.POST("/devices/bulk/update", func(c *gin.Context) {
            var req struct {
                DeviceIDs []string               `json:"device_ids"`
                Updates   map[string]interface{} `json:"updates"`
            }
            if err := c.ShouldBindJSON(&req); err != nil {
                respondError(c, http.StatusBadRequest, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"success": req.DeviceIDs, "failed": []any{}})
        })

        api.POST("/devices/bulk/suspend", func(c *gin.Context) {
            var req struct {
                DeviceIDs []string `json:"device_ids"`
            }
            if err := c.ShouldBindJSON(&req); err != nil {
                respondError(c, http.StatusBadRequest, err)
                return
            }
            c.JSON(http.StatusOK, gin.H{"success": req.DeviceIDs, "failed": []any{}})
        })
    }

	// Backward-compatible health endpoints
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().Format(time.RFC3339),
		})
	})

	router.GET("/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":   "mgmt-api",
			"status":    "running",
			"timestamp": time.Now().Format(time.RFC3339),
			"uptime":    time.Since(startTime).String(),
		})
	})

	addr := getenv("MGMT_ADDR", ":8081")
	logger.Info("mgmt api listening",
		zap.String("addr", addr),
		zap.Bool("tpm_attestation_required", cfg.RequireTPMAttestation),
		zap.Duration("certificate_validity", cfg.CertificateValidity))

	router.Run(addr)
}

func parseCertificateSigningRequest(input string) (*x509.CertificateRequest, error) {
	if strings.TrimSpace(input) == "" {
		return nil, fmt.Errorf("empty CSR")
	}

	if block, _ := pem.Decode([]byte(input)); block != nil {
		if block.Type != "CERTIFICATE REQUEST" && block.Type != "NEW CERTIFICATE REQUEST" {
			return nil, fmt.Errorf("unexpected PEM block type: %s", block.Type)
		}
		return parseCSRBytes(block.Bytes)
	}

	der, err := base64.StdEncoding.DecodeString(input)
	if err != nil {
		return nil, fmt.Errorf("failed to decode CSR: %w", err)
	}
	return parseCSRBytes(der)
}

func parseCSRBytes(der []byte) (*x509.CertificateRequest, error) {
	if len(der) == 0 {
		return nil, fmt.Errorf("CSR data is empty")
	}

	csr, err := x509.ParseCertificateRequest(der)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSR: %w", err)
	}

	if err := csr.CheckSignature(); err != nil {
		return nil, fmt.Errorf("CSR signature verification failed: %w", err)
	}

	return csr, nil
}

func parsePagination(c *gin.Context) (int, int) {
	page := parsePositiveInt(c.DefaultQuery("page", "1"), 1)
	pageSize := parsePositiveInt(c.DefaultQuery("page_size", "50"), 50)
	if pageSize > 200 {
		pageSize = 200
	}
	return page, pageSize
}

func parsePositiveInt(value string, fallback int) int {
	v, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func parseDeviceFilters(c *gin.Context) *devices.DeviceFilters {
	filters := &devices.DeviceFilters{}

	if values := c.QueryArray("status"); len(values) > 0 {
		for _, v := range values {
			filters.Status = append(filters.Status, devices.DeviceStatus(strings.TrimSpace(v)))
		}
	}

	if values := c.QueryArray("device_type"); len(values) > 0 {
		for _, v := range values {
			filters.DeviceType = append(filters.DeviceType, devices.DeviceType(strings.TrimSpace(v)))
		}
	}

	if values := c.QueryArray("manufacturer"); len(values) > 0 {
		filters.Manufacturer = append(filters.Manufacturer, values...)
	}

	if values := c.QueryArray("tags"); len(values) > 0 {
		filters.Tags = append(filters.Tags, values...)
	}

	if values := c.QueryArray("compliance_status"); len(values) > 0 {
		for _, v := range values {
			filters.ComplianceStatus = append(filters.ComplianceStatus, devices.ComplianceStatus(strings.TrimSpace(v)))
		}
	}

	if values := c.QueryArray("health_status"); len(values) > 0 {
		for _, v := range values {
			filters.HealthStatus = append(filters.HealthStatus, devices.HealthStatus(strings.TrimSpace(v)))
		}
	}

	if len(filters.Status) == 0 && len(filters.DeviceType) == 0 && len(filters.Manufacturer) == 0 &&
		len(filters.Tags) == 0 && len(filters.ComplianceStatus) == 0 && len(filters.HealthStatus) == 0 {
		return nil
	}

	return filters
}

func applySearchFilter(term string, devicesList []*devices.Device) []*devices.Device {
	term = strings.TrimSpace(term)
	if term == "" {
		return devicesList
	}
	termLower := strings.ToLower(term)
	var filtered []*devices.Device
	for _, device := range devicesList {
		if strings.Contains(strings.ToLower(device.ID), termLower) ||
			strings.Contains(strings.ToLower(device.Name), termLower) ||
			strings.Contains(strings.ToLower(device.SerialNumber), termLower) {
			filtered = append(filtered, device)
		}
	}
	return filtered
}

func certificateResponse(cert *devices.DeviceCertificate) gin.H {
	encoded := ""
	if len(cert.Certificate) > 0 {
		encoded = base64.StdEncoding.EncodeToString(cert.Certificate)
	}
	return gin.H{
		"id":                cert.ID,
		"device_id":         cert.DeviceID,
		"certificate":       encoded,
		"serial_number":     cert.SerialNumber,
		"subject":           cert.Subject,
		"issuer":            cert.Issuer,
		"not_before":        cert.NotBefore.Format(time.RFC3339),
		"not_after":         cert.NotAfter.Format(time.RFC3339),
		"status":            cert.Status,
		"revoked_at":        timePtrToString(cert.RevokedAt),
		"revocation_reason": cert.RevocationReason,
		"renewed_from":      cert.RenewedFrom,
	}
}

func timePtrToString(t *time.Time) *string {
	if t == nil {
		return nil
	}
	formatted := t.Format(time.RFC3339)
	return &formatted
}

func respondError(c *gin.Context, status int, err error) {
	c.JSON(status, gin.H{"error": err.Error()})
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
